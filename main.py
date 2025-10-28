# Importation
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import numpy as np
from sort import *

# Initialization and variable naming
model = YOLO("yolov8l.pt")
vid = cv.VideoCapture("assets/traffic.mp4")

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age = 22, min_hits = 3, iou_threshold = 0.3)
# We'll create dedicated IN/OUT counting lines (set after reading video size)
line_in = None
line_out = None
prev_centers = {}  # store previous center y for each track id to detect crossing
prev_boxes = {}    # store previous bounding box for each track id
count_up = set()
count_down = set()
total_count = set()
mask = cv.imread("assets/mask.png") # For blocking out noise

# Setting up video writer properties (for saving the output result)
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv.CAP_PROP_FPS)
video_writer = cv.VideoWriter(("result.mp4"), cv.VideoWriter_fourcc("m", "p", "4", "v"),
                              fps, (width, height))

# Define the inbound/outbound counting lines so every vehicle direction is properly captured.
# Percentages are tuned for this camera angle so the triggers sit well above the frame edge.
LINE_IN_PERCENT = 0.555   # Vehicles moving upward (towards downtown bridge) trigger here
LINE_OUT_PERCENT = 0.665  # Vehicles moving downward (towards camera) trigger here
LINE_MARGIN_PX = 12       # Keep line segments away from frame borders
MIN_LINE_GAP_PX = 45      # Ensure we have separation between the two lines
LINE_IN_X_RANGE = (0.055, 0.515)   # Relative (0-1) horizontal span for the inbound segment
LINE_OUT_X_RANGE = (0.51, 0.965)   # Relative (0-1) horizontal span for the outbound segment
LINE_SEGMENT_PADDING = max(24, int(width * 0.03))  # Extra tolerance for center-x/box checks

def clamp_segment(start_px, end_px):
    """Clamp horizontal line segment to the frame and ensure x_start < x_end."""
    start_px = max(LINE_MARGIN_PX, start_px)
    end_px = min(width - LINE_MARGIN_PX, end_px)
    if end_px <= start_px:
        end_px = min(width - LINE_MARGIN_PX, start_px + 2 * LINE_SEGMENT_PADDING)
    return start_px, end_px

line_in_y = int(height * LINE_IN_PERCENT)
line_out_y = int(height * LINE_OUT_PERCENT)
line_in_y = max(LINE_MARGIN_PX, min(line_in_y, height - LINE_MARGIN_PX))
line_out_y = max(LINE_MARGIN_PX, min(line_out_y, height - LINE_MARGIN_PX))

if line_out_y - line_in_y < MIN_LINE_GAP_PX:
    line_in_y = max(LINE_MARGIN_PX, line_out_y - MIN_LINE_GAP_PX)

line_in_x1, line_in_x2 = clamp_segment(int(width * LINE_IN_X_RANGE[0]), int(width * LINE_IN_X_RANGE[1]))
line_out_x1, line_out_x2 = clamp_segment(int(width * LINE_OUT_X_RANGE[0]), int(width * LINE_OUT_X_RANGE[1]))

line_in = {"x_start": line_in_x1, "x_end": line_in_x2, "y": line_in_y}
line_out = {"x_start": line_out_x1, "x_end": line_out_x2, "y": line_out_y}
LINE_IN_COLOR = (0, 200, 0)    # BGR green
LINE_OUT_COLOR = (0, 0, 255)   # BGR red
LINE_HIT_COLOR = (0, 255, 255) # BGR yellow for visual confirmation
print(
    f"Counting line segments => "
    f"IN(y={line_in_y}, x_range=({line_in['x_start']},{line_in['x_end']})), "
    f"OUT(y={line_out_y}, x_range=({line_out['x_start']},{line_out['x_end']}))"
)

def draw_count_line(frame, line, color, thickness=3):
    """Draw a horizontal counting segment."""
    cv.line(frame, (line["x_start"], line["y"]), (line["x_end"], line["y"]), color, thickness=thickness)

def center_within_segment(cx, line, padding=0):
    """Check if the object's center-x lies over the active line segment (with optional padding)."""
    return (line["x_start"] - padding) <= cx <= (line["x_end"] + padding)

def segment_overlap(x1, x2, line, padding=0):
    """Return True if any portion of the bounding box horizontally overlaps the line segment."""
    seg_start = line["x_start"] - padding
    seg_end = line["x_end"] + padding
    return not (x2 < seg_start or x1 > seg_end)

def bbox_straddles_line(y1, y2, line_y):
    """Return True if the vertical span of a bounding box crosses the line."""
    return y1 <= line_y <= y2

# Ensure mask matches video frame size to avoid size-mismatch errors
if mask is not None:
    try:
        mask = cv.resize(mask, (width, height), interpolation=cv.INTER_AREA)
    except Exception as e:
        print(f"Warning: failed to resize mask: {e}")

while True:
    ref, frame = vid.read()
    # Ensure mask matches current frame size; resize on-the-fly if necessary
    if frame is None:
        break
    if mask is None:
        frame_region = frame
    else:
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_AREA)
        frame_region = cv.bitwise_and(frame, mask)
    result = model(frame_region, stream=True)

    # Total count graphics
    frame_graphics = cv.imread("assets/graphics.png", cv.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, frame_graphics, (0,0))

    # Vehicle count graphics
    frame_graphics1 = cv.imread("assets/graphics1.png", cv.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, frame_graphics1, (420,0))

    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2-x1), (y2-y1)

            #Detection confidence
            conf = math.floor(box.conf[0]*100)/100

            # Class names
            cls = int(box.cls[0])
            vehicle_names = class_names[cls]

            if vehicle_names == "car" or vehicle_names == "truck" or vehicle_names == "bus"\
                or vehicle_names == "motorbike":
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))

    # Tracking codes
    tracker_updates = tracker.update(detections)
    # Draw counting lines for inbound (green) and outbound (red) directions
    draw_count_line(frame, line_in, LINE_IN_COLOR, thickness=4)
    draw_count_line(frame, line_out, LINE_OUT_COLOR, thickness=4)

    # Geting bounding boxes points and vehicle ID
    for update in tracker_updates:
        x1, y1, x2, y2, id = update
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = (x2-x1), (y2-y1)

        # Getting tracking marker
        cx, cy = (x1+w//2), (y1+h//2)
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # Crossing-based counting using the IN/OUT lines
        prev_cy = prev_centers.get(id, None)
        within_out_segment = (
            center_within_segment(cx, line_out, LINE_SEGMENT_PADDING)
            or segment_overlap(x1, x2, line_out, LINE_SEGMENT_PADDING)
        )
        within_in_segment = (
            center_within_segment(cx, line_in, LINE_SEGMENT_PADDING)
            or segment_overlap(x1, x2, line_in, LINE_SEGMENT_PADDING)
        )

        # If we have a previous center, check for crossings on each counting line
        if prev_cy is not None:
            if within_out_segment:
                crossed_out = (
                    (prev_cy < line_out["y"] <= cy)
                    or (cy >= line_out["y"] and bbox_straddles_line(y1, y2, line_out["y"]))
                )
                if crossed_out:
                    if id not in total_count:
                        total_count.add(id)
                    if id not in count_down:
                        count_down.add(id)
                    draw_count_line(frame, line_out, LINE_HIT_COLOR, thickness=6)
                    print(f"Counted ID {id} direction=OUT total={len(total_count)}")

            if within_in_segment:
                crossed_in = (
                    (prev_cy > line_in["y"] >= cy)
                    or (cy <= line_in["y"] and bbox_straddles_line(y1, y2, line_in["y"]))
                )
                if crossed_in:
                    if id not in total_count:
                        total_count.add(id)
                    if id not in count_up:
                        count_up.add(id)
                    draw_count_line(frame, line_in, LINE_HIT_COLOR, thickness=6)
                    print(f"Counted ID {id} direction=IN total={len(total_count)}")

        # Update previous center/bounding box for this id
        prev_centers[id] = cy
        prev_boxes[id] = (x1, y1, x2, y2)

        # Adding rectangles and texts
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f'{id}', (x1, y1), scale=1, thickness=2)

    # Adding texts to graphics
    cv.putText(frame, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
    cv.putText(frame, str(len(count_up)), (600, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
    cv.putText(frame, str(len(count_down)), (850, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)

    cv.imshow("vid", frame)

    # Saving the video frame output
    video_writer.write(frame)

    cv.waitKey(1)

# Closing down everything
vid.release()
cv.destroyAllWindows()
video_writer.release()
