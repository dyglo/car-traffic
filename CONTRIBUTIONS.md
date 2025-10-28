# Contributions Guide

Thanks for your interest in improving the Car Traffic Counter! Follow the steps below to get your changes merged smoothly.

## 1. Fork & Clone
1. Fork [`dyglo/car-traffic`](https://github.com/dyglo/car-traffic) to your GitHub account.
2. Clone your fork and add the upstream remote:
   ```bash
   git clone https://github.com/<your-username>/car-traffic.git
   cd car-traffic
   git remote add upstream https://github.com/dyglo/car-traffic.git
   ```

## 2. Create a Feature Branch
Keep `main` clean and branch for every change:
```bash
git checkout -b feature/<short-topic>
```

## 3. Install Tooling
Use a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Development Checklist
- ✅ Run `python -m py_compile main.py` (or your linter of choice) before committing.
- ✅ Update documentation when behavior, inputs, or outputs change.
- ✅ Add/adjust masks, sample footage, or test scripts as needed to demonstrate new features.
- 🚫 **Do not commit large binary artifacts** (`*.pt`, `*.onnx`, long recordings, generated `result.mp4`, etc.). These files are gitignored; use releases or cloud storage instead.
- 📸 When tweaking trigger lines, capture a screenshot or short clip for reviewers to verify alignment.

## 5. Commit & Push
Use clear, conventional commits:
```bash
git commit -m "feat: describe your change"
git push origin feature/<short-topic>
```

## 6. Open a Pull Request
1. Rebase on `upstream/main` if needed.
2. Open a PR against `dyglo/car-traffic:main`.
3. Fill out the PR template (or describe):
   - What changed
   - Why it matters
   - Screenshots / clips for visual tweaks
   - Testing performed

Reviewers will triage within a few business days. Be ready to discuss implementation and respond to feedback.

## 7. Community Standards
- Keep discussions respectful and focused on the solution.
- Prefer descriptive variable names and small, reviewable commits.
- Reference issues in commits/PRs using `Fixes #123` when applicable.

Happy tracking!🚗
