# Homework versions

Latest user instruction: maintain everything in `hw1/`; do not create a separate homework project.

- `hw1.ipynb` is the autonomous original version 1. Preserve its embedded original equations/calibration; its results live in `results_v1/`.
- `hw1_version_2.ipynb` is the current version 2. All `.py` modules and tests here belong to v2; synchronize its embedded modules after code changes.
- `results_v2/` holds the latest v2 results; new GPU runs go under `results_v2/runs/<RUN_ID>/`. Preserve provenance and raw measurements.
- Explain both versions, assumptions, validation, and assignment deliverables in README. Do not calibrate memory parameters or change the required network/backend flags.
- Verify with `python -m unittest discover -s hw1/tests -v` and notebook validation. Version 1 must be checked against its own embedded equations, not the current modules.
