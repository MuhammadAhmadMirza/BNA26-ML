# BNA26 — Machine Learning (Round 2)

Materials and notebooks for the BNA26 Computer Science Module, Round 2: Machine Learning. This repo is structured so delegates can clone or download a zip of the repository and get all the material they need.

## Project Structure

```text
LICENSE
README.md
requirements.txt
data/
  put dataset here
results/
  linear_regression/
  polynomial_regression/
  random_forest_regression/
  ... other algorithms
src/
  data_analysis.ipynb
  models.ipynb
```

## Setup

Prerequisites: Python 3.11+ and pip.

For isolated enviorment either use conda (recommended) or pyenv

## Data

- Place required datasets under `data/`
- Notebooks expect files relative to the repo root; adjust paths if needed.

## Results

Model outputs and artifacts are written under `results/` in model-specific folders. They will contain the `.joblib` file and the parameters / metics in a `.json`:

- `results/linear_regression/`
- `results/polynomial_regression/`
- `results/random_forest_regression/`
- new models and their directories to be added

## TODO

1. Finalize dataset selection and format
2. Select baseline and comparative models
3. Complete a working prototype for each model
4. Prepare a tweaked/augmented dataset for the round

## License

See `LICENSE` for details.
