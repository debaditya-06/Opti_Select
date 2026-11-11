# OptiSelect

OptiSelect is a feature-selection benchmarking application for Group 1 (Optimization). It compares classification models trained on all available features against automatically selected subsets produced by filter (`SelectKBest` with ANOVA F-scores) and wrapper (`RFE` with Logistic Regression) methods across five datasets.

## Project Structure

```
optiselect/
├── app.py                  # Streamlit dashboard calling experiments.py
├── experiments.py          # Single source of truth for datasets, pipelines, CV, and figures
├── server/
│   ├── __init__.py
│   └── main.py             # FastAPI server, API routes, results/ static files, and SPA fallback
├── web/                    # React + TypeScript + Vite frontend
│   ├── public/             # Favicon assets
│   ├── src/                # Pages, layout, hooks, and CSS tokens
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── data/                   # Downloaded UCI dataset files
├── results/                # Generated PNG figures, final_table.csv, and experiment cache
├── tests/
│   ├── test_api.py         # FastAPI endpoint and validation tests
│   └── test_experiments.py # Leakage-prevention, bounds, baseline, and reproducibility tests
├── requirements.txt        # Pinned Python dependencies
└── README.md
```

## Installation

Create a Python 3.11+ virtual environment, install Python dependencies, and build the frontend:

```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

cd web
npm ci
npm run build
cd ..
```

## Running the Application

Start the FastAPI web application (serves both the API and the built React frontend) with one command from the project root:

```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in a browser.

To run the companion Streamlit dashboard instead:

```bash
streamlit run app.py
```

## Rebuilding Results & Figures

Generate `results/final_table.csv` and per-dataset exploratory figures by running:

```bash
python experiments.py --all
```

Run a single configuration from the command line and generate the five mandatory figures:

```bash
python experiments.py --dataset iris --classifier logistic_regression --method selectkbest --k 3 --figures
```

Run the automated test suite:

```bash
pytest -v
```

## Modelling Protocol

- Deterministic seed `random_state=42` across splits, cross-validation folds, and estimators.
- Stratified 80/20 train/test split; the held-out test set is evaluated once at the end and never used for feature selection or hyperparameter choices.
- Preprocessing and selection live inside a single `sklearn.pipeline.Pipeline`: `SimpleImputer(strategy="median") -> StandardScaler() -> Selector(k) -> Classifier`.
- Cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training split only.
- Supported classifiers: Logistic Regression (`max_iter=2000`), Decision Tree, and Random Forest.

## Dataset Sources & Documented Targets

1. **Iris** (4 features, 3 classes): Loaded via `sklearn.datasets.load_iris()`.
2. **Wine** (13 features, 3 classes): Loaded via `sklearn.datasets.load_wine()`.
3. **Breast Cancer Wisconsin** (30 features, 2 classes): Loaded via `sklearn.datasets.load_breast_cancer()`.
4. **Heart Disease — UCI Cleveland** (13 features, binary target): Downloaded into `data/heart_disease_cleveland.csv` from the UCI Machine Learning Repository. Missing `?` entries are parsed as `NaN` and imputed with the training-split median inside the pipeline. The target `num` is binarized as `num > 0` (`disease`) versus `num == 0` (`no disease`).
5. **Wine Quality — UCI Red Wine** (11 features, binary target): Downloaded into `data/winequality-red.csv` from the UCI Machine Learning Repository. The ordinal `quality` score is binarized as `quality >= 6` (`good`) versus `quality < 6` (`not good`).

## Dataset Credits & Licences

- **Iris, Wine, and Breast Cancer Wisconsin**: Bundled in `scikit-learn` (BSD 3-Clause), originally sourced from the UCI Machine Learning Repository.
- **Heart Disease (Cleveland)**: Detrano et al. (1989), UCI Machine Learning Repository (CC BY 4.0).
- **Wine Quality (Red)**: Cortez et al. (2009), *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, UCI Machine Learning Repository (CC BY 4.0).
