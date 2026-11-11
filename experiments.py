"""
experiments.py — Single source of truth for all OptiSelect modelling.

Loads datasets, builds scikit-learn Pipelines with proper train-only fitting,
runs cross-validated experiments and final test evaluations, generates all
mandatory figures, and produces the final comparison table.

Usage:
    python experiments.py --all          # run every combination, write results/final_table.csv
    python experiments.py --dataset iris --classifier logistic_regression --method selectkbest --k 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scipy.optimize import OptimizeWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / ".cache"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Chart styling (no purple, no gradients)
# ---------------------------------------------------------------------------
NAVY = "#12232E"
TEAL = "#0E9F8E"
AMBER = "#F29E4C"
SLATE = "#5C7480"
SURFACE = "#EAF4F4"
WHITE = "#FFFFFF"

CHART_COLORS = [TEAL, AMBER, NAVY, "#2B7A78", "#D97236"]

plt.rcParams.update({
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "axes.edgecolor": NAVY,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": SLATE,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.16,
})

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
DATASET_META: dict[str, dict[str, Any]] = {
    "iris": {
        "name": "Iris",
        "slug": "iris",
        "source": "scikit-learn load_iris()",
        "description": (
            "Fisher's classic Iris dataset with measurements of sepal and petal "
            "dimensions for 150 samples across three species: setosa, versicolor "
            "and virginica."
        ),
        "target_doc": "Three species: setosa (0), versicolor (1), virginica (2).",
        "preprocessing": "No missing values. Median imputation and standard scaling fitted on training split.",
        "example_sizes": [2, 3, 4],
        "n_features": 4,
    },
    "wine": {
        "name": "Wine",
        "slug": "wine",
        "source": "scikit-learn load_wine()",
        "description": (
            "Results of a chemical analysis of wines grown in the same region of "
            "Italy from three different cultivars. 13 constituents measured across "
            "178 samples."
        ),
        "target_doc": "Three cultivars: class_0 (0), class_1 (1), class_2 (2).",
        "preprocessing": "No missing values. Median imputation and standard scaling fitted on training split.",
        "example_sizes": [5, 7, 10, 13],
        "n_features": 13,
    },
    "breast-cancer": {
        "name": "Breast Cancer Wisconsin",
        "slug": "breast-cancer",
        "source": "scikit-learn load_breast_cancer()",
        "description": (
            "Diagnostic dataset with 30 features computed from digitized images of "
            "fine needle aspirates of breast masses. 569 samples classified as "
            "malignant or benign."
        ),
        "target_doc": "Binary: malignant (0), benign (1).",
        "preprocessing": "No missing values. Median imputation and standard scaling fitted on training split.",
        "example_sizes": [5, 10, 15, 30],
        "n_features": 30,
    },
    "heart-disease": {
        "name": "Heart Disease (UCI Cleveland)",
        "slug": "heart-disease",
        "source": "UCI Machine Learning Repository — Cleveland subset",
        "description": (
            "The Cleveland Heart Disease dataset from the UCI repository with 303 "
            "instances and 13 commonly used clinical features including age, sex, "
            "chest pain type, resting blood pressure and cholesterol."
        ),
        "target_doc": (
            "Original target has values 0-4. Binarized: num > 0 is 'disease' (1), "
            "num == 0 is 'no disease' (0)."
        ),
        "preprocessing": (
            "The '?' character is treated as missing. Missing values are imputed "
            "with the median inside the Pipeline, fitted only on training data."
        ),
        "example_sizes": [5, 8, 13],
        "n_features": 13,
    },
    "wine-quality": {
        "name": "Wine Quality (Red)",
        "slug": "wine-quality",
        "source": "UCI Machine Learning Repository — Red Wine Quality",
        "description": (
            "Physicochemical properties of 1599 red wine samples from Portugal. "
            "11 input features including fixed acidity, volatile acidity, citric "
            "acid, residual sugar, chlorides and alcohol content."
        ),
        "target_doc": (
            "Original quality score 0-10. Binarized: quality >= 6 is 'good' (1), "
            "otherwise 'not good' (0)."
        ),
        "preprocessing": "No missing values expected. Median imputation and standard scaling fitted on training split.",
        "example_sizes": [4, 6, 8, 11],
        "n_features": 11,
    },
}

CLASSIFIERS = {
    "logistic_regression": ("Logistic Regression", LogisticRegression),
    "decision_tree": ("Decision Tree", DecisionTreeClassifier),
    "random_forest": ("Random Forest", RandomForestClassifier),
}

METHODS = {
    "selectkbest": "SelectKBest",
    "rfe": "RFE",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    """Download a file with retries."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OptiSelect/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return
        except (urllib.error.URLError, OSError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts: {exc}"
                ) from exc
            time.sleep(2 ** attempt)


def load_dataset(slug: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Load a dataset and return (X, y, feature_names, class_names).

    For Heart Disease: treats '?' as NaN, uses 13 standard features, binarizes target.
    For Wine Quality: binarizes quality >= 6 as 'good'.
    """
    if slug == "iris":
        data = load_iris(as_frame=True)
        return data.data, data.target, list(data.feature_names), list(data.target_names)

    if slug == "wine":
        data = load_wine(as_frame=True)
        return data.data, data.target, list(data.feature_names), list(data.target_names)

    if slug == "breast-cancer":
        data = load_breast_cancer(as_frame=True)
        return data.data, data.target, list(data.feature_names), list(data.target_names)

    if slug == "heart-disease":
        csv_path = DATA_DIR / "heart_disease_cleveland.csv"
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            csv_path,
        )
        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
        ]
        df = pd.read_csv(csv_path, names=columns, na_values="?")
        X = df.drop(columns=["num"]).apply(pd.to_numeric, errors="coerce")
        y = (df["num"] > 0).astype(int)
        feature_names = list(X.columns)
        class_names = ["no disease", "disease"]
        return X, pd.Series(y, name="target"), feature_names, class_names

    if slug == "wine-quality":
        csv_path = DATA_DIR / "winequality-red.csv"
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            csv_path,
        )
        df = pd.read_csv(csv_path, sep=";")
        X = df.drop(columns=["quality"])
        y = (df["quality"] >= 6).astype(int)
        feature_names = list(X.columns)
        class_names = ["not good", "good"]
        return X, pd.Series(y, name="target"), feature_names, class_names

    raise ValueError(f"Unknown dataset: {slug}")


def get_dataset_info(slug: str) -> dict[str, Any]:
    """Return metadata and summary statistics for a dataset."""
    meta = DATASET_META[slug].copy()
    X, y, feature_names, class_names = load_dataset(slug)
    class_counts = y.value_counts().sort_index().to_dict()
    named_counts: dict[str, int] = {}
    for idx, name in enumerate(class_names):
        if idx in class_counts:
            named_counts[name] = int(class_counts[idx])

    preview_df = X.head(6).round(3)
    preview_rows = []
    for row_idx, row in preview_df.iterrows():
        rec = {
            col: (None if pd.isna(row[col]) else float(row[col]))
            for col in feature_names
        }
        t_val = int(y.iloc[preview_df.index.get_loc(row_idx)])
        rec["_target"] = class_names[t_val] if t_val < len(class_names) else str(t_val)
        preview_rows.append(rec)

    meta.update({
        "rows": int(len(X)),
        "features": int(len(feature_names)),
        "n_classes": int(len(class_names)),
        "feature_names": feature_names,
        "class_names": class_names,
        "class_counts": named_counts,
        "missing_values": int(X.isnull().sum().sum()),
        "preview": preview_rows,
    })
    return meta


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(params: dict) -> str:
    """Deterministic cache key from experiment parameters."""
    canonical = json.dumps(params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _cache_get(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _cache_set(key: str, data: dict) -> None:
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def _build_pipeline(
    classifier_key: str,
    method_key: str | None,
    k: int | None,
) -> Pipeline:
    """Build a Pipeline with imputer -> scaler -> [selector] -> classifier."""
    _, clf_class = CLASSIFIERS[classifier_key]
    if classifier_key == "logistic_regression":
        clf = clf_class(max_iter=2000, random_state=42)
    else:
        clf = clf_class(random_state=42)

    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if method_key and k is not None:
        if method_key == "selectkbest":
            steps.append(("selector", SelectKBest(score_func=f_classif, k=k)))
        elif method_key == "rfe":
            rfe_estimator = LogisticRegression(max_iter=2000, random_state=42)
            steps.append(("selector", RFE(estimator=rfe_estimator, n_features_to_select=k)))
        else:
            raise ValueError(f"Unknown method: {method_key}")

    steps.append(("model", clf))
    return Pipeline(steps)


def run_experiment(
    dataset_slug: str,
    classifier_key: str,
    method_key: str,
    k: int,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Run a single experiment: selected features with given k, plus the
    all-features baseline. Returns dict with all metrics.
    """
    if dataset_slug not in DATASET_META:
        raise ValueError(f"Unknown dataset: {dataset_slug}")
    if classifier_key not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {classifier_key}")
    if method_key not in METHODS:
        raise ValueError(f"Unknown method: {method_key}")

    meta = DATASET_META[dataset_slug]
    n_features = meta["n_features"]

    if k < 2 or k > n_features:
        raise ValueError(
            f"k must be between 2 and {n_features} for {meta['name']}, got {k}"
        )

    params = {
        "dataset": dataset_slug,
        "classifier": classifier_key,
        "method": method_key,
        "k": k,
    }
    cache_key = _cache_key(params)
    if use_cache:
        cached = _cache_get(cache_key)
        if cached:
            return cached

    X, y, feature_names, class_names = load_dataset(dataset_slug)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Baseline (all features) ---
    baseline_pipe = _build_pipeline(classifier_key, None, None)
    t0 = time.perf_counter()
    baseline_cv = cross_val_score(baseline_pipe, X_train, y_train, cv=cv, scoring="accuracy")
    baseline_pipe.fit(X_train, y_train)
    baseline_time = time.perf_counter() - t0
    baseline_pred = baseline_pipe.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred, average="macro")

    # --- Selected features ---
    selected_pipe = _build_pipeline(classifier_key, method_key, k)
    t0 = time.perf_counter()
    selected_cv = cross_val_score(selected_pipe, X_train, y_train, cv=cv, scoring="accuracy")
    selected_pipe.fit(X_train, y_train)
    selected_time = time.perf_counter() - t0
    selected_pred = selected_pipe.predict(X_test)
    selected_acc = accuracy_score(y_test, selected_pred)
    selected_f1 = f1_score(y_test, selected_pred, average="macro")

    # Selected feature names
    selector = selected_pipe.named_steps["selector"]
    mask = selector.get_support()
    selected_feature_names = [feature_names[i] for i, m in enumerate(mask) if m]

    # Feature scores/rankings
    if method_key == "selectkbest":
        raw_scores = np.nan_to_num(selector.scores_, nan=0.0)
        score_label = "ANOVA F-score"
        feature_scores = {
            feature_names[i]: float(raw_scores[i]) for i in range(len(feature_names))
        }
    else:
        raw_ranking = selector.ranking_
        max_rank = int(max(raw_ranking))
        score_label = "Reversed RFE rank (higher = selected earlier)"
        feature_scores = {
            feature_names[i]: float(max_rank - raw_ranking[i] + 1)
            for i in range(len(feature_names))
        }

    cm = confusion_matrix(y_test, selected_pred).tolist()

    result = {
        "params": params,
        "dataset_name": meta["name"],
        "classifier_name": CLASSIFIERS[classifier_key][0],
        "method_name": METHODS[method_key],
        "n_features_total": n_features,
        "feature_names": feature_names,
        "class_names": class_names,
        "baseline": {
            "test_accuracy": round(float(baseline_acc), 4),
            "test_macro_f1": round(float(baseline_f1), 4),
            "cv_accuracy_mean": round(float(baseline_cv.mean()), 4),
            "cv_accuracy_std": round(float(baseline_cv.std()), 4),
            "training_time": round(baseline_time, 4),
            "n_features": n_features,
        },
        "selected": {
            "test_accuracy": round(float(selected_acc), 4),
            "test_macro_f1": round(float(selected_f1), 4),
            "cv_accuracy_mean": round(float(selected_cv.mean()), 4),
            "cv_accuracy_std": round(float(selected_cv.std()), 4),
            "training_time": round(selected_time, 4),
            "n_features": k,
            "selected_feature_names": selected_feature_names,
        },
        "feature_scores": feature_scores,
        "score_label": score_label,
        "confusion_matrix": cm,
    }

    if use_cache:
        _cache_set(cache_key, result)

    return result


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _safe_filename(*parts: str) -> str:
    """Create a safe filename from parts."""
    return "_".join(
        p.lower().replace(" ", "-").replace("(", "").replace(")", "")
        for p in parts
    )


def _save_figure(fig: plt.Figure, fname: str, alt: str) -> dict[str, Any]:
    """Save figure to results/ and return metadata including exact pixel size."""
    path = RESULTS_DIR / fname
    fig.savefig(path)
    plt.close(fig)
    # Optimize PNG size and read exact intrinsic dimensions
    with Image.open(path) as img:
        width, height = img.size
    return {
        "filename": fname,
        "url": f"/results/{fname}",
        "width": width,
        "height": height,
        "alt": alt,
    }


def generate_accuracy_vs_features(
    dataset_slug: str,
    classifier_key: str,
    method_key: str,
    extra_k: int | None = None,
) -> dict[str, Any]:
    """Figure 1: Accuracy vs number of features, one line per selection method."""
    meta = DATASET_META[dataset_slug]
    sizes = list(meta["example_sizes"])
    if extra_k and extra_k not in sizes:
        sizes.append(extra_k)
        sizes.sort()

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    clf_name = CLASSIFIERS[classifier_key][0]
    active_method_name = METHODS[method_key]

    best_summary: dict[str, tuple[int, float]] = {}
    for i, m_key in enumerate(METHODS):
        accs = []
        for k_val in sizes:
            if k_val > meta["n_features"]:
                continue
            res = run_experiment(dataset_slug, classifier_key, m_key, k_val)
            accs.append((k_val, res["selected"]["cv_accuracy_mean"]))
        if accs:
            ks, vs = zip(*accs)
            lw = 2.6 if m_key == method_key else 1.8
            ls = "-" if m_key == method_key else "--"
            ax.plot(
                ks, vs, marker="o", color=CHART_COLORS[i],
                label=METHODS[m_key], linewidth=lw, linestyle=ls, markersize=6,
            )
            best_k, best_acc = max(accs, key=lambda x: x[1])
            best_summary[METHODS[m_key]] = (best_k, best_acc)

    ax.set_xlabel("Number of selected features (k)")
    ax.set_ylabel("5-fold CV accuracy (mean)")
    ax.set_title(
        f"Accuracy vs number of features ({active_method_name} highlighted)\n"
        f"Dataset: {meta['name']} | Model: {clf_name} | Methods: SelectKBest vs RFE"
    )
    ax.legend(frameon=True, facecolor=WHITE, edgecolor=SLATE)
    ax.set_xticks(sizes)

    parts = [f"{m} peaks at k={k} ({acc:.3f})" for m, (k, acc) in best_summary.items()]
    alt = (
        f"Line chart of cross-validated accuracy against number of features for "
        f"{meta['name']} with {clf_name}; {'; '.join(parts)}."
    )
    fname = _safe_filename("fig1_accuracy_vs_features", dataset_slug, classifier_key, method_key, str(extra_k or "")) + ".png"
    return _save_figure(fig, fname, alt)


def generate_baseline_vs_optimized(result: dict) -> dict[str, Any]:
    """Figure 2: Grouped bars for accuracy and macro-F1."""
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    metrics = ["Test Accuracy", "Test Macro-F1"]
    baseline_vals = [result["baseline"]["test_accuracy"], result["baseline"]["test_macro_f1"]]
    selected_vals = [result["selected"]["test_accuracy"], result["selected"]["test_macro_f1"]]

    x = np.arange(len(metrics))
    w = 0.32
    bars1 = ax.bar(
        x - w / 2, baseline_vals, w,
        label=f"All features ({result['baseline']['n_features']})", color=NAVY,
    )
    bars2 = ax.bar(
        x + w / 2, selected_vals, w,
        label=f"{result['method_name']} (k={result['selected']['n_features']})", color=TEAL,
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("Evaluation metric")
    ax.set_title(
        f"Baseline vs optimized subset\n"
        f"Dataset: {result['dataset_name']} | Model: {result['classifier_name']} | Method: {result['method_name']}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True, facecolor=WHITE, edgecolor=SLATE)
    ax.set_ylim(0, 1.12)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9,
            )

    alt = (
        f"Grouped bar chart comparing all {result['baseline']['n_features']} features "
        f"(accuracy {baseline_vals[0]:.3f}, macro-F1 {baseline_vals[1]:.3f}) against "
        f"{result['method_name']} with {result['selected']['n_features']} features "
        f"(accuracy {selected_vals[0]:.3f}, macro-F1 {selected_vals[1]:.3f}) on "
        f"{result['dataset_name']} using {result['classifier_name']}."
    )
    fname = _safe_filename(
        "fig2_baseline_vs_optimized",
        result["params"]["dataset"],
        result["params"]["classifier"],
        result["params"]["method"],
        str(result["params"]["k"]),
    ) + ".png"
    return _save_figure(fig, fname, alt)


def generate_feature_importance(result: dict) -> dict[str, Any]:
    """Figure 3: Horizontal bars of feature scores."""
    scores = result["feature_scores"]
    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=False)
    names, vals = zip(*sorted_feats)
    selected_set = set(result["selected"]["selected_feature_names"])

    fig_height = max(3.8, min(9.5, len(names) * 0.30))
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    colors = [TEAL if n in selected_set else "#C4D4D4" for n in names]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(result["score_label"])
    ax.set_ylabel("Feature")
    ax.set_title(
        f"Feature importance and selection scores\n"
        f"Dataset: {result['dataset_name']} | Model: {result['classifier_name']} | "
        f"Method: {result['method_name']} (k={result['params']['k']})"
    )
    legend_handles = [
        Patch(facecolor=TEAL, label=f"Selected (top {result['params']['k']})"),
        Patch(facecolor="#C4D4D4", label="Not selected"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor=WHITE, edgecolor=SLATE)

    top_names = ", ".join(result["selected"]["selected_feature_names"][:3])
    alt = (
        f"Horizontal bar chart of {result['score_label']} for {result['dataset_name']} "
        f"with {result['classifier_name']} using {result['method_name']}; "
        f"selected {result['params']['k']} features including {top_names}."
    )
    fname = _safe_filename(
        "fig3_feature_importance",
        result["params"]["dataset"],
        result["params"]["classifier"],
        result["params"]["method"],
        str(result["params"]["k"]),
    ) + ".png"
    return _save_figure(fig, fname, alt)


def generate_confusion_matrix(result: dict) -> dict[str, Any]:
    """Figure 4: Confusion matrix heatmap on held-out test predictions."""
    cm = np.array(result["confusion_matrix"])
    class_names = result["class_names"]

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor=WHITE,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Held-out test confusion matrix\n"
        f"Dataset: {result['dataset_name']} | Model: {result['classifier_name']} | "
        f"Method: {result['method_name']} (k={result['params']['k']})"
    )

    total = int(cm.sum())
    correct = int(np.trace(cm))
    alt = (
        f"Confusion matrix heatmap on held-out test split for {result['dataset_name']} "
        f"using {result['classifier_name']} and {result['method_name']} (k={result['params']['k']}): "
        f"{correct} of {total} test samples classified correctly."
    )
    fname = _safe_filename(
        "fig4_confusion_matrix",
        result["params"]["dataset"],
        result["params"]["classifier"],
        result["params"]["method"],
        str(result["params"]["k"]),
    ) + ".png"
    return _save_figure(fig, fname, alt)


def generate_accuracy_vs_runtime(
    dataset_slug: str,
    classifier_key: str,
    method_key: str,
    k: int,
) -> dict[str, Any]:
    """Figure 5: Accuracy vs runtime scatter, labelled by method."""
    meta = DATASET_META[dataset_slug]
    clf_name = CLASSIFIERS[classifier_key][0]
    active_method_name = METHODS[method_key]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    res_skb = run_experiment(dataset_slug, classifier_key, "selectkbest", k)
    base_time = res_skb["baseline"]["training_time"]
    base_acc = res_skb["baseline"]["cv_accuracy_mean"]
    ax.scatter([base_time], [base_acc], s=110, color=NAVY, zorder=5, label=f"All features ({meta['n_features']})")
    ax.annotate(
        f"All ({meta['n_features']})", (base_time, base_acc),
        textcoords="offset points", xytext=(8, 4), fontsize=8,
    )

    offsets = [(8, 10), (8, -14)]
    points_summary = [f"All features {base_acc:.3f} in {base_time:.3f}s"]
    for i, m_key in enumerate(METHODS):
        res = run_experiment(dataset_slug, classifier_key, m_key, k)
        t = res["selected"]["training_time"]
        a = res["selected"]["cv_accuracy_mean"]
        ax.scatter([t], [a], s=110, color=CHART_COLORS[i], zorder=5, label=f"{METHODS[m_key]} (k={k})")
        ax.annotate(
            f"{METHODS[m_key]} (k={k})", (t, a),
            textcoords="offset points", xytext=offsets[i], fontsize=8,
        )
        points_summary.append(f"{METHODS[m_key]} {a:.3f} in {t:.3f}s")

    # Add slight x/y padding so annotations fit inside plot area
    ax.margins(x=0.22, y=0.2)
    ax.set_xlabel("Training and CV time (seconds)")
    ax.set_ylabel("5-fold CV accuracy (mean)")
    ax.set_title(
        f"Accuracy vs runtime ({active_method_name} selected, k={k})\n"
        f"Dataset: {meta['name']} | Model: {clf_name} | Methods: All, SelectKBest, RFE"
    )
    ax.legend(frameon=True, facecolor=WHITE, edgecolor=SLATE)

    alt = (
        f"Scatter plot of cross-validated accuracy against runtime for {meta['name']} "
        f"with {clf_name} (k={k}): {'; '.join(points_summary)}."
    )
    fname = _safe_filename(
        "fig5_accuracy_vs_runtime", dataset_slug, classifier_key, method_key, str(k)
    ) + ".png"
    return _save_figure(fig, fname, alt)


def generate_class_distribution(dataset_slug: str) -> dict[str, Any]:
    """Per-dataset class distribution bar chart."""
    meta = DATASET_META[dataset_slug]
    _, y, _, class_names = load_dataset(dataset_slug)
    counts = y.value_counts().sort_index()
    labels = [class_names[i] if i < len(class_names) else str(i) for i in counts.index]
    values = [int(v) for v in counts.values]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(labels, values, color=CHART_COLORS[:len(counts)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample count")
    ax.set_title(f"Class distribution — {meta['name']}")
    ax.set_ylim(0, max(values) * 1.15)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            str(int(h)), xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9,
        )

    dist_desc = ", ".join(f"{l}: {v}" for l, v in zip(labels, values))
    alt = f"Bar chart showing class distribution for {meta['name']}: {dist_desc}."
    fname = _safe_filename("class_distribution", dataset_slug) + ".png"
    return _save_figure(fig, fname, alt)


def generate_correlation_heatmap(dataset_slug: str) -> dict[str, Any]:
    """Per-dataset correlation heatmap."""
    meta = DATASET_META[dataset_slug]
    X, _, feature_names, _ = load_dataset(dataset_slug)
    X_filled = X.fillna(X.median())
    corr = X_filled.corr()

    n = len(feature_names)
    figsize = (max(5.5, min(11.0, n * 0.42)), max(4.5, min(9.5, n * 0.36)))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, annot=n <= 15, fmt=".2f" if n <= 15 else "",
        cmap="YlGnBu", ax=ax, square=True,
        cbar_kws={"shrink": 0.8},
        xticklabels=feature_names, yticklabels=feature_names,
        annot_kws={"size": 7} if n <= 15 else {},
    )
    ax.set_title(f"Feature correlation heatmap — {meta['name']}")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)

    alt = f"Correlation heatmap across all {n} input features in the {meta['name']} dataset."
    fname = _safe_filename("correlation_heatmap", dataset_slug) + ".png"
    return _save_figure(fig, fname, alt)


def generate_all_figures(result: dict) -> dict[str, dict[str, Any]]:
    """Generate all five mandatory figures and return metadata dicts."""
    ds = result["params"]["dataset"]
    clf = result["params"]["classifier"]
    method = result["params"]["method"]
    k = result["params"]["k"]

    return {
        "accuracy_vs_features": generate_accuracy_vs_features(ds, clf, method, extra_k=k),
        "baseline_vs_optimized": generate_baseline_vs_optimized(result),
        "feature_importance": generate_feature_importance(result),
        "confusion_matrix": generate_confusion_matrix(result),
        "accuracy_vs_runtime": generate_accuracy_vs_runtime(ds, clf, method, k),
    }


# ---------------------------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------------------------

def run_all_experiments() -> pd.DataFrame:
    """
    Run all combinations and produce the final comparison table.
    Uses the first example subset size for SelectKBest and RFE per dataset
    with Logistic Regression.
    """
    rows = []
    default_classifier = "logistic_regression"

    for ds_slug, meta in DATASET_META.items():
        representative_k = meta["example_sizes"][0]

        res_skb = run_experiment(ds_slug, default_classifier, "selectkbest", representative_k)
        res_rfe = run_experiment(ds_slug, default_classifier, "rfe", representative_k)

        rows.append({
            "Dataset": meta["name"],
            "Method": "All features",
            "k / total": f"{meta['n_features']}/{meta['n_features']}",
            "CV accuracy mean +/- SD": (
                f"{res_skb['baseline']['cv_accuracy_mean']:.4f} +/- "
                f"{res_skb['baseline']['cv_accuracy_std']:.4f}"
            ),
            "Test accuracy": f"{res_skb['baseline']['test_accuracy']:.4f}",
            "Macro-F1": f"{res_skb['baseline']['test_macro_f1']:.4f}",
        })

        rows.append({
            "Dataset": meta["name"],
            "Method": "SelectKBest",
            "k / total": f"{representative_k}/{meta['n_features']}",
            "CV accuracy mean +/- SD": (
                f"{res_skb['selected']['cv_accuracy_mean']:.4f} +/- "
                f"{res_skb['selected']['cv_accuracy_std']:.4f}"
            ),
            "Test accuracy": f"{res_skb['selected']['test_accuracy']:.4f}",
            "Macro-F1": f"{res_skb['selected']['test_macro_f1']:.4f}",
        })

        rows.append({
            "Dataset": meta["name"],
            "Method": "RFE",
            "k / total": f"{representative_k}/{meta['n_features']}",
            "CV accuracy mean +/- SD": (
                f"{res_rfe['selected']['cv_accuracy_mean']:.4f} +/- "
                f"{res_rfe['selected']['cv_accuracy_std']:.4f}"
            ),
            "Test accuracy": f"{res_rfe['selected']['test_accuracy']:.4f}",
            "Macro-F1": f"{res_rfe['selected']['test_macro_f1']:.4f}",
        })

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "final_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved final comparison table to {csv_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OptiSelect experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments and generate final table")
    parser.add_argument("--dataset", type=str, help="Dataset slug")
    parser.add_argument("--classifier", type=str, default="logistic_regression", help="Classifier key")
    parser.add_argument("--method", type=str, help="Feature selection method")
    parser.add_argument("--k", type=int, help="Number of features to select")
    parser.add_argument("--figures", action="store_true", help="Generate figures")

    args = parser.parse_args()

    if args.all:
        print("Running all experiments...")
        df = run_all_experiments()
        print("\n" + df.to_string(index=False))
        for slug in DATASET_META:
            print(f"Generating dataset charts for {slug}...")
            generate_class_distribution(slug)
            generate_correlation_heatmap(slug)
        print("\nDone. All results saved to results/")
        return

    if not args.dataset or not args.method or not args.k:
        parser.error("Provide --dataset, --method and --k, or use --all")

    result = run_experiment(args.dataset, args.classifier, args.method, args.k)
    print(json.dumps(result, indent=2))

    if args.figures:
        figs = generate_all_figures(result)
        print("\nGenerated figures:")
        for name, meta in figs.items():
            print(f"  {name}: {meta['filename']} ({meta['width']}x{meta['height']})")


if __name__ == "__main__":
    main()
