"""
Tests for OptiSelect experiments.

Covers:
- No data leakage: selector and scaler are inside the Pipeline
- k bounds validation per dataset
- Reproducibility under random_state=42
- Baseline uses all features
- Final comparison table has all expected rows
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments


# ---------------------------------------------------------------------------
# Test: no leakage — selector and scaler fitted only on training data
# ---------------------------------------------------------------------------

class TestNoLeakage:
    """Prove that preprocessing and selection happen inside the Pipeline."""

    def test_pipeline_contains_scaler_and_selector(self):
        """The Pipeline must contain imputer, scaler, and selector steps."""
        pipe = experiments._build_pipeline("logistic_regression", "selectkbest", 3)
        step_names = [name for name, _ in pipe.steps]
        assert "imputer" in step_names, "Pipeline must include imputer"
        assert "scaler" in step_names, "Pipeline must include scaler"
        assert "selector" in step_names, "Pipeline must include selector"
        assert "model" in step_names, "Pipeline must include model"

    def test_pipeline_baseline_has_no_selector(self):
        """Baseline Pipeline should not have a selector step."""
        pipe = experiments._build_pipeline("logistic_regression", None, None)
        step_names = [name for name, _ in pipe.steps]
        assert "selector" not in step_names
        assert "imputer" in step_names
        assert "scaler" in step_names

    def test_scaler_not_fitted_before_pipeline_fit(self):
        """Scaler inside the Pipeline should not be fitted before .fit() is called."""
        pipe = experiments._build_pipeline("logistic_regression", "selectkbest", 3)
        scaler = pipe.named_steps["scaler"]
        # Before fitting, scaler should not have mean_ attribute
        assert not hasattr(scaler, "mean_"), "Scaler should not be pre-fitted"

    def test_fit_only_on_train_data(self):
        """
        When we call pipe.fit(X_train, y_train), the scaler sees only training data.
        This is guaranteed by Pipeline architecture: each step's fit_transform is
        called sequentially, never on test data.
        """
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        X, y = load_iris(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        pipe = experiments._build_pipeline("logistic_regression", "selectkbest", 3)
        pipe.fit(X_train, y_train)

        scaler = pipe.named_steps["scaler"]
        # Scaler was fitted on imputed X_train; it should have n_samples_seen_
        # equal to the training set size
        assert scaler.n_samples_seen_ == len(X_train), (
            f"Scaler saw {scaler.n_samples_seen_} samples, expected {len(X_train)} "
            "(training set size). Leakage detected if this doesn't match."
        )

    def test_selector_fitted_only_on_train(self):
        """SelectKBest inside Pipeline is fitted only on training data."""
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        X, y = load_iris(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        pipe = experiments._build_pipeline("logistic_regression", "selectkbest", 3)
        pipe.fit(X_train, y_train)

        selector = pipe.named_steps["selector"]
        # SelectKBest scores should have length == number of features
        assert len(selector.scores_) == X_train.shape[1]
        # The selector was fitted inside the pipeline, so it never saw X_test
        mask = selector.get_support()
        assert sum(mask) == 3

    def test_fit_never_called_on_test_data(self, monkeypatch):
        """Spy on StandardScaler.fit and SelectKBest.fit during run_experiment to prove test rows are never fitted."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest

        X, _, _, _ = experiments.load_dataset("iris")
        expected_max_rows = int(len(X) * 0.8)  # 80% training split (or 4/5 of 80% inside CV)

        scaler_row_counts = []
        selector_row_counts = []

        orig_scaler_fit = StandardScaler.fit
        orig_selector_fit = SelectKBest.fit

        def spy_scaler_fit(self, X_in, y_in=None, **kwargs):
            scaler_row_counts.append(len(X_in))
            return orig_scaler_fit(self, X_in, y_in, **kwargs)

        def spy_selector_fit(self, X_in, y_in=None, **kwargs):
            selector_row_counts.append(len(X_in))
            return orig_selector_fit(self, X_in, y_in, **kwargs)

        monkeypatch.setattr(StandardScaler, "fit", spy_scaler_fit)
        monkeypatch.setattr(SelectKBest, "fit", spy_selector_fit)

        experiments.run_experiment("iris", "logistic_regression", "selectkbest", 2, use_cache=False)

        assert len(scaler_row_counts) > 0
        assert len(selector_row_counts) > 0
        # Test split is 30 rows for Iris (20% of 150); training split is 120 rows; CV folds are 96 rows.
        assert all(n <= expected_max_rows for n in scaler_row_counts)
        assert all(n <= expected_max_rows for n in selector_row_counts)
        assert 30 not in scaler_row_counts
        assert 30 not in selector_row_counts


# ---------------------------------------------------------------------------
# Test: k bounds per dataset
# ---------------------------------------------------------------------------

class TestKBounds:
    """Validate that k must be in [2, n_features] for each dataset."""

    @pytest.mark.parametrize("slug,max_k", [
        ("iris", 4),
        ("wine", 13),
        ("breast-cancer", 30),
        ("heart-disease", 13),
        ("wine-quality", 11),
    ])
    def test_k_too_large(self, slug: str, max_k: int):
        with pytest.raises(ValueError, match="k must be between"):
            experiments.run_experiment(slug, "logistic_regression", "selectkbest", max_k + 1)

    @pytest.mark.parametrize("slug", [
        "iris", "wine", "breast-cancer", "heart-disease", "wine-quality",
    ])
    def test_k_too_small(self, slug: str):
        with pytest.raises(ValueError, match="k must be between"):
            experiments.run_experiment(slug, "logistic_regression", "selectkbest", 1)

    @pytest.mark.parametrize("slug,max_k", [
        ("iris", 4),
        ("wine", 13),
        ("breast-cancer", 30),
        ("heart-disease", 13),
        ("wine-quality", 11),
    ])
    def test_k_valid_min(self, slug: str, max_k: int):
        result = experiments.run_experiment(slug, "logistic_regression", "selectkbest", 2)
        assert result["selected"]["n_features"] == 2

    @pytest.mark.parametrize("slug,max_k", [
        ("iris", 4),
        ("wine", 13),
        ("breast-cancer", 30),
        ("heart-disease", 13),
        ("wine-quality", 11),
    ])
    def test_k_valid_max(self, slug: str, max_k: int):
        result = experiments.run_experiment(slug, "logistic_regression", "selectkbest", max_k)
        assert result["selected"]["n_features"] == max_k


# ---------------------------------------------------------------------------
# Test: reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Results must be identical across runs with random_state=42."""

    def test_same_results_twice(self):
        r1 = experiments.run_experiment("iris", "logistic_regression", "selectkbest", 3, use_cache=False)
        r2 = experiments.run_experiment("iris", "logistic_regression", "selectkbest", 3, use_cache=False)
        assert r1["selected"]["test_accuracy"] == r2["selected"]["test_accuracy"]
        assert r1["selected"]["test_macro_f1"] == r2["selected"]["test_macro_f1"]
        assert r1["selected"]["cv_accuracy_mean"] == r2["selected"]["cv_accuracy_mean"]
        assert r1["selected"]["selected_feature_names"] == r2["selected"]["selected_feature_names"]
        assert r1["baseline"]["test_accuracy"] == r2["baseline"]["test_accuracy"]


# ---------------------------------------------------------------------------
# Test: baseline uses all features
# ---------------------------------------------------------------------------

class TestBaseline:
    """Baseline configuration must use all features."""

    @pytest.mark.parametrize("slug", list(experiments.DATASET_META.keys()))
    def test_baseline_uses_all_features(self, slug: str):
        meta = experiments.DATASET_META[slug]
        result = experiments.run_experiment(slug, "logistic_regression", "selectkbest", 2)
        assert result["baseline"]["n_features"] == meta["n_features"]


# ---------------------------------------------------------------------------
# Test: final table completeness
# ---------------------------------------------------------------------------

class TestFinalTable:
    """The final comparison table must have rows for every dataset."""

    def test_final_table_has_all_rows(self):
        df = experiments.run_all_experiments()
        # Should have 3 rows per dataset: All features, SelectKBest, RFE
        expected_datasets = set(meta["name"] for meta in experiments.DATASET_META.values())
        actual_datasets = set(df["Dataset"].unique())
        assert expected_datasets == actual_datasets, (
            f"Missing datasets: {expected_datasets - actual_datasets}"
        )
        # Each dataset should have exactly 3 rows
        for ds_name in expected_datasets:
            ds_rows = df[df["Dataset"] == ds_name]
            assert len(ds_rows) == 3, (
                f"{ds_name} has {len(ds_rows)} rows, expected 3"
            )
            methods = set(ds_rows["Method"])
            assert methods == {"All features", "SelectKBest", "RFE"}


# ---------------------------------------------------------------------------
# Test: datasets load correctly
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    """All five datasets must load with correct shapes."""

    @pytest.mark.parametrize("slug,expected_features", [
        ("iris", 4),
        ("wine", 13),
        ("breast-cancer", 30),
        ("heart-disease", 13),
        ("wine-quality", 11),
    ])
    def test_dataset_loads(self, slug: str, expected_features: int):
        X, y, feature_names, class_names = experiments.load_dataset(slug)
        assert X.shape[1] == expected_features
        assert len(feature_names) == expected_features
        assert len(y) == len(X)
        assert len(class_names) >= 2
