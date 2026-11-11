"""
OptiSelect Streamlit Dashboard

Thin Streamlit interface that calls experiments.py for all modelling.
Provides dataset, classifier, method and feature-count controls with the
slider maximum changing per dataset, plus a Run Experiment button showing
real metrics, selected feature names, and all five mandatory figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments

st.set_page_config(
    page_title="OptiSelect — Feature Selection Dashboard",
    layout="wide",
)

st.title("OptiSelect — Feature Selection Dashboard")
st.markdown(
    "Compare classification using all features against automatically "
    "selected subsets across five real-world datasets."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Experiment controls")

dataset_options = {meta["name"]: slug for slug, meta in experiments.DATASET_META.items()}
dataset_name = st.sidebar.selectbox("Dataset", list(dataset_options.keys()))
dataset_slug = dataset_options[dataset_name]
meta = experiments.DATASET_META[dataset_slug]

classifier_options = {name: key for key, (name, _) in experiments.CLASSIFIERS.items()}
classifier_name = st.sidebar.selectbox("Classifier", list(classifier_options.keys()))
classifier_key = classifier_options[classifier_name]

method_options = {name: key for key, name in experiments.METHODS.items()}
method_name = st.sidebar.selectbox("Selection method", list(method_options.keys()))
method_key = method_options[method_name]

default_k = min(meta["example_sizes"][0], meta["n_features"])
k = st.sidebar.slider(
    "Number of features",
    min_value=2,
    max_value=meta["n_features"],
    value=default_k,
)

run_clicked = st.sidebar.button("Run Experiment", type="primary")

# ---------------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------------
try:
    info = experiments.get_dataset_info(dataset_slug)
    st.subheader(f"Dataset: {info['name']}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Samples", info["rows"])
    with col2:
        st.metric("Total features", info["features"])
    with col3:
        st.metric("Selected k", k)
    with col4:
        st.metric("Classes", info["n_classes"])

    st.markdown(f"**Source:** {info['source']}")
    st.markdown(f"**Description:** {info['description']}")
    st.markdown(f"**Target:** {info.get('target_doc', '')}")
    st.markdown(f"**Preprocessing:** {info.get('preprocessing', '')}")

    if info.get("class_counts"):
        st.markdown("**Class distribution:**")
        cc_df = pd.DataFrame(
            list(info["class_counts"].items()),
            columns=["Class", "Count"],
        )
        st.dataframe(cc_df, use_container_width=True, hide_index=True)

    if info.get("preview"):
        st.markdown("**Data preview (first 6 rows):**")
        preview_df = pd.DataFrame(info["preview"]).rename(columns={"_target": "target"})
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Could not load dataset info: {e}")

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
if run_clicked:
    with st.spinner(f"Running {method_name} with k={k} on {dataset_name} using {classifier_name}..."):
        try:
            result = experiments.run_experiment(
                dataset_slug=dataset_slug,
                classifier_key=classifier_key,
                method_key=method_key,
                k=k,
            )
            figures = experiments.generate_all_figures(result)
        except Exception as e:
            st.error(f"Experiment failed: {e}")
            st.stop()

    st.success("Experiment complete")
    st.subheader("Results")

    col_b, col_s = st.columns(2)
    bl = result["baseline"]
    sel = result["selected"]

    with col_b:
        st.markdown(f"**Baseline (all {bl['n_features']} features)**")
        st.metric("Test accuracy", f"{bl['test_accuracy']:.4f}")
        st.metric("Test macro-F1", f"{bl['test_macro_f1']:.4f}")
        st.metric("CV accuracy", f"{bl['cv_accuracy_mean']:.4f} +/- {bl['cv_accuracy_std']:.4f}")
        st.metric("Training time", f"{bl['training_time']:.4f}s")

    with col_s:
        st.markdown(f"**{result['method_name']} (k={k})**")
        st.metric("Test accuracy", f"{sel['test_accuracy']:.4f}")
        st.metric("Test macro-F1", f"{sel['test_macro_f1']:.4f}")
        st.metric("CV accuracy", f"{sel['cv_accuracy_mean']:.4f} +/- {sel['cv_accuracy_std']:.4f}")
        st.metric("Training time", f"{sel['training_time']:.4f}s")

    st.markdown(f"**Selected features ({k}):** {', '.join(sel['selected_feature_names'])}")

    st.subheader("Five mandatory figures")
    results_dir = experiments.RESULTS_DIR

    fig_cols = st.columns(2)
    with fig_cols[0]:
        fig_info = figures["accuracy_vs_features"]
        st.image(str(results_dir / fig_info["filename"]), caption=fig_info["alt"])
    with fig_cols[1]:
        fig_info = figures["baseline_vs_optimized"]
        st.image(str(results_dir / fig_info["filename"]), caption=fig_info["alt"])

    fig_cols2 = st.columns(2)
    with fig_cols2[0]:
        fig_info = figures["feature_importance"]
        st.image(str(results_dir / fig_info["filename"]), caption=fig_info["alt"])
    with fig_cols2[1]:
        fig_info = figures["confusion_matrix"]
        st.image(str(results_dir / fig_info["filename"]), caption=fig_info["alt"])

    fig_info = figures["accuracy_vs_runtime"]
    st.image(str(results_dir / fig_info["filename"]), caption=fig_info["alt"])

# ---------------------------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Final comparison table")
csv_path = experiments.RESULTS_DIR / "final_table.csv"
if csv_path.exists():
    final_df = pd.read_csv(csv_path)
    st.dataframe(final_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download CSV",
        data=final_df.to_csv(index=False),
        file_name="final_table.csv",
        mime="text/csv",
    )
else:
    st.info(
        "No final table yet. Run `python experiments.py --all` to generate it."
    )
