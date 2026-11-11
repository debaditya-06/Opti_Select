import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { usePageMeta } from '../hooks/usePageMeta';
import type { BreadcrumbItem, DatasetSummary, ExperimentResult } from '../types';

const BREADCRUMBS: BreadcrumbItem[] = [
  { label: 'Home', path: '/' },
  { label: 'Run Experiment' },
];

const CLASSIFIERS = [
  { key: 'logistic_regression', label: 'Logistic Regression' },
  { key: 'decision_tree', label: 'Decision Tree' },
  { key: 'random_forest', label: 'Random Forest' },
];

const METHODS = [
  { key: 'selectkbest', label: 'SelectKBest (Filter — ANOVA F-score)' },
  { key: 'rfe', label: 'RFE (Wrapper — Recursive Feature Elimination)' },
];

export default function ExperimentPage() {
  usePageMeta({
    title: 'Run Experiment — OptiSelect',
    description:
      'Select a dataset, classifier, feature selection method, and feature count k to compare subset performance against the all-feature baseline.',
    canonicalPath: '/experiment',
    breadcrumbs: BREADCRUMBS,
  });

  const [searchParams] = useSearchParams();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  const [selectedDataset, setSelectedDataset] = useState<string>('iris');
  const [classifier, setClassifier] = useState<string>('logistic_regression');
  const [method, setMethod] = useState<string>('selectkbest');
  const [k, setK] = useState<number>(2);

  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<ExperimentResult | null>(null);

  useEffect(() => {
    let active = true;
    fetch('/api/datasets')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: DatasetSummary[]) => {
        if (!active) return;
        setDatasets(data);
        const paramDs = searchParams.get('dataset');
        const initialDs =
          data.find((d) => d.slug === paramDs) ?? data[0];
        if (initialDs) {
          setSelectedDataset(initialDs.slug);
          setK(Math.min(initialDs.example_sizes[0] ?? 2, initialDs.n_features));
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setDatasetsError(
            err instanceof Error ? err.message : 'Failed to load dataset options'
          );
        }
      });
    return () => {
      active = false;
    };
  }, [searchParams]);

  const currentDataset = datasets.find((d) => d.slug === selectedDataset);
  const maxFeatures = currentDataset?.n_features ?? 4;

  function handleDatasetChange(newSlug: string) {
    setSelectedDataset(newSlug);
    const ds = datasets.find((d) => d.slug === newSlug);
    if (ds) {
      const defaultK = ds.example_sizes[0] ?? 2;
      setK(Math.max(2, Math.min(defaultK, ds.n_features)));
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setRunError(null);
    setRunning(true);

    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 60000);

    try {
      const resp = await fetch('/api/experiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset: selectedDataset,
          classifier,
          method,
          k,
        }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        let message = `Request failed (${resp.status})`;
        try {
          const errBody = await resp.json();
          if (typeof errBody.detail === 'string') {
            message = errBody.detail;
          } else if (Array.isArray(errBody.detail) && errBody.detail[0]?.msg) {
            message = errBody.detail[0].msg;
          }
        } catch {
          // keep fallback message
        }
        throw new Error(message);
      }

      const data: ExperimentResult = await resp.json();
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setRunError('Experiment timed out after 60 seconds. Please try again.');
      } else {
        setRunError(err instanceof Error ? err.message : 'Failed to run experiment');
      }
    } finally {
      window.clearTimeout(timer);
      setRunning(false);
    }
  }

  const figureEntries = result
    ? [
        { title: '1. Accuracy vs number of features', fig: result.figures.accuracy_vs_features, lazy: false },
        { title: '2. Baseline vs optimized subset', fig: result.figures.baseline_vs_optimized, lazy: false },
        { title: '3. Feature importance / scores', fig: result.figures.feature_importance, lazy: true },
        { title: '4. Confusion matrix (held-out test)', fig: result.figures.confusion_matrix, lazy: true },
        { title: '5. Accuracy vs runtime', fig: result.figures.accuracy_vs_runtime, lazy: true },
      ]
    : [];

  return (
    <Layout breadcrumbs={BREADCRUMBS}>
      <h1>Run feature selection experiment</h1>
      <p className="lead">
        Configure a dataset, classifier, selection strategy, and target subset size{' '}
        <code>k</code>. OptiSelect evaluates both the selected subset and the
        all-features baseline on the identical stratified split.
      </p>

      {datasetsError && (
        <div className="status-box status-error" role="alert">
          Error loading dataset configuration: {datasetsError}
        </div>
      )}

      <div className="experiment-layout">
        <aside aria-label="Experiment controls">
          <form className="panel" onSubmit={handleSubmit}>
            <div className="form-field">
              <label className="form-label" htmlFor="dataset-select">
                Dataset
              </label>
              <select
                id="dataset-select"
                className="form-select"
                value={selectedDataset}
                onChange={(e) => handleDatasetChange(e.target.value)}
                disabled={datasets.length === 0 || running}
              >
                {datasets.length === 0 ? (
                  <option value="iris">Loading datasets...</option>
                ) : (
                  datasets.map((ds) => (
                    <option key={ds.slug} value={ds.slug}>
                      {ds.name} ({ds.n_features} features)
                    </option>
                  ))
                )}
              </select>
            </div>

            <div className="form-field">
              <label className="form-label" htmlFor="classifier-select">
                Classifier
              </label>
              <select
                id="classifier-select"
                className="form-select"
                value={classifier}
                onChange={(e) => setClassifier(e.target.value)}
                disabled={running}
              >
                {CLASSIFIERS.map((clf) => (
                  <option key={clf.key} value={clf.key}>
                    {clf.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-field">
              <label className="form-label" htmlFor="method-select">
                Feature-selection method
              </label>
              <select
                id="method-select"
                className="form-select"
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                disabled={running}
              >
                {METHODS.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-field">
              <label className="form-label" htmlFor="k-slider">
                Number of features (k): {k} of {maxFeatures}
              </label>
              <input
                id="k-slider"
                type="range"
                className="form-range"
                min={2}
                max={maxFeatures}
                step={1}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                disabled={running}
              />
              <div className="form-hint">
                Valid range: 2 to {maxFeatures} features.
              </div>
              {currentDataset && (
                <div className="preset-row" aria-label="Example subset sizes">
                  {currentDataset.example_sizes.map((size) => (
                    <button
                      key={size}
                      type="button"
                      className={`preset-btn${k === size ? ' is-active' : ''}`}
                      onClick={() => setK(size)}
                      disabled={running}
                    >
                      k={size}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: '100%' }}
              disabled={running || datasets.length === 0}
            >
              {running ? 'Running Experiment...' : 'Run Experiment'}
            </button>
          </form>
        </aside>

        <section aria-label="Experiment results">
          {runError && (
            <div className="status-box status-error" role="alert">
              <strong>Experiment error:</strong> {runError}
            </div>
          )}

          {running && (
            <div className="status-box" role="status" aria-live="polite">
              <strong>Training pipelines and generating charts...</strong>
              <p style={{ marginTop: '0.5rem', marginBottom: 0 }}>
                Fitting 5-fold cross-validation on the training split and evaluating once on
                the held-out test split.
              </p>
            </div>
          )}

          {!running && !result && !runError && (
            <div className="status-box">
              <h2>Ready to evaluate</h2>
              <p>
                Choose a dataset, classifier, selection method, and feature count{' '}
                <code>k</code> on the left, then select <strong>Run Experiment</strong> to
                view accuracy, macro-F1, selected feature names, training times, and all
                five comparison figures.
              </p>
            </div>
          )}

          {!running && result && (
            <div>
              <h2>
                Results: {result.dataset_name} ({result.classifier_name})
              </h2>

              <div className="kpi-grid">
                <div className="kpi-card">
                  <div className="kpi-label">Selected Test Accuracy</div>
                  <div className="kpi-value">
                    {(result.selected.test_accuracy * 100).toFixed(2)}%
                  </div>
                  <div className="kpi-sub">
                    Baseline: {(result.baseline.test_accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Selected Macro-F1</div>
                  <div className="kpi-value">
                    {result.selected.test_macro_f1.toFixed(4)}
                  </div>
                  <div className="kpi-sub">
                    Baseline: {result.baseline.test_macro_f1.toFixed(4)}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">5-Fold CV Accuracy</div>
                  <div className="kpi-value">
                    {(result.selected.cv_accuracy_mean * 100).toFixed(2)}%
                  </div>
                  <div className="kpi-sub">
                    +/- {(result.selected.cv_accuracy_std * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Features Retained</div>
                  <div className="kpi-value">
                    {result.selected.n_features} / {result.n_features_total}
                  </div>
                  <div className="kpi-sub">
                    Runtime: {result.selected.training_time.toFixed(3)}s
                  </div>
                </div>
              </div>

              <div className="card card-accent" style={{ marginBottom: '1.5rem' }}>
                <h3>
                  Selected feature names ({result.selected.n_features} via{' '}
                  {result.method_name})
                </h3>
                <ul className="feature-tags">
                  {result.selected.selected_feature_names.map((fname) => (
                    <li key={fname} className="feature-tag">
                      {fname}
                    </li>
                  ))}
                </ul>
              </div>

              <h3>Baseline vs. {result.method_name} metrics</h3>
              <table className="data-table responsive-table">
                <thead>
                  <tr>
                    <th scope="col">Configuration</th>
                    <th scope="col">Features</th>
                    <th scope="col">CV accuracy (mean +/- SD)</th>
                    <th scope="col">Test accuracy</th>
                    <th scope="col">Test macro-F1</th>
                    <th scope="col">Training time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td data-label="Configuration">
                      <strong>All features (Baseline)</strong>
                    </td>
                    <td data-label="Features">
                      {result.baseline.n_features} / {result.n_features_total}
                    </td>
                    <td data-label="CV accuracy">
                      {result.baseline.cv_accuracy_mean.toFixed(4)} +/-{' '}
                      {result.baseline.cv_accuracy_std.toFixed(4)}
                    </td>
                    <td data-label="Test accuracy">
                      {result.baseline.test_accuracy.toFixed(4)}
                    </td>
                    <td data-label="Test macro-F1">
                      {result.baseline.test_macro_f1.toFixed(4)}
                    </td>
                    <td data-label="Training time (s)">
                      {result.baseline.training_time.toFixed(4)}
                    </td>
                  </tr>
                  <tr>
                    <td data-label="Configuration">
                      <strong>{result.method_name}</strong>
                    </td>
                    <td data-label="Features">
                      {result.selected.n_features} / {result.n_features_total}
                    </td>
                    <td data-label="CV accuracy">
                      {result.selected.cv_accuracy_mean.toFixed(4)} +/-{' '}
                      {result.selected.cv_accuracy_std.toFixed(4)}
                    </td>
                    <td data-label="Test accuracy">
                      {result.selected.test_accuracy.toFixed(4)}
                    </td>
                    <td data-label="Test macro-F1">
                      {result.selected.test_macro_f1.toFixed(4)}
                    </td>
                    <td data-label="Training time (s)">
                      {result.selected.training_time.toFixed(4)}
                    </td>
                  </tr>
                </tbody>
              </table>

              <h2>Five mandatory visualizations</h2>
              <div className="figures-grid">
                {figureEntries.map(({ title, fig, lazy }) => (
                  <figure key={fig.filename} className="figure-card">
                    <h3>{title}</h3>
                    <img
                      src={fig.url}
                      width={fig.width}
                      height={fig.height}
                      alt={fig.alt}
                      loading={lazy ? 'lazy' : undefined}
                    />
                    <figcaption className="figure-caption">{fig.alt}</figcaption>
                  </figure>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </Layout>
  );
}
