import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { usePageMeta } from '../hooks/usePageMeta';
import type { DatasetSummary } from '../types';

export default function HomePage() {
  usePageMeta({
    title: 'OptiSelect — Choose fewer features without losing accuracy',
    description:
      'Compare classification models trained on all features against automatically selected subsets across five benchmark datasets.',
    canonicalPath: '/',
  });

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch('/api/datasets')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: DatasetSummary[]) => {
        if (active) {
          setDatasets(data);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setError(err instanceof Error ? err.message : 'Failed to load datasets');
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <Layout>
      <section aria-labelledby="home-heading">
        <div className="panel card-accent">
          <h1 id="home-heading">Choose fewer features without losing accuracy</h1>
          <p className="lead">
            OptiSelect evaluates filter-based and wrapper-based feature selection against
            all-feature baselines across five classification datasets under a strict,
            leakage-free validation protocol.
          </p>
          <div className="btn-group">
            <Link to="/experiment" className="btn btn-primary">
              Run an Experiment
            </Link>
            <Link to="/results" className="btn btn-secondary">
              View Final Comparison Table
            </Link>
            <Link to="/methodology" className="btn btn-secondary">
              Read Methodology
            </Link>
          </div>
        </div>
      </section>

      <section aria-labelledby="workflow-heading">
        <h2 id="workflow-heading">Experimental workflow</h2>
        <div className="grid-2">
          <div className="card">
            <h3>1. Leakage-free pipeline</h3>
            <p>
              Every dataset is divided with a stratified 80/20 train/test split using{' '}
              <code>random_state=42</code>. Median imputation, standard scaling, and
              feature selection fit exclusively on the training folds inside a scikit-learn{' '}
              <code>Pipeline</code>.
            </p>
          </div>
          <div className="card">
            <h3>2. Filter vs. wrapper selection</h3>
            <p>
              Compare univariate ANOVA F-score ranking (<code>SelectKBest</code>) against
              Recursive Feature Elimination (<code>RFE</code> with Logistic Regression)
              across Logistic Regression, Decision Tree, and Random Forest classifiers.
            </p>
          </div>
        </div>
      </section>

      <section aria-labelledby="datasets-overview-heading">
        <h2 id="datasets-overview-heading">Five benchmark datasets</h2>
        {loading && (
          <p className="status-box" role="status">
            Loading dataset summary...
          </p>
        )}
        {error && (
          <p className="status-box status-error" role="alert">
            Could not load datasets: {error}
          </p>
        )}
        {!loading && !error && (
          <table className="data-table responsive-table">
            <thead>
              <tr>
                <th scope="col">Dataset</th>
                <th scope="col">Rows</th>
                <th scope="col">Features</th>
                <th scope="col">Classes</th>
                <th scope="col">Example subset sizes</th>
                <th scope="col">Inspect</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((ds) => (
                <tr key={ds.slug}>
                  <td data-label="Dataset">
                    <strong>{ds.name}</strong>
                  </td>
                  <td data-label="Rows">{ds.rows}</td>
                  <td data-label="Features">{ds.n_features}</td>
                  <td data-label="Classes">{ds.n_classes}</td>
                  <td data-label="Example sizes">{ds.example_sizes.join(', ')}</td>
                  <td data-label="Inspect">
                    <Link to={`/datasets/${ds.slug}`}>Details</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </Layout>
  );
}
