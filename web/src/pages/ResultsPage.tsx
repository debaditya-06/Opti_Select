import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { usePageMeta } from '../hooks/usePageMeta';
import type { BreadcrumbItem, FinalTableResponse } from '../types';

const BREADCRUMBS: BreadcrumbItem[] = [
  { label: 'Home', path: '/' },
  { label: 'Final Results' },
];

export default function ResultsPage() {
  usePageMeta({
    title: 'Final Comparison Results — OptiSelect',
    description:
      'Measured baseline and feature-selection metrics from results/final_table.csv across all five benchmark datasets.',
    canonicalPath: '/results',
    breadcrumbs: BREADCRUMBS,
  });

  const [data, setData] = useState<FinalTableResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch('/api/results/final')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((payload: FinalTableResponse) => {
        if (active) {
          setData(payload);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setError(err instanceof Error ? err.message : 'Failed to load results table');
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <Layout breadcrumbs={BREADCRUMBS}>
      <h1>Final comparison table</h1>
      <p className="lead">
        Measured results read directly from <code>results/final_table.csv</code>,
        generated with Logistic Regression (<code>random_state=42</code>) comparing All
        features against SelectKBest and RFE on each dataset.
      </p>

      {loading && (
        <p className="status-box" role="status">
          Loading final comparison table...
        </p>
      )}
      {error && (
        <p className="status-box status-error" role="alert">
          Error loading results: {error}
        </p>
      )}

      {!loading && !error && data && (
        <>
          <div className="btn-group" style={{ marginBottom: '1.5rem' }}>
            <a
              href={data.csv_url}
              download="final_table.csv"
              className="btn btn-primary"
            >
              Download CSV (final_table.csv)
            </a>
            <Link to="/experiment" className="btn btn-secondary">
              Run Custom Experiment
            </Link>
          </div>

          <table className="data-table responsive-table">
            <thead>
              <tr>
                {data.columns.map((col) => (
                  <th scope="col" key={col}>
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, idx) => (
                <tr key={idx}>
                  {data.columns.map((col) => (
                    <td key={col} data-label={col}>
                      {col === 'Method' && row[col] !== 'All features' ? (
                        <span className="badge-amber">{row[col]}</span>
                      ) : col === 'Dataset' ? (
                        <strong>{row[col]}</strong>
                      ) : (
                        row[col]
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Layout>
  );
}
