import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { usePageMeta } from '../hooks/usePageMeta';
import type { BreadcrumbItem, DatasetSummary } from '../types';

const BREADCRUMBS: BreadcrumbItem[] = [
  { label: 'Home', path: '/' },
  { label: 'Datasets' },
];

export default function DatasetsPage() {
  usePageMeta({
    title: 'Datasets — OptiSelect',
    description:
      'Explore the five classification datasets evaluated in OptiSelect: Iris, Wine, Breast Cancer Wisconsin, Heart Disease, and Wine Quality.',
    canonicalPath: '/datasets',
    breadcrumbs: BREADCRUMBS,
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
    <Layout breadcrumbs={BREADCRUMBS}>
      <h1>Benchmark datasets</h1>
      <p className="lead">
        All five datasets are loaded from real scikit-learn bundles or downloaded UCI
        Machine Learning Repository data files in <code>data/</code>.
      </p>

      {loading && (
        <p className="status-box" role="status">
          Loading dataset catalog...
        </p>
      )}
      {error && (
        <p className="status-box status-error" role="alert">
          Error loading datasets: {error}
        </p>
      )}

      {!loading && !error && (
        <table className="data-table responsive-table">
          <thead>
            <tr>
              <th scope="col">Dataset</th>
              <th scope="col">Source</th>
              <th scope="col">Rows</th>
              <th scope="col">Features</th>
              <th scope="col">Classes</th>
              <th scope="col">Missing values</th>
              <th scope="col">Actions</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((ds) => (
              <tr key={ds.slug}>
                <td data-label="Dataset">
                  <strong>{ds.name}</strong>
                </td>
                <td data-label="Source">{ds.source}</td>
                <td data-label="Rows">{ds.rows}</td>
                <td data-label="Features">{ds.n_features}</td>
                <td data-label="Classes">{ds.n_classes}</td>
                <td data-label="Missing values">{ds.missing_values}</td>
                <td data-label="Actions">
                  <Link to={`/datasets/${ds.slug}`} style={{ marginRight: '0.75rem' }}>
                    Inspect
                  </Link>
                  <Link to={`/experiment?dataset=${ds.slug}`}>Run</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Layout>
  );
}
