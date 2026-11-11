import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { usePageMeta } from '../hooks/usePageMeta';
import type { BreadcrumbItem, DatasetDetail } from '../types';

const DATASET_TITLES: Record<string, string> = {
  iris: 'Iris',
  wine: 'Wine',
  'breast-cancer': 'Breast Cancer Wisconsin',
  'heart-disease': 'Heart Disease (UCI Cleveland)',
  'wine-quality': 'Wine Quality (Red)',
};

export default function DatasetDetailPage() {
  const { slug = '' } = useParams<{ slug: string }>();
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [resolvedSlug, setResolvedSlug] = useState<string>('');
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loading = resolvedSlug !== slug;
  const activeDetail = detail?.slug === slug ? detail : null;
  const displayName = activeDetail?.name ?? DATASET_TITLES[slug] ?? 'Dataset';

  const breadcrumbs: BreadcrumbItem[] = useMemo(
    () => [
      { label: 'Home', path: '/' },
      { label: 'Datasets', path: '/datasets' },
      { label: notFound && !loading ? 'Not Found' : displayName },
    ],
    [displayName, notFound, loading]
  );

  usePageMeta({
    title:
      notFound && !loading
        ? 'Dataset not found — OptiSelect'
        : `${displayName} Dataset — OptiSelect`,
    description:
      notFound && !loading
        ? 'The requested dataset could not be found.'
        : `Class distribution, preprocessing notes, feature correlations, and sample preview for the ${displayName} dataset.`,
    canonicalPath: `/datasets/${slug}`,
    breadcrumbs,
  });

  useEffect(() => {
    let active = true;

    fetch(`/api/datasets/${encodeURIComponent(slug)}`)
      .then((res) => {
        if (res.status === 404) {
          throw new Error('NOT_FOUND');
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: DatasetDetail) => {
        if (active) {
          setDetail(data);
          setNotFound(false);
          setError(null);
          setResolvedSlug(slug);
        }
      })
      .catch((err: unknown) => {
        if (!active) return;
        if (err instanceof Error && err.message === 'NOT_FOUND') {
          setNotFound(true);
          setError(null);
        } else {
          setNotFound(false);
          setError(err instanceof Error ? err.message : 'Failed to load dataset');
        }
        setResolvedSlug(slug);
      });

    return () => {
      active = false;
    };
  }, [slug]);

  if (!loading && notFound) {
    return (
      <Layout breadcrumbs={breadcrumbs}>
        <h1>Dataset not found</h1>
        <p className="lead">
          No dataset matches <code>{slug}</code>.
        </p>
        <Link to="/datasets" className="btn btn-primary">
          Back to Datasets
        </Link>
      </Layout>
    );
  }

  return (
    <Layout breadcrumbs={breadcrumbs}>
      <h1>{displayName}</h1>

      {loading && (
        <p className="status-box" role="status">
          Loading dataset profile and figures...
        </p>
      )}
      {error && (
        <p className="status-box status-error" role="alert">
          Error loading dataset: {error}
        </p>
      )}

      {!loading && !error && activeDetail && (
        <>
          <p className="lead">{activeDetail.description}</p>

          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">Samples</div>
              <div className="kpi-value">{activeDetail.rows}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Input Features</div>
              <div className="kpi-value">{activeDetail.features}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Classes</div>
              <div className="kpi-value">{activeDetail.n_classes}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Missing Values</div>
              <div className="kpi-value">{activeDetail.missing_values}</div>
            </div>
          </div>

          <div className="btn-group" style={{ marginBottom: '2rem' }}>
            <Link to={`/experiment?dataset=${activeDetail.slug}`} className="btn btn-primary">
              Run Experiment on {activeDetail.name}
            </Link>
            <Link to="/datasets" className="btn btn-secondary">
              All Datasets
            </Link>
          </div>

          <div className="grid-2">
            <div className="card card-accent">
              <h2>Target &amp; preprocessing</h2>
              <p>
                <strong>Source:</strong> {activeDetail.source}
              </p>
              <p>
                <strong>Target definition:</strong> {activeDetail.target_doc}
              </p>
              <p>
                <strong>Preprocessing:</strong> {activeDetail.preprocessing}
              </p>
              <p style={{ marginBottom: 0 }}>
                <strong>Example subset sizes:</strong> {activeDetail.example_sizes.join(', ')}
              </p>
            </div>

            <div className="card">
              <h2>Class counts</h2>
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Class</th>
                    <th scope="col">Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(activeDetail.class_counts).map(([cls, count]) => (
                    <tr key={cls}>
                      <td>{cls}</td>
                      <td>{count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <h2>Exploratory figures</h2>
          <div className="figures-grid">
            <figure className="figure-card">
              <h3>Class distribution</h3>
              <img
                src={activeDetail.charts.class_distribution.url}
                width={activeDetail.charts.class_distribution.width}
                height={activeDetail.charts.class_distribution.height}
                alt={activeDetail.charts.class_distribution.alt}
                loading="lazy"
              />
              <figcaption className="figure-caption">
                {activeDetail.charts.class_distribution.alt}
              </figcaption>
            </figure>

            <figure className="figure-card">
              <h3>Feature correlation heatmap</h3>
              <img
                src={activeDetail.charts.correlation_heatmap.url}
                width={activeDetail.charts.correlation_heatmap.width}
                height={activeDetail.charts.correlation_heatmap.height}
                alt={activeDetail.charts.correlation_heatmap.alt}
                loading="lazy"
              />
              <figcaption className="figure-caption">
                {activeDetail.charts.correlation_heatmap.alt}
              </figcaption>
            </figure>
          </div>

          <h2>Data preview (first 6 rows, first 6 columns)</h2>
          <p className="form-hint">
            Showing up to 6 input features plus the target label for readability. All{' '}
            {activeDetail.features} features are included in modelling.
          </p>
          <table className="data-table responsive-table">
            <thead>
              <tr>
                {activeDetail.feature_names.slice(0, 6).map((col) => (
                  <th scope="col" key={col} className="mono">
                    {col}
                  </th>
                ))}
                <th scope="col">Target</th>
              </tr>
            </thead>
            <tbody>
              {activeDetail.preview.map((row, rIdx) => (
                <tr key={rIdx}>
                  {activeDetail.feature_names.slice(0, 6).map((col) => (
                    <td key={col} data-label={col}>
                      {row[col] === null || row[col] === undefined ? 'NaN' : String(row[col])}
                    </td>
                  ))}
                  <td data-label="Target">
                    <strong>{String(row._target)}</strong>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Layout>
  );
}
