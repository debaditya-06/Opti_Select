import { Link } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { usePageMeta } from '../hooks/usePageMeta';
import type { BreadcrumbItem } from '../types';

const BREADCRUMBS: BreadcrumbItem[] = [
  { label: 'Home', path: '/' },
  { label: 'Page not found' },
];

export default function NotFoundPage() {
  usePageMeta({
    title: '404 Page Not Found — OptiSelect',
    description: 'The requested page does not exist in OptiSelect.',
    canonicalPath: '/404',
    breadcrumbs: BREADCRUMBS,
  });

  return (
    <Layout breadcrumbs={BREADCRUMBS}>
      <div className="panel card-amber">
        <h1>404 — Page not found</h1>
        <p className="lead">
          The requested URL does not match any page in OptiSelect.
        </p>
        <div className="btn-group">
          <Link to="/" className="btn btn-primary">
            Return Home
          </Link>
          <Link to="/experiment" className="btn btn-secondary">
            Open Experiment Runner
          </Link>
        </div>
      </div>
    </Layout>
  );
}
