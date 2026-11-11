import { lazy, Suspense } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

const HomePage = lazy(() => import('./pages/HomePage'));
const ExperimentPage = lazy(() => import('./pages/ExperimentPage'));
const DatasetsPage = lazy(() => import('./pages/DatasetsPage'));
const DatasetDetailPage = lazy(() => import('./pages/DatasetDetailPage'));
const MethodologyPage = lazy(() => import('./pages/MethodologyPage'));
const ResultsPage = lazy(() => import('./pages/ResultsPage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));

function RouteFallback() {
  return (
    <div className="container" style={{ paddingTop: '2rem', paddingBottom: '2rem' }}>
      <p className="status-box" role="status">
        Loading page...
      </p>
    </div>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/experiment" element={<ExperimentPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:slug" element={<DatasetDetailPage />} />
          <Route path="/methodology" element={<MethodologyPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
