import { Link } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { usePageMeta } from '../hooks/usePageMeta';
import type { BreadcrumbItem } from '../types';

const BREADCRUMBS: BreadcrumbItem[] = [
  { label: 'Home', path: '/' },
  { label: 'Methodology' },
];

export default function MethodologyPage() {
  usePageMeta({
    title: 'Methodology — OptiSelect',
    description:
      'Leakage-prevention pipeline, stratified train/test split and cross-validation protocol, SelectKBest vs RFE feature selection, and documented dataset targets.',
    canonicalPath: '/methodology',
    breadcrumbs: BREADCRUMBS,
  });

  return (
    <Layout breadcrumbs={BREADCRUMBS}>
      <h1>Methodology &amp; leakage prevention</h1>
      <p className="lead">
        OptiSelect isolates all preprocessing and feature selection inside a scikit-learn{' '}
        <code>Pipeline</code> so no information from validation folds or the held-out
        test split influences scaling, imputation, or feature rankings.
      </p>

      <section aria-labelledby="pipeline-heading">
        <h2 id="pipeline-heading">Leakage-prevention pipeline</h2>
        <div className="panel card-accent">
          <p>
            Every experiment runs the identical sequence of estimators inside a single{' '}
            <code>sklearn.pipeline.Pipeline</code>:
          </p>
          <p className="mono" style={{ marginBottom: 0 }}>
            SimpleImputer(strategy=&quot;median&quot;) -&gt; StandardScaler() -&gt; Selector(k) -&gt; Classifier
          </p>
        </div>
        <p style={{ marginTop: '1rem' }}>
          For the all-features baseline, the selector step is omitted while keeping the
          same imputer, scaler, classifier, and data splits. Because each step is fitted
          exclusively via <code>Pipeline.fit</code> on training folds, test samples are
          transformed using statistics learned from training data only.
        </p>
      </section>

      <section aria-labelledby="split-heading">
        <h2 id="split-heading">Train/test split &amp; cross-validation protocol</h2>
        <div className="grid-2">
          <div className="card">
            <h3>Stratified 80/20 split</h3>
            <p>
              Each dataset is split once with{' '}
              <code>train_test_split(..., test_size=0.2, stratify=y, random_state=42)</code>.
              Class proportions are preserved across both splits, and all runs with the
              same parameters are deterministic.
            </p>
          </div>
          <div className="card">
            <h3>5-fold StratifiedKFold on train only</h3>
            <p>
              Validation metrics use{' '}
              <code>StratifiedKFold(n_splits=5, shuffle=True, random_state=42)</code>{' '}
              executed solely on the 80% training split, reporting mean accuracy and
              standard deviation across folds. The 20% test split is evaluated once for
              final test accuracy and macro-F1.
            </p>
          </div>
        </div>
      </section>

      <section aria-labelledby="selectors-heading">
        <h2 id="selectors-heading">Feature-selection methods</h2>
        <div className="grid-2">
          <div className="card">
            <h3>SelectKBest (Filter method)</h3>
            <p>
              Uses univariate ANOVA F-values (<code>f_classif</code>) to score each
              feature independently against the target class labels and retains the top{' '}
              <code>k</code> features. Fast and classifier-agnostic, though it evaluates
              features individually rather than in combination.
            </p>
          </div>
          <div className="card">
            <h3>RFE (Wrapper method)</h3>
            <p>
              Recursive Feature Elimination fits{' '}
              <code>LogisticRegression(max_iter=2000, random_state=42)</code> as the base
              estimator, recursively prunes the weakest features by coefficient magnitude,
              and retains <code>k</code> features. Captures multivariate interactions at
              higher computational cost.
            </p>
          </div>
        </div>
      </section>

      <section aria-labelledby="classifiers-heading">
        <h2 id="classifiers-heading">Classifiers &amp; documented targets</h2>
        <h3>Supported classifiers</h3>
        <ul>
          <li>
            <strong>Logistic Regression</strong>: <code>LogisticRegression(max_iter=2000, random_state=42)</code>
          </li>
          <li>
            <strong>Decision Tree</strong>: <code>DecisionTreeClassifier(random_state=42)</code>
          </li>
          <li>
            <strong>Random Forest</strong>: <code>RandomForestClassifier(random_state=42)</code>
          </li>
        </ul>

        <h3>Documented UCI dataset transformations</h3>
        <ul>
          <li>
            <strong>Heart Disease (UCI Cleveland)</strong>: 13 clinical features;{' '}
            <code>?</code> entries are parsed as missing values (<code>NaN</code>) and
            imputed using the training-fold median inside the Pipeline. The target column{' '}
            <code>num</code> (0-4) is binarized as <code>num &gt; 0</code> (&quot;disease&quot;)
            versus <code>num == 0</code> (&quot;no disease&quot;).
          </li>
          <li>
            <strong>Wine Quality (UCI Red Wine)</strong>: 11 physicochemical features. The
            ordinal quality rating (0-10) is binarized as <code>quality &gt;= 6</code>{' '}
            (&quot;good&quot;) versus <code>quality &lt; 6</code> (&quot;not good&quot;).
          </li>
        </ul>

        <div className="btn-group">
          <Link to="/experiment" className="btn btn-primary">
            Run an Experiment
          </Link>
          <Link to="/results" className="btn btn-secondary">
            Inspect Final Comparison Table
          </Link>
        </div>
      </section>
    </Layout>
  );
}
