export interface FigureAsset {
  filename: string;
  url: string;
  width: number;
  height: number;
  alt: string;
}

export interface DatasetSummary {
  slug: string;
  name: string;
  rows: number;
  n_features: number;
  n_classes: number;
  class_names: string[];
  source: string;
  description: string;
  example_sizes: number[];
  missing_values: number;
}

export interface DatasetDetail extends DatasetSummary {
  features: number;
  target_doc: string;
  preprocessing: string;
  feature_names: string[];
  class_counts: Record<string, number>;
  preview: Array<Record<string, number | string | null>>;
  charts: {
    class_distribution: FigureAsset;
    correlation_heatmap: FigureAsset;
  };
}

export interface ExperimentMetrics {
  test_accuracy: number;
  test_macro_f1: number;
  cv_accuracy_mean: number;
  cv_accuracy_std: number;
  training_time: number;
  n_features: number;
  selected_feature_names?: string[];
}

export interface ExperimentResult {
  params: {
    dataset: string;
    classifier: string;
    method: string;
    k: number;
  };
  dataset_name: string;
  classifier_name: string;
  method_name: string;
  n_features_total: number;
  feature_names: string[];
  class_names: string[];
  baseline: ExperimentMetrics;
  selected: ExperimentMetrics & { selected_feature_names: string[] };
  feature_scores: Record<string, number>;
  score_label: string;
  confusion_matrix: number[][];
  figures: {
    accuracy_vs_features: FigureAsset;
    baseline_vs_optimized: FigureAsset;
    feature_importance: FigureAsset;
    confusion_matrix: FigureAsset;
    accuracy_vs_runtime: FigureAsset;
  };
}

export interface FinalTableResponse {
  columns: string[];
  rows: Array<Record<string, string>>;
  csv_url: string;
}

export interface BreadcrumbItem {
  label: string;
  path?: string;
}
