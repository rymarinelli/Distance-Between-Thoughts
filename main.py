from data.fetch_data import load_responses, load_external_dataset
from data.merge import merge_datasets
from logic.step_parser import parse_steps
from logic.embedding import embed_steps
from logic.logic_analysis import logic_error_magnitude
from evaluation.rouge_eval import compute_rouge
from evaluation.results_aggregator import collect_results
from evaluation.regression import run_regression_analysis
from visualization.plot_scores import plot_rouge_and_errors
from visualization.correlation_heatmap import plot_correlation_heatmap


def main():
    df = load_responses()
    external_df = load_external_dataset()
    merged = merge_datasets(df, external_df)
    results_df = collect_results(merged)
    plot_rouge_and_errors(results_df)
    plot_correlation_heatmap(results_df)
    run_regression_analysis(results_df)


if __name__ == "__main__":
    main()
