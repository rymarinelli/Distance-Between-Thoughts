import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    """
    Displays a correlation heatmap for numeric evaluation metrics.
    """
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Evaluation Metrics")
    plt.tight_layout()
    plt.show()
