import matplotlib.pyplot as plt
import seaborn as sns

def plot_rouge_and_errors(df):
    """
    Creates grouped bar plots for ROUGE and logic error metrics.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Size", y="ROUGE_L", hue="Level")
    plt.title("ROUGE-L by Model Size and Difficulty")
    plt.ylabel("ROUGE-L Score")
    plt.xlabel("Model Size")
    plt.legend(title="Difficulty")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Size", y="Logic_Mistakes", hue="Level")
    plt.title("Logic Mistakes by Model Size and Difficulty")
    plt.ylabel("Number of Logic Mistakes")
    plt.xlabel("Model Size")
    plt.legend(title="Difficulty")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Size", y="Avg_Angle_Error", hue="Adversarial")
    plt.title("Average Angle Error by Model Size and Adversarial Status")
    plt.ylabel("Average Angular Deviation (°)")
    plt.xlabel("Model Size")
    plt.tight_layout()
    plt.show()
