import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def run_regression_analysis(df):
    """
    Runs regression analysis and plots relationships between:
    - Logic_Mistakes vs ROUGE_L
    - Avg_Angle_Error vs ROUGE_L
    """

    # Mistakes vs ROUGE
    x1 = df["Logic_Mistakes"]
    y1 = df["ROUGE_L"]
    X1 = sm.add_constant(x1)
    model1 = sm.OLS(y1, X1).fit()
    print("ROUGE-L drop per mistake:", model1.params[1], "p =", model1.pvalues[1])

    plt.figure(figsize=(8, 5))
    sns.regplot(x="Logic_Mistakes", y="ROUGE_L", data=df)
    plt.title("ROUGE-L vs Logic Mistakes")
    plt.show()

    # Angle error vs ROUGE
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Avg_Angle_Error", "ROUGE_L"])
    x2 = df_clean["Avg_Angle_Error"]
    y2 = df_clean["ROUGE_L"]
    X2 = sm.add_constant(x2)
    model2 = sm.OLS(y2, X2).fit()
    print("ROUGE-L drop per angle unit:", model2.params[1], "p =", model2.pvalues[1])

    plt.figure(figsize=(8, 5))
    sns.regplot(x="Avg_Angle_Error", y="ROUGE_L", data=df_clean)
    plt.title("ROUGE-L vs Avg Angle Error")
    plt.show()
