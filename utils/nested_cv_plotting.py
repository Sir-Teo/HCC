import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_histogram(outer_scores, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(outer_scores, bins=10, kde=True)
    plt.xlabel("Concordance Index")
    plt.ylabel("Frequency")
    plt.title("Distribution of Outer Fold Concordance Indices")
    hist_path = os.path.join(output_dir, "nested_cv_outer_scores_distribution.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved histogram of outer fold scores to {hist_path}")
    return hist_path

def plot_boxplot(outer_scores, output_dir):
    plt.figure(figsize=(6, 8))
    sns.boxplot(y=outer_scores)
    plt.ylabel("Concordance Index")
    plt.title("Boxplot of Outer Fold Concordance Indices")
    boxplot_path = os.path.join(output_dir, "nested_cv_outer_scores_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Saved boxplot of outer fold scores to {boxplot_path}")
    return boxplot_path

def save_hyperparams_table(best_hyperparams_list, output_dir):
    df_best_hyperparams = pd.DataFrame(best_hyperparams_list)
    csv_path = os.path.join(output_dir, "nested_cv_best_hyperparams.csv")
    df_best_hyperparams.to_csv(csv_path, index=False)
    print(f"Saved best hyperparameters per fold to {csv_path}")
    
    plt.figure(figsize=(8, 2 + len(df_best_hyperparams)*0.5))
    plt.table(cellText=df_best_hyperparams.values, colLabels=df_best_hyperparams.columns, loc='center')
    plt.axis('off')
    plt.title("Best Hyperparameters per Outer Fold")
    table_path = os.path.join(output_dir, "nested_cv_best_hyperparams_table.png")
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()
    print(f"Saved table of best hyperparameters to {table_path}")
    return csv_path, table_path

def print_summary_stats(outer_scores):
    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    summary = f"Nested CV Outer Fold Concordance Scores: Mean = {mean_score:.4f}, Std = {std_score:.4f}"
    print(summary)
    return summary

def plot_nested_cv_stats(outer_scores, best_hyperparams_list, output_dir):
    """
    Aggregate nested CV plotting:
      - Histogram of outer fold scores
      - Boxplot of outer fold scores
      - CSV and table of best hyperparameters
      - Summary statistics printed to console
    """
    os.makedirs(output_dir, exist_ok=True)
    hist_path = plot_histogram(outer_scores, output_dir)
    boxplot_path = plot_boxplot(outer_scores, output_dir)
    csv_path, table_path = save_hyperparams_table(best_hyperparams_list, output_dir)
    summary = print_summary_stats(outer_scores)
    
    return {
        "histogram": hist_path,
        "boxplot": boxplot_path,
        "hyperparams_csv": csv_path,
        "hyperparams_table": table_path,
        "summary": summary
    }
