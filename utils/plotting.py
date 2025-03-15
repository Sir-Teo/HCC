import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

sns.set(style="whitegrid")

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def plot_training_log(log, output_dir):
    """Plot training loss over epochs."""
    plt.figure(figsize=(10, 6))
    log.plot()
    plt.title("Training Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_log.png"))
    plt.close()


def plot_survival_functions(surv, y_test_durations, y_test_events, output_dir, num_samples=15):
    """Plot survival functions for test samples and highlight recurrence events."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for i in range(min(num_samples, surv.shape[1])):
        ax.plot(surv.index, surv.iloc[:, i], lw=2, label=f"Sample {i}")
        if y_test_events[i] == 1:
            event_time = y_test_durations[i]
            nearest_idx = np.abs(surv.index - event_time).argmin()
            nearest_time = surv.index[nearest_idx]
            survival_prob_at_event = surv.iloc[nearest_idx, i]
            ax.scatter(nearest_time, survival_prob_at_event, color='red', s=50, zorder=5,
                       label='Recurrence' if i == 0 else "")
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_title("Survival Functions for Test Samples with Recurrence Highlighted", fontsize=14)
    ax.legend(title="Sample Index", fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "survival_functions.png"))
    plt.close()


def plot_brier_score(time_grid, brier_score, output_dir):
    """Plot Brier score over time with a filled area for visual emphasis."""
    plt.figure(figsize=(10, 6))
    plt.plot(time_grid, brier_score, lw=2, label='Brier Score')
    plt.fill_between(time_grid, brier_score, alpha=0.2)
    plt.title("Brier Score Over Time", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Brier Score", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "brier_score.png"))
    plt.close()


def plot_risk_score_distribution(risk_scores, median_risk, output_dir):
    """Plot the distribution of predicted risk scores with KDE and a violin inset."""
    plt.figure(figsize=(10, 6))
    sns.histplot(risk_scores, bins=30, kde=True, color='skyblue', edgecolor='k')
    plt.axvline(median_risk, color='red', linestyle='--', label=f'Median Risk ({median_risk:.2f})')
    plt.xlabel("Predicted Risk Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Predicted Risk Scores", fontsize=14)
    plt.legend()
    plt.grid(True)
    # Adding a violin plot as an inset
    inset_ax = plt.axes([0.65, 0.65, 0.2, 0.2])
    sns.violinplot(x=risk_scores, ax=inset_ax, color='lightgreen', inner='quartile')
    inset_ax.set_xlabel("")
    inset_ax.set_yticks([])
    plt.savefig(os.path.join(output_dir, "risk_score_distribution.png"))
    plt.close()


def plot_kaplan_meier(kmf_low, kmf_high, y_test_durations, y_test_events, low_risk_idx, high_risk_idx, output_dir):
    """Plot Kaplan-Meier survival curves for low and high risk groups with median survival annotations."""
    plt.figure(figsize=(10, 6))
    kmf_low.fit(y_test_durations[low_risk_idx], event_observed=y_test_events[low_risk_idx], label='Low Risk')
    ax = kmf_low.plot(ci_show=True, color='green', lw=2)
    kmf_high.fit(y_test_durations[high_risk_idx], event_observed=y_test_events[high_risk_idx], label='High Risk')
    kmf_high.plot(ci_show=True, ax=ax, color='red', lw=2)
    plt.title("Kaplan-Meier Survival Curves by Risk Group", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    # Annotate median survival times
    median_low = kmf_low.median_survival_time_
    median_high = kmf_high.median_survival_time_
    plt.text(0.6, 0.2, f"Median Low Risk: {median_low:.2f}", transform=ax.transAxes, color='green', fontsize=12)
    plt.text(0.6, 0.1, f"Median High Risk: {median_high:.2f}", transform=ax.transAxes, color='red', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "km_risk_stratification.png"))
    plt.close()


def plot_calibration_plot(surv, fixed_time, predicted_surv_probs, y_test_durations, y_test_events, output_dir):
    """Plot calibration curve at a fixed time point comparing predicted vs observed survival probabilities."""
    decile_bins = np.percentile(predicted_surv_probs, np.arange(0, 110, 10))
    bin_indices = np.digitize(predicted_surv_probs, decile_bins, right=True)
    observed_probs = []
    predicted_avg = []
    for i in range(1, 11):
        idx = bin_indices == i
        if np.sum(idx) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(y_test_durations[idx], event_observed=y_test_events[idx])
            observed_probs.append(kmf.predict(fixed_time))
            predicted_avg.append(np.mean(predicted_surv_probs[idx]))
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_avg, observed_probs, 'o-', lw=2, label='Calibration Curve')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal Calibration')
    plt.xlabel(f"Predicted Survival Probability at {fixed_time} months", fontsize=12)
    plt.ylabel("Observed Survival Probability", fontsize=12)
    plt.title(f"Calibration Plot at {fixed_time} Months", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_plot.png"))
    plt.close()


def plot_multi_calibration(surv, y_test_durations, y_test_events, time_points, output_dir):
    """Plot calibration curves at multiple time points."""
    plt.figure(figsize=(15, 5))
    for idx, t in enumerate(time_points):
        original_time = t
        if t not in surv.index:
            nearest_idx = np.abs(surv.index - t).argmin()
            t = surv.index[nearest_idx]
            print(f"Time {original_time} not found; using nearest time {t} instead.")
        predicted_probs = surv.loc[t].values
        decile_bins = np.percentile(predicted_probs, np.arange(0, 110, 10))
        bin_indices = np.digitize(predicted_probs, decile_bins, right=True)
        observed_probs = []
        predicted_avg = []
        for i in range(1, 11):
            idx_bin = bin_indices == i
            if np.sum(idx_bin) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(y_test_durations[idx_bin], event_observed=y_test_events[idx_bin])
                observed_probs.append(kmf.predict(t))
                predicted_avg.append(np.mean(predicted_probs[idx_bin]))
        plt.subplot(1, len(time_points), idx + 1)
        plt.plot(predicted_avg, observed_probs, 'o-', lw=2, label='Calibration Curve')
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal Calibration')
        plt.xlabel(f"Predicted Prob at {t} months", fontsize=10)
        plt.ylabel("Observed Survival Prob", fontsize=10)
        plt.title(f"Calibration at {t} months", fontsize=12)
        plt.legend(fontsize=8)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_calibration_plot.png"))
    plt.close()


def plot_cumulative_hazard(surv, output_dir):
    """
    Plot cumulative hazard functions computed from the survival function.
    The cumulative hazard is given by H(t) = -log(S(t)).
    """
    plt.figure(figsize=(10, 6))
    cumulative_hazard = -np.log(surv)
    # Plot cumulative hazard for a few sample patients for clarity
    for i in range(min(10, cumulative_hazard.shape[1])):
        plt.plot(cumulative_hazard.index, cumulative_hazard.iloc[:, i],
                 lw=2, alpha=0.7, label=f"Sample {i}" if i < 5 else None)
    plt.title("Cumulative Hazard Functions", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Cumulative Hazard", fontsize=12)
    plt.grid(True)
    plt.legend(title="Sample Index", fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_hazard.png"))
    plt.close()


def plot_survival_probability_distribution(surv, output_dir, time_point=None):
    """
    Plot the distribution of predicted survival probabilities at a specific time point.
    If time_point is None, the last time point in the survival function is used.
    """
    if time_point is None:
        time_point = surv.index[-1]
    elif time_point not in surv.index:
        nearest_idx = np.abs(surv.index - time_point).argmin()
        time_point = surv.index[nearest_idx]
        print(f"Specified time_point not found; using nearest time {time_point} instead.")
    predicted_probs = surv.loc[time_point].values
    plt.figure(figsize=(10, 6))
    sns.histplot(predicted_probs, bins=30, kde=True, color='purple', edgecolor='k')
    plt.title(f"Distribution of Predicted Survival Probabilities at {time_point} Time Units", fontsize=14)
    plt.xlabel("Predicted Survival Probability", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"survival_probability_distribution_{time_point}.png"))
    plt.close()

def plot_cv_metrics(fold_stats, output_dir, figsize=(14, 8), dpi=100):
    """
    Plot cross-validation metrics with improved visualization:
      - Bar plot of the concordance index per fold with mean line
      - Bar plot of the event rate for both training and testing data per fold
      - Summary statistics table
      
    Parameters:
    -----------
    fold_stats : list of dict
        List of dictionaries containing fold statistics
    output_dir : str
        Directory to save the output plots
    figsize : tuple, optional
        Figure size in inches (width, height)
    dpi : int, optional
        Resolution of the output figure
    """
    ensure_dir(output_dir)
    folds = [stat["fold"] for stat in fold_stats]
    concordances = [stat["concordance"] for stat in fold_stats]
    train_event_rates = [stat["train_events"] / stat["train_total"] for stat in fold_stats]
    test_event_rates = [stat["test_events"] / stat["test_total"] for stat in fold_stats]
    
    # Calculate summary statistics
    mean_concordance = np.mean(concordances)
    std_concordance = np.std(concordances)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    # Plot Concordance Index per fold
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(folds, concordances, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add mean line
    ax1.axhline(y=mean_concordance, color='red', linestyle='--', 
                label=f'Mean: {mean_concordance:.3f}')
    
    # Add labels to bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel("Fold", fontsize=11)
    ax1.set_ylabel("Concordance Index", fontsize=11)
    ax1.set_title("Concordance Index per Fold", fontsize=13, fontweight='bold')
    ax1.set_ylim(max(0, min(concordances) - 0.1), min(1, max(concordances) + 0.1))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot Event Rates per fold for train and test
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    indices = np.arange(len(folds))
    
    train_bars = ax2.bar(indices - width/2, train_event_rates, width=width, 
                         color='forestgreen', label='Train Event Rate', alpha=0.8, edgecolor='black')
    test_bars = ax2.bar(indices + width/2, test_event_rates, width=width, 
                        color='orangered', label='Test Event Rate', alpha=0.8, edgecolor='black')
    
    # Add labels to bars
    for bars, rates in [(train_bars, train_event_rates), (test_bars, test_event_rates)]:
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel("Fold", fontsize=11)
    ax2.set_ylabel("Event Rate", fontsize=11)
    ax2.set_title("Event Rates per Fold", fontsize=13, fontweight='bold')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(folds)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add summary statistics table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Calculate additional statistics
    mean_train_rate = np.mean(train_event_rates)
    mean_test_rate = np.mean(test_event_rates)
    max_concordance = np.max(concordances)
    min_concordance = np.min(concordances)
    
    table_data = [
        ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
        ['Concordance', f'{mean_concordance:.3f}', f'{std_concordance:.3f}', 
         f'{min_concordance:.3f}', f'{max_concordance:.3f}'],
        ['Train Event Rate', f'{mean_train_rate:.3f}', f'{np.std(train_event_rates):.3f}', 
         f'{np.min(train_event_rates):.3f}', f'{np.max(train_event_rates):.3f}'],
        ['Test Event Rate', f'{mean_test_rate:.3f}', f'{np.std(test_event_rates):.3f}', 
         f'{np.min(test_event_rates):.3f}', f'{np.max(test_event_rates):.3f}'],
    ]
    
    table = ax3.table(cellText=table_data, loc='center', cellLoc='center', 
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Save high-resolution figure
    fig_path = os.path.join(output_dir, "cv_metrics.png")
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')

    
    plt.close()
    
    return fig_path