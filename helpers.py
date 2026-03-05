import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Any
from helpers import run_isolation_forest_detection

def visualize_regime_segments(time_series,
                              change_points=None,
                              plot_title="Time Series Regime Segmentation"):

    data_series = time_series.dropna().sort_index()

    if change_points is None:
        change_points = []
    else:
        change_points = [pd.to_datetime(cp) for cp in change_points]

    valid_changes = sorted(
        [cp for cp in change_points
         if data_series.index.min() < cp < data_series.index.max()]
    )

    segment_limits = [data_series.index.min()] + valid_changes + [data_series.index.max()]

    plt.figure(figsize=(18, 6))
    plt.plot(data_series.index,
             data_series.values,
             linewidth=1,
             label="Observed Series", color="blue")

    for cp in valid_changes:
        plt.axvline(cp,
                    linestyle="--",
                    linewidth=1)

    for idx in range(len(segment_limits) - 1):

        start_date = segment_limits[idx]
        end_date   = segment_limits[idx + 1]

        segment_data = data_series.loc[
            (data_series.index >= start_date) &
            (data_series.index < end_date)
        ]

        if segment_data.empty:
            continue

        segment_mean = segment_data.mean()

        print(f"Segment Mean {idx + 1}: {segment_mean}")

        plt.hlines(segment_mean,
                   xmin=segment_data.index.min(),
                   xmax=segment_data.index.max(),
                   linewidth=3,
                   color="red",
                   label="Segment Mean" if idx == 0 else None)

    plt.title(plot_title)
    plt.xlabel("Per Day")
    plt.ylabel("CO₂ Emissions")
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_piecewise_linear_fit(series_input,
                                   change_dates=None,
                                   figure_title="Segmented Linear Trend"):

    cleaned_series = series_input.dropna().sort_index()

    if change_dates is None:
        change_dates = []
    else:
        change_dates = [pd.to_datetime(dt) for dt in change_dates]

    valid_points = sorted(
        [dt for dt in change_dates
         if cleaned_series.index.min() < dt < cleaned_series.index.max()]
    )

    interval_edges = [cleaned_series.index.min()] + valid_points + [cleaned_series.index.max()]

    plt.figure(figsize=(18, 6))
    plt.plot(cleaned_series.index,
             cleaned_series.values,
             linewidth=1,
             label="Observed Data",
             color="blue")

    for dt in valid_points:
        plt.axvline(dt,
                    linestyle="--",
                    linewidth=1)

    for j in range(len(interval_edges) - 1):

        start_bound = interval_edges[j]
        end_bound   = interval_edges[j + 1]

        segment_series = cleaned_series.loc[
            (cleaned_series.index >= start_bound) &
            (cleaned_series.index < end_bound)
        ]

        if len(segment_series) < 5:
            continue

        time_index = (segment_series.index - segment_series.index.min()).days.astype(float)
        values = segment_series.values.astype(float)

        slope, intercept = np.polyfit(time_index, values, 1)
        fitted_values = intercept + slope * time_index

        plt.plot(segment_series.index,
                 fitted_values,
                 linewidth=3,
                 color="green",
                 label="Segment Linear Trend" if j == 0 else None)

    plt.title(figure_title)
    plt.xlabel("Per Day")
    plt.ylabel("CO₂ Emissions")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_seasonal_pattern(data_frame,
                          season_unit='year',
                          period_unit='month',
                          value_column=None,
                          plot_title = ""):

    if value_column is None:
        value_column = data_frame.columns[0]

    period_values = getattr(data_frame.index, period_unit)
    season_values = getattr(data_frame.index, season_unit)

    seasonal_table = pd.pivot_table(
        data_frame,
        index=period_values,
        columns=season_values,
        values=value_column
    )

    seasonal_table.plot(figsize=(12, 8))

    plt.title(plot_title)
    plt.xlabel("Per Month")
    plt.ylabel("CO₂ Emissions")
    plt.legend(title=season_unit.capitalize())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def monthly_subseries_visualization(input_df,
                                     month_field='month',
                                     year_field='year',
                                     emission_field='value',
                                     plot_title=""):

    monthly_summary = (
        input_df
        .groupby([month_field, year_field])[emission_field]
        .mean()
        .reset_index()
    )

    fig, subplot_array = plt.subplots(3, 4, figsize=(20, 15), sharey=True)
    subplot_array = subplot_array.flatten()
    fig.suptitle(plot_title, fontsize=16)

    month_names_list = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

    for position, month_number in enumerate(range(1, 13)):
        current_axis = subplot_array[position]

        monthly_data = monthly_summary[
            monthly_summary[month_field] == month_number
        ]

        sns.lineplot(
            x=year_field,
            y=emission_field,
            data=monthly_data,
            ax=current_axis,
            marker='o'
        )

        overall_month_avg = monthly_data[emission_field].mean()

        current_axis.axhline(
            overall_month_avg,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Mean: {overall_month_avg:.3f}'
        )

        current_axis.set_title(month_names_list[month_number - 1])
        current_axis.tick_params(axis='x', rotation=45)
        current_axis.set_xlabel('Year')
        current_axis.set_ylabel('CO₂ Emissions')
        current_axis.grid(True, linestyle='--', alpha=0.6)
        current_axis.legend()

    plt.tight_layout()
    plt.show()

def visualize_acf_with_fixed_bounds(data_series,
                                    lag_count=50,
                                    plot_heading="Autocorrelation Function (ACF)",
                                    conf_level=0.95):

    cleaned_series = data_series.dropna()

    total_obs = len(cleaned_series)

    if conf_level == 0.95:
        critical_z = 1.96
    elif conf_level == 0.99:
        critical_z = 2.576
    elif conf_level == 0.90:
        critical_z = 1.645
    else:
        raise ValueError("Allowed confidence levels: 0.90, 0.95, 0.99")

    bound_limit = critical_z / np.sqrt(total_obs)

    fig_container, axis_container = plt.subplots(figsize=(16, 8))

    plot_acf(
        cleaned_series,
        lags=lag_count,
        alpha=None,
        ax=axis_container
    )

    axis_container.axhline(bound_limit,
                           color='red',
                           linestyle='--',
                           linewidth=1.5,
                           label=f'+{int(conf_level*100)}% CI ({bound_limit:.3f})')

    axis_container.axhline(-bound_limit,
                           color='red',
                           linestyle='--',
                           linewidth=1.5,
                           label=f'-{int(conf_level*100)}% CI ({-bound_limit:.3f})')

    axis_container.set_title(plot_heading)
    axis_container.set_ylim(-1.05, 1.05)
    axis_container.legend()

    plt.tight_layout()
    plt.show()

def visualize_time_series_with_moving_statistics(data_frame,
                                                  target_column="value",
                                                  moving_window=52,
                                                  figure_title="Rolling Statistics",
                                                  y_axis_label="CO₂ Emissions",
                                                  x_axis_label="Per Day",
                                                  figure_dimensions=(15, 8),
                                                  display_series=True,
                                                  display_mean=True,
                                                  display_std=True):

    cleaned_values = data_frame[target_column].dropna()

    fig_canvas, axis_canvas = plt.subplots(nrows=1, ncols=1, figsize=figure_dimensions)

    if display_series:
        sns.lineplot(
            x=cleaned_values.index,
            y=cleaned_values.values,
            ax=axis_canvas,
            color="blue",
            label="Observed Values"
        )

    if display_mean:
        moving_average = cleaned_values.rolling(moving_window).mean()
        sns.lineplot(
            x=moving_average.index,
            y=moving_average.values,
            ax=axis_canvas,
            color="red",
            label=f"Moving Average (Window={moving_window})"
        )

    if display_std:
        moving_std_dev = cleaned_values.rolling(moving_window).std()
        sns.lineplot(
            x=moving_std_dev.index,
            y=moving_std_dev.values,
            ax=axis_canvas,
            color="green",
            label=f"Moving Std Dev (Window={moving_window})"
        )

    axis_canvas.set_title(figure_title, fontsize=14)
    axis_canvas.set_ylabel(y_axis_label, fontsize=14)
    axis_canvas.set_xlabel(x_axis_label, fontsize=14)
    axis_canvas.legend()

    axis_canvas.set_xlim([cleaned_values.index.min(),
                          cleaned_values.index.max()])

    plt.tight_layout()
    plt.show()

def custom_boxplot_layout(data_frame,
                            feature_name="value",
                            main_title="Box Plot",
                            vertical_label="CO₂ Emissions",
                            canvas_size=(10, 7),
                            fill_color="#BFDCE5"):

    observations = data_frame[feature_name].dropna().to_numpy(dtype=float)

    if observations.size == 0:
        raise ValueError(f"No valid values found in column '{feature_name}'.")

    p25 = np.percentile(observations, 25)
    p50 = np.percentile(observations, 50)
    p75 = np.percentile(observations, 75)
    interquartile_span = p75 - p25

    lower_limit = p25 - 1.5 * interquartile_span
    upper_limit = p75 + 1.5 * interquartile_span

    whisker_min = np.min(observations[observations >= lower_limit])
    whisker_max = np.max(observations[observations <= upper_limit])

    extreme_low  = observations[observations < lower_limit]
    extreme_high = observations[observations > upper_limit]

    anomaly_total = extreme_low.size + extreme_high.size
    anomaly_ratio = 100.0 * anomaly_total / observations.size

    percent_below_p25 = 100.0 * np.mean(observations < p25)
    percent_below_p75 = 100.0 * np.mean(observations < p75)
    percent_middle    = 100.0 * np.mean((observations >= p25) & (observations <= p75))

    value_range = observations.max() - observations.min()
    offset = value_range * 0.07 if value_range > 0 else 1e-6

    fig_obj, axis_obj = plt.subplots(figsize=canvas_size)

    sns.boxplot(
        y=observations,
        ax=axis_obj,
        color=fill_color,
        width=0.35,
        linewidth=1,
        medianprops=dict(color='blue', linewidth=2),
        meanprops=dict(color='green', linewidth=2),
        flierprops=dict(
            marker='o',
            markerfacecolor='red',
            markeredgecolor='red',
            markersize=5,
            alpha=0.8
        )
    )

    axis_obj.set_title(main_title, fontsize=14, pad=18)
    axis_obj.set_ylabel(vertical_label, fontsize=12)
    axis_obj.set_xticks([])
    axis_obj.grid(axis="y", linestyle="--", alpha=0.25)

    summary_line = (
        f"Outliers: {anomaly_total} ({anomaly_ratio:.2f}%) | "
        f"Lower: {extreme_low.size} | Upper: {extreme_high.size}"
    )

    fig_obj.text(0.5, 0.93,
                 summary_line,
                 ha="center",
                 va="center",
                 fontsize=11,
                 color="red",
                 fontweight="bold")

    axis_obj.set_ylim(observations.min() - offset * 2.0,
                      observations.max() + offset * 3.0)

    axis_obj.text(0,
                  p50 + offset * 0.50,
                  f"Median (Q2) = {p50:.4f}",
                  ha="center",
                  va="center",
                  fontsize=11,
                  color="blue",
                  fontweight="bold")

    axis_obj.text(0,
                  p50 - offset * 0.80,
                  f"IQR = {interquartile_span:.4f}\n{percent_middle:.1f}% inside IQR",
                  ha="center",
                  va="center",
                  fontsize=10)

    def side_annotation(y_point, text_label, x_position, y_position):
        axis_obj.annotate(
            text_label,
            xy=(0, y_point),
            xytext=(x_position, y_position),
            textcoords="data",
            ha="left" if x_position > 0 else "right",
            va="center",
            fontsize=10,
            arrowprops=dict(
                arrowstyle="-",
                lw=1,
                color="orange",
                connectionstyle="angle3,angleA=0,angleB=90"
            ),
        )

    right_anchor = 0.22
    left_anchor  = -0.22

    side_annotation(p75,
                    f"Q3 = {p75:.4f}\n{percent_below_p75:.1f}% below Q3",
                    right_anchor,
                    p75 + offset * 0.50)

    side_annotation(p25,
                    f"Q1 = {p25:.4f}\n{percent_below_p25:.1f}% below Q1",
                    right_anchor,
                    p25 - offset * 0.50)

    side_annotation(whisker_max,
                    f"Upper whisker = {whisker_max:.4f}",
                    left_anchor,
                    whisker_max + offset * 0.25)

    side_annotation(whisker_min,
                    f"Lower whisker = {whisker_min:.4f}",
                    left_anchor,
                    whisker_min - offset * 0.25)

    if extreme_low.size > 0:
        lowest_outlier_value = extreme_low.min()

        axis_obj.annotate(
            f"Lower outliers = {extreme_low.size}",
            xy=(0, lowest_outlier_value),
            xytext=(-0.28, lowest_outlier_value - offset * 0.8),
            textcoords="data",
            ha="right",
            va="center",
            fontsize=10,
            color="black",
            arrowprops=dict(
                arrowstyle="-",
                lw=1,
                color="orange",
                connectionstyle="angle3,angleA=0,angleB=90"
            ),
        )

    if extreme_high.size > 0:
        highest_outlier_value = extreme_high.max()

        axis_obj.annotate(
            f"Upper outliers = {extreme_high.size}",
            xy=(0, highest_outlier_value),
            xytext=(0.28, highest_outlier_value + offset * 0.8),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            arrowprops=dict(
                arrowstyle="-",
                lw=1,
                color="orange",
                connectionstyle="angle3,angleA=0,angleB=90"
            ),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    return {
        "Q1": p25,
        "Median": p50,
        "Q3": p75,
        "IQR": interquartile_span,
        "LowerWhisker": whisker_min,
        "UpperWhisker": whisker_max,
        "OutliersTotal": anomaly_total,
        "OutlierPercent": anomaly_ratio
    }

def run_isolation_forest_detection(
    input_dataframe,
    target_feature="value",
    num_trees=100,
    sample_limit=256,
    anomaly_fraction=0.10,
    seed_value=42
):

    processed_df = input_dataframe.copy().sort_index()

    iso_model = IsolationForest(
        n_estimators=num_trees,
        max_samples=sample_limit,
        contamination=anomaly_fraction,
        random_state=seed_value
    )

    processed_df["anomaly"] = iso_model.fit_predict(
        processed_df[[target_feature]]
    )

    processed_df["anomaly_score"] = iso_model.decision_function(
        processed_df[[target_feature]]
    )

    processed_df["is_anomaly"] = processed_df["anomaly"] == -1

    detected_points = processed_df[processed_df["is_anomaly"]]
    anomaly_percentage = (len(detected_points) / len(processed_df)) * 100

    print(
        f"Isolation Forest detected {len(detected_points)} anomalies "
        f"({anomaly_percentage:.2f}% of total observations).\n"
    )

    return processed_df, anomaly_percentage

def visualize_anomaly_scores(
    input_df,
    score_column="anomaly_score",
    label_column="anomaly",
    time_axis=None,
    figure_title="Isolation Forest Anomaly Score Visualization",
    canvas_dim=(10, 5)
):

    data_copy = input_df.copy()

    if time_axis is None:
        time_axis = data_copy.index

    regular_points = data_copy[data_copy[label_column] == 1]

    abnormal_points = data_copy[data_copy[label_column] == -1]

    plt.figure(figsize=canvas_dim)

    plt.scatter(
        regular_points.index,
        regular_points[score_column],
        color="blue",
        label="Regular Observations",
        alpha=0.7
    )

    plt.scatter(
        abnormal_points.index,
        abnormal_points[score_column],
        color="red",
        label="Detected Anomalies",
        alpha=0.9
    )

    plt.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.6)

    plt.xlabel("Instances (CO2 Emissions)")
    plt.ylabel("Anomaly Score")
    plt.title(figure_title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def initialize_random_state(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def scale_using_baseline(raw_series, baseline_cutoff, min_required_points):

    baseline_timestamp = pd.to_datetime(baseline_cutoff)
    baseline_mask = raw_series.index <= baseline_timestamp

    if baseline_mask.sum() < min_required_points:
        raise ValueError("Not enough baseline points.")

    values_2d = raw_series.values.reshape(-1, 1).astype(np.float64)

    baseline_mean = float(values_2d[baseline_mask].mean())
    baseline_std = float(values_2d[baseline_mask].std())

    print("Before Scaling (Baseline Only):")
    print("Baseline Mean:", baseline_mean)
    print("Baseline Std :", baseline_std)

    scaler = StandardScaler()
    scaler.fit(values_2d[baseline_mask])
    scaled_values = scaler.transform(values_2d)

    scaled_series = pd.Series(scaled_values.flatten(), index=raw_series.index)
    print("\nAfter Scaling (Baseline Only):")
    print("Baseline Mean:", float(scaled_series[baseline_mask].mean()))
    print("Baseline Std :", float(scaled_series[baseline_mask].std()),"\n")

    return scaled_values, baseline_timestamp


def create_sequence_windows(scaled_values, index, window_length):
    windows = []
    for t in range(window_length - 1, len(scaled_values)):
        windows.append(scaled_values[t - window_length + 1 : t + 1])
    return np.stack(windows), index[window_length - 1:]


def extract_baseline_windows(all_windows, window_end_dates, baseline_timestamp):
    return all_windows[window_end_dates <= baseline_timestamp]


class LSTMSequenceAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        context = hidden[-1]
        repeated = context.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)

def train_lstm_autoencoder(baseline_windows, hidden_dim, epochs, batch_size, lr):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSequenceAutoencoder(hidden_dim=hidden_dim).to(device)

    tensor_data = torch.tensor(baseline_windows, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor_data, tensor_data),
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epoch_mse_data = []

    model.train()
    for ep in range(1, epochs + 1):
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        epoch_mse_data.append([ep, np.mean(losses)])

    epoch_mse_df = pd.DataFrame(epoch_mse_data, columns=['EPOCH', 'MSE'])

    return model, epoch_mse_df

def compute_sequences_dataframe(model, all_windows, window_end_dates):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    tensor_data = torch.tensor(all_windows, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(tensor_data).cpu().numpy()

    mse = np.mean((reconstructed - all_windows) ** 2, axis=(1, 2))

    df_sequences = pd.DataFrame(
        {
            "window_end_date": pd.to_datetime(window_end_dates),
            "input_sequence": list(all_windows.squeeze(-1)),
            "reconstructed_sequence": list(reconstructed.squeeze(-1)),
            "reconstruction_error": mse
        }
    ).set_index("window_end_date")

    return df_sequences

def compute_robust_zscore(reconstruction_error, baseline_cutoff):

    baseline_timestamp = pd.to_datetime(baseline_cutoff)

    baseline_values = reconstruction_error[reconstruction_error.index <= baseline_timestamp]

    median = float(baseline_values.median())
    mad = float(np.median(np.abs(baseline_values - median)))

    robust_sigma = 1.4826 * mad + 0.000000000001

    robust_z = (reconstruction_error - median) / robust_sigma

    return robust_z, median, robust_sigma

def detect_break_start_dates(z_score, z_threshold, min_consecutive_windows):
    above = (z_score > z_threshold)

    above_cp = above.copy()
    display(above_cp.astype(int))
    display(above_cp.rolling(min_consecutive_windows).sum())

    display(above.astype(int)
                  .rolling(min_consecutive_windows)
                  .sum().fillna(False).head(35))

    persistent = (above.astype(int)
                  .rolling(min_consecutive_windows)
                  .sum()
                  .ge(min_consecutive_windows)
                  .fillna(False))

    break_dates = []
    inside = False
    idx = z_score.index

    for t, flag in persistent.items():
        if flag and not inside:
            end_pos = idx.get_loc(t)
            start_pos = end_pos - (min_consecutive_windows - 1)
            start = idx[start_pos]
            break_dates.append(pd.to_datetime(start))
            inside = True
        elif not flag:
            inside = False

    return break_dates

def detect_structural_breaks(
    dataframe,
    target_column="value",
    baseline_cutoff="2020-03-01",
    window_length=7,
    hidden_dim=64,
    epochs=30,
    batch_size=128,
    learning_rate=0.001,
    smoothing_window=7,
    z_threshold=3.0,
    min_consecutive_days=7,
    seed=42,
):

    initialize_random_state(seed)

    raw_series = dataframe[target_column].astype(np.float64).sort_index()

    scaled_values, baseline_timestamp = scale_using_baseline(
        raw_series,
        baseline_cutoff,
        min_required_points=window_length + 60
    )

    all_windows, window_end_dates = create_sequence_windows(
        scaled_values,
        raw_series.index,
        window_length
    )

    baseline_windows = extract_baseline_windows(
        all_windows,
        window_end_dates,
        baseline_timestamp
    )

    model, epoch_mse_df = train_lstm_autoencoder(
        baseline_windows=baseline_windows,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate
    )

    df_sequences = compute_sequences_dataframe(
        model=model,
        all_windows=all_windows,
        window_end_dates=window_end_dates
    )

    reconstruction_error = df_sequences["reconstruction_error"]

    robust_z, center, scale = compute_robust_zscore(
        reconstruction_error=reconstruction_error,
        baseline_cutoff=baseline_cutoff
    )

    df_sequences['robust_z'] = robust_z

    classification = (robust_z > z_threshold)
    df_sequences['classification'] = classification.astype(int)

    persistent = (classification.astype(int)
                  .rolling(min_consecutive_days)
                  .sum()
                  .ge(min_consecutive_days)
                  .fillna(False))

    break_dates = []
    inside = False
    idx = robust_z.index

    for t, flag in persistent.items():
        if flag and not inside:
            end_pos = idx.get_loc(t)
            start_pos = end_pos - (min_consecutive_days - 1)
            start = idx[start_pos]
            break_dates.append(pd.to_datetime(start))
            inside = True
        elif not flag:
            inside = False

    return {
        "df_sequences": df_sequences,
        "reconstruction_error": reconstruction_error,
        "robust_z": robust_z,
        "break_dates": break_dates,
        "baseline_center": center,
        "baseline_scale": scale,
        "z_threshold": z_threshold,
        "epoch_mse_df": epoch_mse_df
    }

