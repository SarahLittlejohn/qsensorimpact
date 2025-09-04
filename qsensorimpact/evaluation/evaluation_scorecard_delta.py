from generation.parity_data import generate_parity_series, generate_parity_series_dynamic, generate_parity_series_with_noise, generate_e2e_parity_series_with_noise
from generation.simulation_data import find_static_switching_rate_clean_series, find_static_switching_rate_noisy_series, find_dynamic_switching_rates_noisy_series, estimate_switching_rate_with_resampling, estimate_switching_rate_with_resampling_hmm
from generation.tools import generate_gaussian_matrix, generate_gaussian_matrix_variable_impact, generate_2d_gaussian_matrix_single_impact_time_independent, generate_2d_time_dependent_gaussian_matrix_single_impact, generate_2d_time_dependent_delta_impact, generate_2d_switching_rate_tensor_from_gamma
from analysis.one_d_impact import analyse_one_d_impact
from analysis.two_d_impact import analyse_two_d_impact_snapshot, analyse_two_d_impact, analyse_with_detection, analyse_quartiled_qubits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import pandas as pd
from pathlib import Path
weights = Path(__file__).parent.parent / "weights" / "best.pt"
weights = str(weights.resolve())

# Parameters repeated through each region
snapshots = 50
grid_size = 25
baseline = 25
total_snapshots = 50
t_1 = 25
noise_std = 0.05
total_snapshots = t_1 + snapshots
true_x = 12.5
true_y = 12.5
true_t = t_1

# region 1. Ratio of impacts
ratios = [i/50 for i in range(1, 14)]

def build_tensor_with_noise(rng, ratio):
    sigma = ratio * grid_size
    length_impact = sigma**2
    raw = generate_2d_time_dependent_delta_impact(baseline, 7, 12.5, 12.5, snapshots, grid_size,  spatial_spread=sigma)
    total = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)
    total[t_1:t_1 + snapshots] -= (baseline - raw)
    noise = rng.normal(loc=0.0, scale=noise_std, size=total.shape)
    return total + noise

detailed_rows = [] 
summary_rows  = [] 

for seed in range(100):
    print(f"SEED: {seed}")
    rng = np.random.default_rng(seed)

    for ratio in ratios:
        tensor = build_tensor_with_noise(rng, ratio)

        centers, min_t_by_cluster = analyse_with_detection(
            tensor,
            weights,
            grid_size=grid_size,
            conf_thres=0.005
        )

        if not centers:
            summary_rows.append({
                "seed": seed,
                "ratio": ratio,
                "mean_abs_dx": np.nan,
                "mean_abs_dy": np.nan,
                "mean_abs_dt": np.nan,
                "clusters_recalled": 0
            })
            detailed_rows.append({
                "seed": seed,
                "ratio": ratio,
                "cluster_label": None,
                "center_x": np.nan,
                "center_y": np.nan,
                "center_t": np.nan,
                "min_t": np.nan,
                "abs_dx": np.nan,
                "abs_dy": np.nan,
                "abs_dt": np.nan,
                "clusters_recalled": 0
            })
            continue

        abs_dx_list, abs_dy_list, abs_dt_list = [], [], []
        k = len(centers)

        for label, (cx, cy, ct) in centers.items():
            mt = min_t_by_cluster.get(label, np.nan)
            abs_dx = abs(cx - true_x)
            abs_dy = abs(cy - true_y)
            abs_dt = abs(ct - true_t)

            detailed_rows.append({
                "seed": seed,
                "ratio": ratio,
                "cluster_label": label,
                "center_x": cx,
                "center_y": cy,
                "center_t": ct,
                "min_t": mt,
                "abs_dx": abs_dx,
                "abs_dy": abs_dy,
                "abs_dt": abs_dt,
                "clusters_recalled": k
            })

            abs_dx_list.append(abs_dx)
            abs_dy_list.append(abs_dy)
            abs_dt_list.append(abs_dt)

        summary_rows.append({
            "seed": seed,
            "ratio": ratio,
            "mean_abs_dx": float(np.mean(abs_dx_list)) if abs_dx_list else np.nan,
            "mean_abs_dy": float(np.mean(abs_dy_list)) if abs_dy_list else np.nan,
            "mean_abs_dt": float(np.mean(abs_dt_list)) if abs_dt_list else np.nan,
            "clusters_recalled": k
        })

df_detail = pd.DataFrame(detailed_rows)
df_detail.to_csv("delta_ratio_yolo_detail.csv", index=False)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("delta_ratio_yolo_summary.csv", index=False)

print("Saved delta_ratio.csv and delta_ratio_summary.csv")

# endregion

# region 2. Cluster Seperation
# Helper to make tensor pairs
def make_delta_tensor_pair(baseline, snapshots, grid_size, noise_std,
                           x1, y1, x2, y2, t_start, ratio, rng):
    sigma = ratio * grid_size
    tensor1 = generate_2d_time_dependent_delta_impact(
        baseline, 7, x1, y1, snapshots, grid_size, spatial_spread=sigma
    )
    tensor2 = generate_2d_time_dependent_delta_impact(
        baseline, 7, x2, y2, snapshots, grid_size, spatial_spread=sigma
    )

    drop1 = baseline - tensor1
    drop2 = baseline - tensor2
    combined_drop = np.maximum(drop1, drop2)

    total_tensor = np.full((total_snapshots, grid_size, grid_size),
                           baseline, dtype=np.float64)
    total_tensor[t_start:t_start+snapshots] -= combined_drop

    noise = rng.normal(0.0, noise_std, size=total_tensor.shape)
    return total_tensor + noise

anchor = (7, 7)          
k_values = list(range(17, 0, -1)) 
ratio = 0.08
num_seeds = 100         

rows = []

for k in k_values:
    x1, y1 = anchor
    x2, y2 = x1 + k, y1 + k  
    D = np.sqrt(2.0) * k

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)

        tensor = make_delta_tensor_pair(
            baseline, snapshots, grid_size, noise_std,
            x1, y1, x2, y2, t_1, ratio, rng
        )

        centers, min_t_by_cluster = analyse_with_detection(
            tensor,
            weights,
            grid_size=grid_size,
            conf_thres=0.005
        )

        if centers:
            clusters_recalled = len(centers)
            for label, (cx, cy, ct) in centers.items():
                mt = min_t_by_cluster.get(label, np.nan)
                rows.append({
                    "k": k,
                    "D": D,
                    "seed": seed,
                    "ratio": ratio,
                    "cluster_label": label,
                    "center_x": cx,
                    "center_y": cy,
                    "center_t": ct,
                    "min_t": mt,
                    "clusters_recalled": clusters_recalled,
                })
        else:
            rows.append({
                "k": k,
                "D": D,
                "seed": seed,
                "ratio": ratio,
                "cluster_label": None,
                "center_x": float("nan"),
                "center_y": float("nan"),
                "center_t": float("nan"),
                "min_t": float("nan"),
                "clusters_recalled": 0,
            })

df_delta = pd.DataFrame(rows).sort_values(["k", "seed", "cluster_label"])
df_delta.to_csv("delta_yolo_outputs_all_distances.csv", index=False)
print("Saved delta_yolo_outputs_all_distances.csv")

def frac(series, value_or_pred):
    if callable(value_or_pred):
        return (series.map(value_or_pred)).mean()
    return (series == value_or_pred).mean()

summary_delta = (
    df_delta.groupby(["k", "D"], as_index=False)
            .agg(
                mean_clusters=("clusters_recalled", "mean"),
                median_clusters=("clusters_recalled", "median"),
                frac_1=("clusters_recalled", lambda s: frac(s, 1)),
                frac_2=("clusters_recalled", lambda s: frac(s, 2)),
                frac_ge3=("clusters_recalled", lambda s: frac(s, lambda x: x >= 3)),
            )
            .sort_values("k", ascending=False)
)

summary_delta.to_csv("delta_yolo_summary_by_distance.csv", index=False)
print("Saved delta_yolo_summary_by_distance.csv")
print(summary_delta.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
# endregion

# region 3. Number of impacts

ratio = 0.08
sigma = ratio * grid_size

tensor_1 = generate_2d_time_dependent_delta_impact(baseline, 7, 12.5, 12.5, snapshots, grid_size, sigma)
tensor_2 = generate_2d_time_dependent_delta_impact(baseline, 7, 9, 8, snapshots, grid_size, sigma)
tensor_3 = generate_2d_time_dependent_delta_impact(baseline, 7, 16, 17, snapshots, grid_size, sigma)
tensor_4 = generate_2d_time_dependent_delta_impact(baseline, 7, 4, 20, snapshots, grid_size, sigma)
tensor_5 = generate_2d_time_dependent_delta_impact(baseline, 7, 21, 5, snapshots, grid_size, sigma)
tensor_6 = generate_2d_time_dependent_delta_impact(baseline, 7, 20, 21, snapshots, grid_size, sigma)

drop1 = baseline - tensor_1
drop2 = baseline - tensor_2
drop3 = baseline - tensor_3
drop4 = baseline - tensor_4
drop5 = baseline - tensor_5
drop6 = baseline - tensor_6

SCENARIOS = {
    "ONE":   [drop1],
    "TWO":   [drop2, drop3],
    "THREE": [drop2, drop3, drop4],
    "FOUR":  [drop2, drop3, drop4, drop5],
    "FIVE":  [drop2, drop3, drop4, drop5, drop6],
}

def compose_tensor_from_drops(drops):
    """Compose a total tensor for the time window using max of provided drops (no noise yet)."""
    combined_drop = drops[0] if len(drops) == 1 else np.maximum.reduce(drops)
    total = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)
    total[t_1:t_1+snapshots] -= combined_drop
    return total

BASE_TENSORS = {name: compose_tensor_from_drops(drops) for name, drops in SCENARIOS.items()}

num_seeds = 100
rows = []

for seed in range(num_seeds):
    print(f"SEED: {seed}")
    rng = np.random.default_rng(seed)
    for name, base_tensor in BASE_TENSORS.items():
        noisy = base_tensor + rng.normal(0.0, noise_std, size=base_tensor.shape)

        centers, min_t_by_cluster = analyse_with_detection(
            noisy,
            weights,
            grid_size=grid_size,
            conf_thres=0.005
        )

        if centers:
            clusters_recalled = len(centers)
            for label, (cx, cy, ct) in centers.items():
                rows.append({
                    "scenario": name,
                    "seed": seed,
                    "ratio": ratio,
                    "cluster_label": label,
                    "center_x": cx,
                    "center_y": cy,
                    "center_t": ct,
                    "min_t": min_t_by_cluster.get(label, np.nan),
                    "clusters_recalled": clusters_recalled,
                })
        else:
            rows.append({
                "scenario": name,
                "seed": seed,
                "ratio": ratio,
                "cluster_label": None,
                "center_x": float("nan"),
                "center_y": float("nan"),
                "center_t": float("nan"),
                "min_t": float("nan"),
                "clusters_recalled": 0,
            })

df = pd.DataFrame(rows).sort_values(["scenario", "seed", "cluster_label"])
df.to_csv("delta_yolo_outputs_seeds_ONE_to_FIVE.csv", index=False)
print("Saved delta_yolo_outputs_seeds_ONE_to_FIVE.csv")

def frac(series, predicate):
    if callable(predicate):
        return (series.map(predicate)).mean()
    return (series == predicate).mean()

summary = (
    df.groupby("scenario", as_index=False)
      .agg(
          mean_clusters=("clusters_recalled", "mean"),
          median_clusters=("clusters_recalled", "median"),
          frac_1=("clusters_recalled", lambda s: frac(s, 1)),
          frac_2=("clusters_recalled", lambda s: frac(s, 2)),
          frac_3p=("clusters_recalled", lambda s: frac(s, lambda x: x >= 3)),
      )
      .sort_values("scenario")
)

summary.to_csv("delta_yolo_summary_ONE_to_FIVE.csv", index=False)
print("Saved delta_yolo_summary_ONE_to_FIVE.csv")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
# endregion

