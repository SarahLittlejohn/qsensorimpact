# region Import and Set-up
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.optimize import curve_fit, fsolve
from scipy.ndimage import map_coordinates
import cv2
import torch
import shutil
from sklearn.cluster import DBSCAN
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qsensorimpact.yolo.yolov5.utils.general import non_max_suppression, scale_boxes
from qsensorimpact.yolo.yolov5.utils.torch_utils import select_device
from qsensorimpact.yolo.yolov5.utils.plots import Annotator
from qsensorimpact.yolo.yolov5.models.common import DetectMultiBackend
from qsensorimpact.yolo.yolov5.utils.dataloaders import LoadImages
# endregion

# region 1. Snapshot-based 2D Analysis
def analyse_two_d_impact_snapshot(matrix_switching_rates, grid_size, baseline):
    """
    Analyze and visualize a 2D switching rate matrix to extract the spatial coordinates of an impact.

    This function fits each row and column of a 2D matrix (typically representing qubit switching rates)
    to a reverse Gaussian curve in order to estimate the impact's location (`d_impact`) in the grid.
    The extracted impact point is the average center position of the fitted Gaussian dips in both
    horizontal and vertical directions.

    Visualizations include:
        - A heatmap of the input matrix (switching rates).
        - Line plots of original data and Gaussian fits for each row and each column.
        - A schematic qubit layout showing the estimated impact location.

    Parameters:
        matrix_switching_rates (np.ndarray): 2D array (grid_size x grid_size) of switching rate values.
        grid_size (int): The number of qubits along one side of the square grid.
        baseline (float): The expected baseline switching rate, used as the initial guess for curve fitting.

    Returns:
        None. (Displays plots and prints the extracted spatial coordinates of the impact.)
    
    Notes:
        - The reverse Gaussian function used is: -a * exp(-((x - b)^2) / (2 * c^2)) + d
        - The extracted impact position is interpreted as the point where the switching rate was most suppressed.
        - Assumes only one major impact exists in the grid.
    """ 
    def reverse_bell_curve(x, a, b, c, d):
        return -a * np.exp(-((x - b)**2) / (2 * c**2)) + d

    def fit_row_or_col(x, y):
        popt, _ = curve_fit(reverse_bell_curve, x, y, p0=[1, np.mean(x), 3, baseline])
        return popt, reverse_bell_curve(x, *popt)

    fitted_rows, fitted_cols = [], []
    all_fits_rows, all_fits_cols = [], []

    x = np.arange(grid_size)
    for i, row in enumerate(matrix_switching_rates):
        popt, fit = fit_row_or_col(x, row)
        fitted_rows.append(popt)
        all_fits_rows.append(fit)

    grid_data_T = matrix_switching_rates.T
    for i, col in enumerate(grid_data_T):
        popt, fit = fit_row_or_col(x, col)
        fitted_cols.append(popt)
        all_fits_cols.append(fit)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    c = axs[0, 0].imshow(matrix_switching_rates, cmap='viridis', origin='lower', extent=[0, grid_size, 0, grid_size])
    axs[0, 0].set_title("2D Qubit Matrix with Gaussian Impact")
    axs[0, 0].set_xlabel("X Position")
    axs[0, 0].set_ylabel("Y Position")
    plt.colorbar(c, ax=axs[0, 0])

    for i, (row, fit) in enumerate(zip(matrix_switching_rates, all_fits_rows)):
        axs[0, 1].plot(x, row, label=f"Row {i}", alpha=0.5)
        axs[0, 1].plot(x, fit, '--', linewidth=1.5)
    axs[0, 1].set_title("Fits for Rows")
    axs[0, 1].set_xlabel("X Position")
    axs[0, 1].set_ylabel("Switching Rate")

    for i, (col, fit) in enumerate(zip(grid_data_T, all_fits_cols)):
        axs[1, 0].plot(x, col, label=f"Col {i}", alpha=0.5)
        axs[1, 0].plot(x, fit, '--', linewidth=1.5)
    axs[1, 0].set_title("Fits for Columns")
    axs[1, 0].set_xlabel("Y Position")
    axs[1, 0].set_ylabel("Switching Rate")

    axs[1, 1].set_aspect('equal', adjustable='box')
    axs[1, 1].set_xlim(0, grid_size)
    axs[1, 1].set_ylim(0, grid_size)
    axs[1, 1].set_title("Qubit Layout with d_impact")
    axs[1, 1].set_xlabel("X Position")
    axs[1, 1].set_ylabel("Y Position")

    for i in range(grid_size):
        for j in range(grid_size):
            circle = plt.Circle((j + 0.5, i + 0.5), 0.3, color='C0', fill=False)
            axs[1, 1].add_patch(circle)

    d_impact_extracted_x = np.mean([popt[1] for popt in fitted_rows])
    d_impact_extracted_y = np.mean([popt[1] for popt in fitted_cols])
    axs[1, 1].scatter(d_impact_extracted_x + 0.5, d_impact_extracted_y + 0.5, color='red', marker='x', s=100, linewidths=2, label=f"d_impact = ({d_impact_extracted_x:.2f}, {d_impact_extracted_y:.2f})")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Extracted d_impact: ({d_impact_extracted_x:.2f}, {d_impact_extracted_y:.2f})")
# endregion

# region 2. Time-dependent 2D Tensor Animation
def analyse_two_d_impact(tensor, interval=100, cmap='viridis'):
    fig, ax = plt.subplots()
    im = ax.imshow(tensor[0], cmap=cmap, vmin=np.min(tensor), vmax=np.max(tensor))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    title_text = ax.text(0.5, 1.05, f"Snapshot 1/{tensor.shape[0]}", transform=ax.transAxes,
                         ha="center", va="bottom", fontsize=12)

    def update(frame):
        im.set_array(tensor[frame])
        title_text.set_text(f"Snapshot {frame + 1}/{tensor.shape[0]}")
        return [im, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=tensor.shape[0], interval=interval, blit=False
    )

    plt.tight_layout()
    plt.show()
    return ani
# endregion

# region 3. YOLO generation and inference
def run_yolo_on_tensor(tensor, weights_path, grid_size, conf_thres=0.25, temp_dir="frames_detect", output_video="detected_impacts.mp4"):
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    height = width = grid_size

    for i, frame in enumerate(tensor):
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='viridis', vmin=np.min(tensor), vmax=np.max(tensor))
        ax.axis('off')
        filename = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    device = select_device('')
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    model.warmup(imgsz=(1, 3, 640, 640))
    dataset = LoadImages(temp_dir, img_size=640, stride=stride, auto=pt)

    result_frames = []
    detections = []

    for path, img, im0s, _, _ in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=conf_thres)

        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=2, example=str(names))

        frame_id = int(os.path.basename(path).split("_")[1].split(".")[0])
        image_height, image_width = im0.shape[:2]
        scale_x = grid_size / image_width
        scale_y = grid_size / image_height

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(c.item()) for c in xyxy]
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    x_center_grid = x_center * scale_x
                    y_center_grid = y_center * scale_y

                    detections.append((frame_id, x_center_grid, y_center_grid))
                    # label = f"{names[int(cls)]} {conf:.2f}"
                    # annotator.box_label(xyxy, label=label)

        result_frames.append(annotator.result())

    if result_frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 10, (result_frames[0].shape[1], result_frames[0].shape[0]))
        for frame in result_frames:
            out.write(frame)
        out.release()
        print(f"Annotated video saved to {output_video}")
    else:
        print("No frames annotated — check confidence threshold?")

    return detections
# endregion

# region 4. YOLO Detection Post-Processing
def extract_detections_3d(temp_dir, image_width, image_height, grid_size):
    scale_x = grid_size / image_width
    scale_y = grid_size / image_height

    data = []
    for fname in sorted(os.listdir(temp_dir)):
        if fname.endswith(".txt") and fname.startswith("frame_"):
            frame_id = int(fname.split("_")[1].split(".")[0])
            with open(os.path.join(temp_dir, fname), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls, x_center, y_center = int(parts[0]), float(parts[1]), float(parts[2])
                    # Convert to grid scale
                    x_center = x_center * image_width * scale_x
                    y_center = y_center * image_height * scale_y
                    data.append((frame_id, x_center, y_center))
    return data

def cluster_and_find_centers(detections, eps=3, min_samples=10, temporal_weight=0.1):
    raw_data = np.array([[x, y, t * temporal_weight] for t, x, y in detections])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(raw_data)
    labels = clustering.labels_

    clusters = {}
    for label, (x, y, t) in zip(labels, raw_data):
        if label == -1:
            continue
        clusters.setdefault(label, []).append((x, y, t))

    centers = {}
    for label, points in clusters.items():
        xs, ys, ts = zip(*points)
        centers[label] = (np.mean(xs), np.mean(ys), np.mean(ts) / temporal_weight)

    return labels, centers, np.array([[x, y, t / temporal_weight] for x, y, t in raw_data])

def predict_midpoint(detections, save_path="predicted_midpoint_plot.png"):
    """
    Given a list of detections (frame, x, y), compute and plot the midpoint.

    Parameters:
        detections (list of tuples): Each tuple is (frame, x, y)
        save_path (str): File path to save the 3D plot

    Returns:
        tuple: (mean_x, mean_y, mean_frame)
    """
    if not detections:
        print("No detections provided.")
        return None

    frames, xs, ys = zip(*detections)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    mean_frame = sum(frames) / len(frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, frames, c='blue', alpha=0.6, label='Detections')
    ax.scatter([mean_x], [mean_y], [mean_frame], c='red', s=100, marker='X', label='Predicted Midpoint')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Frame")
    ax.set_title("Detections and Predicted Midpoint")
    ax.legend()

    # Save plot
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved midpoint plot to {save_path}")
    return mean_x, mean_y, mean_frame

# endregion

# region 5. YOLO Detection and Clustering Plots
def plot_detections_3d(detections, output_path="detection_plot_3d.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    frames = [d[0] for d in detections]
    x_vals = [d[1] for d in detections]
    y_vals = [d[2] for d in detections]

    scatter = ax.scatter(x_vals, y_vals, frames, c=frames, cmap='viridis')
    ax.set_xlabel('X Position (normalized)')
    ax.set_ylabel('Y Position (normalized)')
    ax.set_zlabel('Frame Number (Time)')
    ax.set_title('Detected Object Positions Over Time')

    plt.colorbar(scatter, label='Frame Number')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"3D detection plot saved to {output_path}")

def plot_clusters_3d(raw_data, labels, centers, output_path="clustered_3d.png", grid_size=25):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    raw_data = np.array(raw_data)
    labels = np.array(labels)

    for label in set(labels):
        if label == -1:
            continue
        cluster_points = raw_data[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {label}')

        cx, cy, ct = centers[label]
        ax.scatter(cx, cy, ct, c='black', marker='X', s=100, label=f'Center {label}')

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, max(raw_data[:, 2]) + 10)

    ax.set_xlabel("X Position (grid)")
    ax.set_ylabel("Y Position (grid)")
    ax.set_zlabel("Time (Frame #)")
    ax.set_title("Clustered Impacts with Centers (Grid Units)")
    ax.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Cluster plot saved to {output_path}")
# endregion

# region 6. FULL YOLO Detection Pipeline
def analyse_with_detection(tensor, weights_path, grid_size, conf_thres=0.25):
    detections = run_yolo_on_tensor(tensor, weights_path, grid_size, conf_thres=conf_thres)

    # ---- robust "is empty" check for lists/tuples/ndarrays/None ----
    def _is_empty(d):
        if d is None:
            return True
        # numpy arrays
        if hasattr(d, "size"):
            return d.size == 0
        # sequences
        try:
            return len(d) == 0
        except Exception:
            return True
    # ----------------------------------------------------------------

    if _is_empty(detections):
        print("No detections; skipping midpoint/clustering.")
        return {}, {}   # centers, min_t_by_cluster

    # Midpoint (guarded)
    mp = predict_midpoint(detections)
    if mp is not None:
        mid_x, mid_y, mid_t = mp
        print(mid_x, mid_y, mid_t)
    else:
        print("predict_midpoint returned None (continuing).")

    # Optional sweep (guarded against empties by the early return above)
    for e in [1, 2, 3, 4, 5]:
        labels, centers_tmp, raw = cluster_and_find_centers(detections, eps=e, min_samples=10)
        print(f"eps={e}, clusters={len(set(labels) - {-1})}")

    # Main clustering
    labels, centers, raw_data = cluster_and_find_centers(detections)
    if not centers:
        print("No clusters found.")
        return {}, {}

    plot_clusters_3d(raw_data, labels, centers, grid_size=grid_size)

    for label, (x, y, t_center) in centers.items():
        print(f"  Cluster {label}: X = {x:.2f}, Y = {y:.2f}, Time = {t_center:.2f}")

    T, Y, X = tensor.shape

    print("\nMin values at each cluster center:")
    min_t_by_cluster = {}
    for label, (x, y, _) in centers.items():
        coords = np.vstack([
            np.arange(T),
            np.full(T, y, dtype=float),
            np.full(T, x, dtype=float),
        ])
        interpolated_vals = map_coordinates(tensor, coords, order=1, mode='nearest')
        min_idx = int(np.argmin(interpolated_vals))
        min_val = float(interpolated_vals[min_idx])
        min_t_by_cluster[label] = min_idx
        print(f"  Cluster {label}: Min = {min_val:.4f} at time index t = {min_idx}")

    return centers, min_t_by_cluster

# endregion

# region 7. Curve Fitting Models
def reverse_bell_curve(x, a, b, c, d):
    return -a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d

def exp_recovery(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c

def fit_models_and_select_best(x_data, y_data):
    try:
        p0_gauss = [1, np.argmin(y_data), 10, np.mean(y_data)]
        popt_gauss, _ = curve_fit(reverse_bell_curve, x_data, y_data, p0=p0_gauss)
        fit_gauss = reverse_bell_curve(x_data, *popt_gauss)
    except RuntimeError:
        fit_gauss, popt_gauss = None, None

    try:
        t0 = np.argmin(y_data)
        x_exp = x_data[t0:] - x_data[t0]
        y_exp = y_data[t0:]
        p0_exp = [np.ptp(y_exp), 0.2, y_exp[0]]
        bounds = ([0, 0.01, -np.inf], [np.inf, 5.0, np.inf])
        popt_exp, _ = curve_fit(exp_recovery, x_exp, y_exp, p0=p0_exp, bounds=bounds)
        fit_exp = exp_recovery(x_exp, *popt_exp)
    except RuntimeError:
        fit_exp, popt_exp = None, None

    if fit_gauss is not None:
        residuals_gauss = np.sum((fit_gauss[t0:] - y_data[t0:])**2)
    else:
        residuals_gauss = np.inf

    if fit_exp is not None:
        residuals_exp = np.sum((fit_exp - y_data[t0:])**2)
    else:
        residuals_exp = np.inf

    if residuals_exp < residuals_gauss:
        t_min = t0 + np.argmin(fit_exp)
        full_curve = np.concatenate([np.full(t0, fit_exp[0]), fit_exp])
        return full_curve, t_min, 'exponential', t0
    else:
        t_min = popt_gauss[1] if popt_gauss is not None else None
        return fit_gauss, t_min, 'gaussian', t0
# endregion

# region 8. Quartile Analysis
def plot_quartile_schematic(avg_values, quartile_indices, save_path):
    grid_size = avg_values.shape[0]
    quartile_map = np.zeros((grid_size * grid_size,), dtype=int)

    for q, indices in enumerate(quartile_indices):
        quartile_map[indices] = q + 1

    quartile_map = quartile_map.reshape((grid_size, grid_size))

    cmap = mcolors.ListedColormap(['lightblue', 'lightgreen', 'orange', 'red'])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(quartile_map, cmap=cmap, norm=norm)
    plt.colorbar(im, ticks=[1, 2, 3, 4], label='Quartile assigment')
    plt.title("Qubit Quartile Assignment")
    plt.xticks(np.arange(grid_size))
    plt.xlabel("Horizontal qubit co-ordinate")
    plt.yticks(np.arange(grid_size))
    plt.ylabel("Vertical qubit co-ordinate")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyse_quartiled_qubits(tensor, flag_1, flag_2, save_dir="quartile_analysis_plots"):
    os.makedirs(save_dir, exist_ok=True)

    sliced_tensor = tensor[flag_1:flag_2]
    avg_values = np.mean(sliced_tensor, axis=0)
    flat_avg = avg_values.flatten()
    sorted_indices = np.argsort(flat_avg)
    quartile_indices = np.array_split(sorted_indices, 4)

    # Plot schematic
    plot_quartile_schematic(avg_values, quartile_indices, os.path.join(save_dir, "quartile_schematic.png"))

    # Prepare data
    time_series = tensor.reshape(tensor.shape[0], -1)
    quartile_means = []
    for indices in quartile_indices:
        group_vals = time_series[:, indices]
        mean_vals = np.mean(group_vals, axis=1)
        quartile_means.append(mean_vals)

    x_data = np.arange(time_series.shape[0])
    y_data = quartile_means[0]

    # Fit both models and choose best
    fitted_curve, t_min, model_label, t0 = fit_models_and_select_best(x_data, y_data)

    # --- Plot 1: Quartile averages ---
    plt.figure(figsize=(10, 6))
    for i, mean_vals in enumerate(quartile_means):
        plt.plot(mean_vals, label=f'Quartile {i+1}')
    if fitted_curve is not None:
        if model_label == 'exponential':
            # Only plot from t0 onward
            x_fit = x_data[t0:]
            y_fit = fitted_curve[t0:]
        else:
            x_fit = x_data
            y_fit = fitted_curve

        plt.plot(x_fit, y_fit, 'k--', label=f'Best Fit ({model_label})')        
    plt.axvline(t_min, color='k', linestyle=':', label = rf'Minimum at $\tau = {t_min:.1f}$')
    plt.axvspan(flag_1, flag_2, color='gray', alpha=0.3, label='Analysis window')
    plt.xlabel(r'$\tau$', fontsize=16, labelpad=6)
    plt.ylabel(r'$\Gamma{(quartile mean)}$', fontsize=16, labelpad=6)
    plt.title("Quartile Means Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quartile_means.png"))
    plt.close()

    # --- Plot 2: All qubits ---
    plt.figure(figsize=(10, 6))
    for i in range(time_series.shape[1]):
        plt.plot(time_series[:, i], alpha=0.5)
    plt.axvspan(flag_1, flag_2, color='gray', alpha=0.3)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\Gamma$')
    plt.title("Individual Qubit Values Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_qubits_over_time.png"))
    plt.close()

    return f"Plots saved in: {save_dir}. Best model: {model_label}, minimum at t = {t_min:.2f}" if t_min is not None else "Plots saved, but model fitting failed."
# endregion