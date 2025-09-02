import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
import cv2
import torch
import shutil
from sklearn.cluster import DBSCAN
import numpy as np
from qsensorimpact.yolo.yolov5.utils.general import non_max_suppression, scale_boxes
from qsensorimpact.yolo.yolov5.utils.torch_utils import select_device
from qsensorimpact.yolo.yolov5.utils.plots import Annotator
from qsensorimpact.yolo.yolov5.models.common import DetectMultiBackend
from qsensorimpact.yolo.yolov5.utils.dataloaders import LoadImages
# endregion

# region 1. Snapshot-based 2D Analysis

def analyse_two_d_impact_snapshot(matrix_switching_rates, grid_size, baseline, true_impact=(7.50, 7.50)):

    def reverse_bell_curve(x, a, b, c, d):
        return -a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d

    def fit_vector(vec):
        x = np.arange(grid_size, dtype=float)
        a0 = max(1e-6, float(np.max(vec) - np.min(vec)))
        b0 = float(np.argmin(vec))
        c0 = max(1.0, grid_size / 5.0)
        d0 = float(np.median(vec)) if np.isfinite(np.median(vec)) else baseline
        p0 = [a0, b0, c0, d0]
        try:
            popt, _ = curve_fit(reverse_bell_curve, x, vec, p0=p0, maxfev=10000)
        except Exception:
            popt = np.array(p0, dtype=float)
        return popt

    x_idx   = np.arange(grid_size, dtype=float)
    x_dense = np.linspace(0, grid_size - 1, 400)

    row_params = [fit_vector(row) for row in matrix_switching_rates]
    col_params = [fit_vector(col) for col in matrix_switching_rates.T]

    x_pred = float(np.mean([p[1] for p in row_params])) 
    y_pred = float(np.mean([p[1] for p in col_params])) 

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    hm = axs[0, 0].imshow(
        matrix_switching_rates, cmap='viridis', origin='lower',
        extent=[0, grid_size, 0, grid_size]
    )
    axs[0, 0].set_title("Gaussian impact")
    axs[0, 0].set_xlabel("x coordinate")
    axs[0, 0].set_ylabel("y coordinate")
    plt.colorbar(hm, ax=axs[0, 0])

    for i, (row, p) in enumerate(zip(matrix_switching_rates, row_params)):
        axs[0, 1].plot(x_idx, row, marker='o', linestyle='-', alpha=0.45)
        axs[0, 1].plot(x_dense, reverse_bell_curve(x_dense, *p), '--', linewidth=1.3, label=f"y{i}")
    axs[0, 1].axvline(x_pred, color="red", linestyle="--", linewidth=1.6,
                      label=fr"$x_{{\mathrm{{detected}}}} = {x_pred:.2f}$")
    axs[0, 1].set_title("Fits for rows")
    axs[0, 1].set_xlabel("x co-ordinate")
    axs[0, 1].set_ylabel(r"$T_{\mathrm{sw}}$")
    axs[0, 1].legend(ncol=2, fontsize=8)

    grid_T = matrix_switching_rates.T
    for i, (col, p) in enumerate(zip(grid_T, col_params)):
        axs[1, 0].plot(x_idx, col, marker='o', linestyle='-', alpha=0.45)
        axs[1, 0].plot(x_dense, reverse_bell_curve(x_dense, *p), '--', linewidth=1.3, label=f"x{i}")
    axs[1, 0].axvline(y_pred, color="red", linestyle="--", linewidth=1.6,
                      label=fr"$y_{{\mathrm{{detected}}}} = {y_pred:.2f}$")
    axs[1, 0].set_title("Fits for columns")
    axs[1, 0].set_xlabel("y co-ordinate")
    axs[1, 0].set_ylabel(r"$T_{\mathrm{sw}}$")
    axs[1, 0].legend(ncol=2, fontsize=8)

    axs[1, 1].set_aspect('equal', adjustable='box')
    axs[1, 1].set_xlim(0, grid_size)
    axs[1, 1].set_ylim(0, grid_size)
    axs[1, 1].set_title("Qubit layout with detected and true impact")
    axs[1, 1].set_xlabel("x co-ordinate")
    axs[1, 1].set_ylabel("y co-ordinate")

    for i in range(grid_size):
        for j in range(grid_size):
            axs[1, 1].add_patch(plt.Circle((j + 0.5, i + 0.5), 0.3, color='C0', fill=False))

    axs[1, 1].scatter(
        x_pred + 0.5, y_pred + 0.5,
        color='black', marker='x', s=120, linewidths=2,
        label=fr"Detected: ({x_pred:.2f}, {y_pred:.2f})"
    )

    x_true, y_true = true_impact
    axs[1, 1].scatter(
        x_true + 0.5, y_true + 0.5,
        color='yellow', edgecolors='black', marker='o', s=90,
        label=fr"Impact: ({x_true:.2f}, {y_true:.2f})"
    )

    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Extracted d_impact (detected): ({x_pred:.2f}, {y_pred:.2f})")
    return x_pred, y_pred

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
def run_yolo_on_tensor(
    tensor,
    weights_path,
    grid_size,
    conf_thres=0.25,
    temp_dir="frames_detect",
    output_dir="annotated_frames",
    output_video="detected_impacts.mp4",
    draw_centers=True,
    save_annotated_frames=True,
):
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    vmin, vmax = float(np.min(tensor)), float(np.max(tensor))
    for i, frame in enumerate(tensor):
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        plt.savefig(os.path.join(temp_dir, f"frame_{i:04d}.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    device = select_device('')
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    model.warmup(imgsz=(1, 3, 640, 640))
    dataset = LoadImages(temp_dir, img_size=640, stride=stride, auto=pt)

    detections = []
    result_frames = []

    for path, img, im0, _, _ in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=conf_thres)

        annotator = Annotator(im0.copy(), line_width=2, example=str(names))
        frame_id = int(os.path.basename(path).split("_")[1].split(".")[0])

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(c.item()) for c in xyxy]
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label([x1, y1, x2, y2], label=label)

                    x_center_px = (x1 + x2) // 2
                    y_center_px = (y1 + y2) // 2

                    if draw_centers:
                        cv2.circle(annotator.im, (x_center_px, y_center_px), 6, (0, 0, 255), -1)

                    h, w = im0.shape[:2]
                    x_center_grid = (x_center_px / w) * grid_size
                    y_center_grid = (y_center_px / h) * grid_size

                    detections.append((frame_id, float(x_center_grid), float(y_center_grid)))

        annotated = annotator.result()
        result_frames.append(annotated)

        if save_annotated_frames:
            cv2.imwrite(os.path.join(output_dir, f"annot_{frame_id:04d}.png"), annotated)

    if result_frames:
        h, w = result_frames[0].shape[:2]
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for frame in result_frames:
            out.write(frame)
        out.release()
        print(f"Annotated video: {output_video}")
        print(f"Annotated frames (browse visually): {output_dir}/")
    else:
        print("No frames annotated — try lowering conf_thres or verify weights.")

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
    if not detections:
        print("No detections provided.")
        return None

    frames, xs, ys = zip(*detections)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    mean_frame = sum(frames) / len(frames)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.margins(0.1)

    ax.scatter(xs, ys, frames, c='blue', alpha=0.6, label='Detections')
    ax.scatter([mean_x], [mean_y], [mean_frame], c='red', s=100, marker='X', label='Predicted Midpoint')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Frame")
    ax.set_title("Detections and Predicted Midpoint")
    ax.legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"Saved midpoint plot to {save_path}")
    return mean_x, mean_y, mean_frame

# endregion

# region 5. YOLO Detection and Clustering Plots
def plot_detections_3d(detections, output_path="detection_plot_3d.png"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.margins(0.2)

    frames = [d[0] for d in detections]
    x_vals = [d[1] for d in detections]
    y_vals = [d[2] for d in detections]

    scatter = ax.scatter(x_vals, y_vals, frames, c=frames, cmap='viridis')
    ax.set_xlabel('x co-ordinate')
    ax.set_ylabel('y co-ordinate')
    ax.set_zlabel('Time step')
    ax.set_title('Detected Object Positions Over Time')

    plt.colorbar(scatter, label='Frame Number')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"3D detection plot saved to {output_path}")

def plot_clusters_3d(raw_data, labels, centers, output_path="clustered_3d.png", grid_size=25):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.margins(0.2)

    raw_data = np.array(raw_data)
    labels = np.array(labels)

    for label in set(labels):
        if label == -1:
            continue
        cluster_points = raw_data[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Detections')

        cx, cy, ct = centers[label]
        ax.scatter(cx, cy, ct, c='black', marker='X', s=100,
                label=f'Center: ({cx:.2f}, {cy:.2f}, {ct:.2f})')

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, max(raw_data[:, 2]) + 10)

    ax.set_xlabel('x co-ordinate')
    ax.set_ylabel('y co-ordinate')
    ax.set_zlabel('time step')
    ax.set_title("Clustered Detections with Centers")
    ax.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    print(f"Cluster plot saved to {output_path}")
# endregion

# region 6. FULL YOLO Detection Pipeline
def analyse_with_detection(tensor, weights_path, grid_size, conf_thres=0.25):
    detections = run_yolo_on_tensor(tensor, weights_path, grid_size, conf_thres=conf_thres)

    def _is_empty(d):
        if d is None:
            return True
        if hasattr(d, "size"):
            return d.size == 0
        try:
            return len(d) == 0
        except Exception:
            return True

    if _is_empty(detections):
        print("No detections; skipping midpoint/clustering.")
        return {}, {} 

    mp = predict_midpoint(detections)
    if mp is not None:
        mid_x, mid_y, mid_t = mp
        print(mid_x, mid_y, mid_t)
    else:
        print("predict_midpoint returned None (continuing).")

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

Q_COLORS = ['blue', 'green', 'orange', 'red']

def plot_quartile_schematic(avg_values, quartile_indices, save_path, q_colors=Q_COLORS):
    grid_size = avg_values.shape[0]
    quartile_map = np.zeros((grid_size * grid_size,), dtype=int)
    for q, idx in enumerate(quartile_indices):
        quartile_map[idx] = q + 1
    quartile_map = quartile_map.reshape((grid_size, grid_size))

    cmap  = mcolors.ListedColormap(q_colors)
    norm  = mcolors.BoundaryNorm([0.5,1.5,2.5,3.5,4.5], cmap.N)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(quartile_map, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=[1,2,3,4])
    cbar.set_label('Quartile assignment')
    plt.title("Qubit Quartile Assignment")
    plt.xticks(np.arange(grid_size)); plt.yticks(np.arange(grid_size))
    plt.xlabel("Horizontal qubit co-ordinate"); plt.ylabel("Vertical qubit co-ordinate")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def analyse_quartiled_qubits(tensor, flag_1, flag_2, save_dir="quartile_analysis_plots"):
    os.makedirs(save_dir, exist_ok=True)

    sliced = tensor[flag_1:flag_2]
    avg_values = np.mean(sliced, axis=0)
    flat_avg = avg_values.flatten()
    sorted_idx = np.argsort(flat_avg)
    quartile_indices = np.array_split(sorted_idx, 4)

    plot_quartile_schematic(avg_values, quartile_indices,
                            os.path.join(save_dir, "quartile_schematic.png"),
                            q_colors=Q_COLORS)

    time_series = tensor.reshape(tensor.shape[0], -1)
    quartile_means = [np.mean(time_series[:, idx], axis=1) for idx in quartile_indices]

    x_data = np.arange(time_series.shape[0])
    y_data = quartile_means[0]
    fitted_curve, t_min, model_label, t0 = fit_models_and_select_best(x_data, y_data)

    plt.figure(figsize=(10, 6))
    for i, mean_vals in enumerate(quartile_means):
        plt.plot(mean_vals, label=f'Quartile {i+1}', color=Q_COLORS[i])
    if fitted_curve is not None:
        x_fit = x_data[t0:] if model_label == 'exponential' else x_data
        y_fit = fitted_curve[t0:] if model_label == 'exponential' else fitted_curve
        plt.plot(x_fit, y_fit, 'k--', label=f'Best Fit ({model_label})')

    plt.axvline(t_min, color='k', linestyle=':', label=rf'Minimum at $t = {t_min:.1f}$')
    plt.axvspan(flag_1, flag_2, color='gray', alpha=0.3, label='Analysis window')
    plt.xlabel(r'$t$', fontsize=18, labelpad=8)
    plt.ylabel(r'$T_{\mathrm{quartile\ mean}}$', fontsize=18, labelpad=8)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Quartile Means Over Time", fontsize=18)
    plt.legend(fontsize=12)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quartile_means.png")); plt.close()

    return (f"Plots saved in: {save_dir}. Best model: {model_label}, "
            f"minimum at t = {t_min:.2f}" if t_min is not None else
            "Plots saved, but model fitting failed.")


# endregion

# region 9. GPU Yolo
def run_yolo_on_tensor_GPU(
    tensor,
    weights_path,
    grid_size,
    conf_thres=0.25,
    temp_dir="frames_detect",
    output_dir="annotated_frames",
    output_video="detected_impacts.mp4",
    draw_centers=True,
    save_annotated_frames=True,
    device="cuda" 
):
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    vmin, vmax = float(np.min(tensor)), float(np.max(tensor))
    for i, frame in enumerate(tensor):
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        plt.savefig(os.path.join(temp_dir, f"frame_{i:04d}.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    device = select_device(device)
    print(f"DEVICE: {device}")
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    model.warmup(imgsz=(1, 3, 640, 640))
    dataset = LoadImages(temp_dir, img_size=640, stride=stride, auto=pt)

    detections = []
    result_frames = []

    for path, img, im0, _, _ in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)
        pred = non_max_suppression(pred, conf_thres=conf_thres)

        annotator = Annotator(im0.copy(), line_width=2, example=str(names))
        frame_id = int(os.path.basename(path).split("_")[1].split(".")[0])

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(c.item()) for c in xyxy]
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label([x1, y1, x2, y2], label=label)

                    x_center_px = (x1 + x2) // 2
                    y_center_px = (y1 + y2) // 2
                    if draw_centers:
                        cv2.circle(annotator.im, (x_center_px, y_center_px), 6, (0, 0, 255), -1)

                    h, w = im0.shape[:2]
                    x_center_grid = (x_center_px / w) * grid_size
                    y_center_grid = (y_center_px / h) * grid_size
                    detections.append((frame_id, float(x_center_grid), float(y_center_grid)))

        annotated = annotator.result()
        result_frames.append(annotated)
        if save_annotated_frames:
            cv2.imwrite(os.path.join(output_dir, f"annot_{frame_id:04d}.png"), annotated)

    if result_frames:
        h, w = result_frames[0].shape[:2]
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for frame in result_frames:
            out.write(frame)
        out.release()
        print(f"Annotated video: {output_video}")
    else:
        print("No frames annotated — try lowering conf_thres or verify weights.")

    return detections

# endregion