import os
import numpy as np
import random
from PIL import Image

from qsensorimpact.generation.tools import generate_2d_time_dependent_gaussian_matrix_single_impact


def save_tensor_as_yolo_dataset(tensor, impact_centers, output_root="data", box_size=4):
    img_h, img_w = tensor.shape[1:]  # Assumes shape (frames, H, W)
    img_dir = os.path.join(output_root, "images", "train")
    label_dir = os.path.join(output_root, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Normalize the entire tensor for consistent image brightness
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    for i, frame in enumerate(tensor):
        # Save image using PIL for pixel-perfect output
        norm_frame = (255 * (frame - tensor_min) / (tensor_max - tensor_min)).astype(np.uint8)
        img_path = os.path.join(img_dir, f"frame_{i:04d}.png")
        Image.fromarray(norm_frame).save(img_path)

        # Save YOLO label
        label_path = os.path.join(label_dir, f"frame_{i:04d}.txt")
        with open(label_path, "w") as f:
            for (x, y) in impact_centers.get(i, []):
                x_center = x / img_w
                y_center = y / img_h
                w = h = box_size / img_w  # Assuming square box
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"Saved {tensor.shape[0]} frames and labels to {img_dir}")


if __name__ == "__main__":
    snapshots = 50
    grid_size = 100
    baseline = 15
    initial_amplitude = 3
    num_impacts = 25
    impact_spacing = 55
    total_snapshots = impact_spacing * (num_impacts - 1) + snapshots  # Ensures space for all impacts

    total_tensor = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)
    impact_centers = {}

    # Generate impacts
    for i in range(num_impacts):
        start_frame = i * impact_spacing
        x = random.randint(10, grid_size - 10)  # Avoid edges
        y = random.randint(10, grid_size - 10)

        impact_tensor = generate_2d_time_dependent_gaussian_matrix_single_impact(
            baseline, initial_amplitude, x, y, snapshots, grid_size
        )

        end_frame = start_frame + snapshots
        if end_frame > total_snapshots:
            break  # Prevent overflow

        total_tensor[start_frame:end_frame] -= (baseline - impact_tensor)

        # ✅ Label all 50 frames for this impact
        for j in range(start_frame, end_frame):
            impact_centers.setdefault(j, []).append((x, y))

    save_tensor_as_yolo_dataset(total_tensor, impact_centers)