import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

def generate_one_d_impact(matrix_data, d, params=True):
    all_fitted_values = []
    min_switching_rates = []
    min_times = []

    def reverse_bell_curve(x, a, b, c, d):
        """Reverse Gaussian (bell curve) model."""
        return -a * np.exp(-((x - b)**2) / (2 * c**2)) + d

    def fit_switching_rate(x, y):
        """Fit the data to the reverse bell curve model."""
        popt, _ = curve_fit(reverse_bell_curve, x, y, p0=[1, np.mean(x), 50, 7])
        return reverse_bell_curve(x, *popt), popt

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # --- Top Left: Switching Rates with Fits ---
    baseline_row = matrix_data[0]
    x_baseline = np.arange(len(baseline_row))
    valid_baseline_indices = ~np.isnan(baseline_row)
    x_valid_baseline = x_baseline[valid_baseline_indices]
    y_valid_baseline = baseline_row[valid_baseline_indices]

    axs[0, 0].plot(x_valid_baseline, y_valid_baseline, color='grey', linewidth=2, label="Baseline")

    adjusted_d = d[:-1]
    for i, (row, dist) in enumerate(zip(matrix_data[1:], adjusted_d)):
        valid_indices = ~np.isnan(row)
        x_valid = np.where(valid_indices)[0]
        y_valid = row[valid_indices]
        
        if len(x_valid) == 0:
            continue

        fitted_values, _ = fit_switching_rate(x_valid, y_valid)
        all_fitted_values.append((x_valid, fitted_values))

        min_rate = np.min(fitted_values)
        min_index = np.argmin(fitted_values)
        min_time = x_valid[min_index]
        min_switching_rates.append(min_rate)
        min_times.append(min_time)

        axs[0, 0].plot(x_valid, y_valid, alpha=0.7, label=f"d={dist}")
        axs[0, 0].plot(x_valid, fitted_values, '--', linewidth=1.5, label=f"Fit d={dist}")

    axs[0, 0].set_title("Switching Rates with Fitted Curves")
    axs[0, 0].set_xlabel("Time Steps")
    axs[0, 0].set_ylabel("Switching Rate")

    # --- Top Right: Legend for the First Plot ---
    axs[0, 1].axis("off")  # Turn off the axis
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 1].legend(handles, labels, loc='center', title="Legend", ncol=3, frameon=False, fontsize=8, borderpad=1)

    # --- Bottom Left: Scatter Plot of Minimum Switching Rates with Fits ---
    axs[1, 0].scatter(min_times, min_switching_rates, color='red', s=50)
    axs[1, 0].set_title("Scatter of Minimum Switching Rates with Fits")
    axs[1, 0].set_xlabel("Time (min_times)")
    axs[1, 0].set_ylabel("Min Switching Rate")

    # Add left and right exponential fits
    def exp_left(x, alpha, c):
        return -np.exp(alpha * x) + c

    def exp_right(x, alpha, x0, c):
        return -np.exp(-alpha * (x - x0)) + c

    min_index = np.argmin(min_switching_rates)
    left_times = min_times[:min_index + 1]
    right_times = min_times[min_index:]

    left_rates = min_switching_rates[:min_index + 1]
    right_rates = min_switching_rates[min_index:]

    # Fit exponential curves
    left_params, _ = curve_fit(exp_left, left_times, left_rates, p0=[0.001, 6.0], maxfev=5000)
    right_params, _ = curve_fit(exp_right, right_times, right_rates, p0=[0.001, right_times[0], 4.0], maxfev=5000)

    x_left_smooth = np.linspace(min(left_times), max(left_times), 500)
    y_left_fit = exp_left(x_left_smooth, *left_params)

    x_right_smooth = np.linspace(min(right_times), max(right_times), 500)
    y_right_fit = exp_right(x_right_smooth, *right_params)

    axs[1, 0].plot(x_left_smooth, y_left_fit, color='blue', linewidth=2, label="Left Fit")
    axs[1, 0].plot(x_right_smooth, y_right_fit, color='green', linewidth=2, label="Right Fit")
    axs[1, 0].legend()

    # --- Bottom Right: Qubit Layout ---
    axs[1, 1].set_aspect('equal', adjustable='box')
    for dist in adjusted_d:
        circle = plt.Circle((dist, 0), 0.2, color='C0', fill=True)
        axs[1, 1].add_patch(circle)

    # Highlight the extracted minimum distance
    crossover_time = fsolve(lambda x: exp_left(x, *left_params) - exp_right(x, *right_params), min_times[min_index])[0]
    global_min_d = (crossover_time - 100) / 50
    if global_min_d is not None:
        axs[1, 1].scatter(global_min_d, 0, color='red', marker='x', s=100, linewidths=2, label=f"Extracted d_impact = {global_min_d:.2f}")

    axs[1, 1].set_title('Qubit Layout (1D)')
    axs[1, 1].set_xlabel('Distance (d)')
    axs[1, 1].set_xlim(-1, max(d) + 1)
    axs[1, 1].set_ylim(-1, 1)
    axs[1, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=1)

    # Final layout adjustments
    plt.tight_layout()
    plt.show()

    # Print extracted parameters
    print(f"Crossover Time: {crossover_time:.2f}")
    print(f"Extracted d_impact: {global_min_d:.2f}")

    if params:
        # Fit the exponetial decay
        def exp_decay(distance, lambda_):
            return 7 - 4 * np.exp(-lambda_ * distance)

        def linear_model(distance, sigma):
            return min_times[0] + distance * sigma

        adjusted_d = np.array(adjusted_d)

        params_A, _ = curve_fit(exp_decay, adjusted_d, min_switching_rates)
        lambda_estimate = params_A[0]

        params_t, _ = curve_fit(linear_model, adjusted_d, min_times)
        sigma_estimate = params_t[0]

        print(lambda_estimate)
        print(sigma_estimate)