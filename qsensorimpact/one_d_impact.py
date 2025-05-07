import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

def analyse_one_d_impact(matrix_data, d, matrix_errors=None, params=True):
    """
    Analyse and visualise a 1D array of time-dependent switching rate data to extract impact characteristics.

    This function fits each row of switching rate data (over time) to a reverse Gaussian model to identify 
    the minimum switching rate (interpreted as the strongest impact) and the time at which it occurs. It then
    fits exponential decay models to the minimum points across distances to estimate the location and dynamics 
    of the impact in a qubit array or similar system.

    Visual outputs include:
        - Raw switching rate data and fitted reverse Gaussian curves
        - Scatter plot of minimum points with error bars
        - Exponential fits to left and right sides of the impact
        - Schematic plot of the qubit layout with estimated impact location

    Parameters:
        matrix_data (np.ndarray): 2D array of switching rates with shape (time, spatial distance).
                                  The first row is assumed to be a baseline measurement.
        d (list or np.ndarray): List of distances corresponding to each row in matrix_data (excluding the baseline row).
        matrix_errors (np.ndarray, optional): 2D array of the same shape as matrix_data, providing error estimates.
        params (bool, optional): If True, fit and print exponential decay and propagation parameters.

    Returns:
        None. (Produces plots and prints extracted physical parameters such as impact location, decay rate, and velocity.)

    Notes:
        - The minimum switching rate is taken as an indicator of the impact strength at each spatial location.
        - The crossover point between the left and right exponential fits is interpreted as the true impact location.
        - Exponential decay and linear time shift models are fitted to estimate physical characteristics like λ (decay) and σ (propagation speed).
    """
    all_fitted_values = []
    min_switching_rates = []
    min_times = []
    min_switching_rate_errors = []
    min_times_errors = []

    # Create a matrix of simulated switching rates & errors
    def propagate_fit_errors(x, popt, perr):
        a, b, c, d = popt
        sigma_a, sigma_b, sigma_c, sigma_d = perr

        # Compute partial derivatives
        exp_term = np.exp(-((x - b) ** 2) / (2 * c ** 2))
        df_da = -exp_term
        df_db = a * exp_term * ((x - b) / (c ** 2))
        df_dc = a * exp_term * ((x - b) ** 2) / (c ** 3)
        df_dd = 1  # d contributes directly

        # Propagated error formula
        sigma_y = np.sqrt((df_da * sigma_a) ** 2 +
                        (df_db * sigma_b) ** 2 +
                        (df_dc * sigma_c) ** 2 +
                        (df_dd * sigma_d) ** 2)

        return sigma_y

    def reverse_bell_curve(x, a, b, c, d):
        """Reverse Gaussian (bell curve) model."""
        return -a * np.exp(-((x - b)**2) / (2 * c**2)) + d

    def fit_switching_rate(x, y):
        """Fit the data to the reverse bell curve model."""
        popt, pcov = curve_fit(reverse_bell_curve, x, y, p0=[1, np.mean(x), 50, 7])
        perr = np.sqrt(np.diag(pcov))

        return reverse_bell_curve(x, *popt), popt, perr

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

        fitted_values, popt, perr = fit_switching_rate(x_valid, y_valid)
        all_fitted_values.append((x_valid, fitted_values))
        
        fit_errors = propagate_fit_errors(x_valid, popt, perr)

        min_rate = np.min(fitted_values)
        min_index = np.argmin(fitted_values)
        min_time = x_valid[min_index]
        min_switching_rates.append(min_rate)
        min_times.append(min_time)

        if matrix_errors is not None:
            error_valid = matrix_errors[i][valid_indices]
            total_error = np.sqrt(fit_errors[min_index]**2 + error_valid[min_index]**2)
            min_switching_rate_errors.append(total_error)
            min_times_errors.append(perr[1])
        else:
            min_switching_rate_errors.append(0)
            min_times_errors.append(0)

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
    if matrix_errors is not None:
        axs[1, 0].errorbar(
            min_times,
            min_switching_rates,
            xerr=min_times_errors,
            yerr=min_switching_rate_errors,
            fmt='o',
            ecolor='black',
            elinewidth=1.5,
            capsize=4,
            color='red',
            label='Min Points with Error'
        )
    else:
        axs[1, 0].scatter(
            min_times,
            min_switching_rates,
            color='red',
            s=50,
            label='Min Points'
        )
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
    if matrix_errors is not None:
        print("Fit error at min:", fit_errors[min_index])
        print("Original data error at min:", error_valid[min_index])

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