import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit, fsolve

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

def analyse_two_d_impact(tensor, interval=100, cmap='viridis'):
    fig, ax = plt.subplots()
    im = ax.imshow(tensor[0], cmap=cmap, vmin=np.min(tensor), vmax=np.max(tensor))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Create title as a Text object
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