import numpy as np
import math

# region 1: 1D Gaussian Impact Generators
# Generating the perfect gaussian data
def generate_gaussian_matrix(baseline, initial_amplitude, distances, length_impact=200, noise_std=0.3, baseline_noise_std=0.3):
    """
    Generate a matrix where each row is a Gaussian dip series, with noise added to the Gaussian region
    and to the baseline row.

    Parameters:
    - baseline (float): Value of the constant baseline.
    - initial_amplitude (float): Minimum value the first Gaussian dip reaches.
    - distances (array): Array of distances controlling the spread of dips.
    - length_impact (int): Length of each Gaussian dip (default: 200).
    - noise_std (float): Standard deviation of the noise to add to the Gaussian region.
    - baseline_noise_std (float): Standard deviation of the noise to add to the baseline.

    Returns:
    - matrix (np.ndarray): Matrix where each row is a Gaussian dip or noisy baseline series.
    """
    # Compute the dip times based on the distances
    dip_times = np.array(distances) * 50  # Times for dips: t = 50 * distance
    
    # Total length of the series based on the last Gaussian dip position
    full_length = int(max(dip_times)) + length_impact + 50  # Add some buffer
    
    # Create the baseline series with noise
    baseline_series = np.random.normal(loc=baseline, scale=baseline_noise_std, size=full_length)
    
    # Initialize a list to hold all the rows
    rows = [baseline_series]  # Start with the noisy baseline as the first row
    
    # Generate Gaussian dip template
    x = np.linspace(-3, 3, length_impact)  # x-values for the Gaussian curve
    
    # Generate each Gaussian dip series
    for distance in distances:
        # Calculate the minimum value for this Gaussian dip
        min_value = baseline - (baseline - initial_amplitude) * math.exp(-distance / 8)
        
        # Scale the Gaussian dip so it dips to the correct minimum value
        gaussian_dip = baseline - (baseline - min_value) * np.exp(-0.5 * x**2)
        
        # Add Gaussian noise to the valid Gaussian region
        noise = np.random.normal(loc=0, scale=noise_std, size=length_impact)
        gaussian_dip_noisy = gaussian_dip + noise
        
        # Create a series with NaNs everywhere
        dip_series = np.full(full_length, np.nan, dtype=float)
        
        # Add the Gaussian dip with noise at the correct time
        start_index = int(distance * 50)
        end_index = start_index + length_impact
        if end_index <= full_length:
            dip_series[start_index:end_index] = gaussian_dip_noisy
        
        # Add this dip series as a new row
        rows.append(dip_series)
    
    # Convert the list of rows into a matrix
    matrix = np.vstack(rows)
    return matrix

# Gaussian data of impact on 1D array
def generate_gaussian_matrix_variable_impact(baseline, initial_amplitude, distances, length_impact=200, noise_std=0.3, baseline_noise_std=0.3, impact=0):
    """
    Generate a matrix where each row is a Gaussian dip series, with noise added to the Gaussian region
    and to the baseline row.

    Parameters:
    - baseline (float): Value of the constant baseline.
    - initial_amplitude (float): Minimum value the first Gaussian dip reaches.
    - distances (array): Array of distances controlling the spread of dips.
    - length_impact (int): Length of each Gaussian dip (default: 200).
    - noise_std (float): Standard deviation of the noise to add to the Gaussian region.
    - baseline_noise_std (float): Standard deviation of the noise to add to the baseline.

    Returns:
    - matrix (np.ndarray): Matrix where each row is a Gaussian dip or noisy baseline series.
    """
    # Compute the dip times based on the distances
    dip_times = np.array(distances) * 50  # Times for dips: t = 50 * distance
    
    # Total length of the series based on the last Gaussian dip position
    full_length = int(max(dip_times)) + length_impact + 50  # Add some buffer
    
    # Create the baseline series with noise
    baseline_series = np.random.normal(loc=baseline, scale=baseline_noise_std, size=int(full_length))
    
    # Initialize a list to hold all the rows
    rows = [baseline_series]  # Start with the noisy baseline as the first row
    
    # Generate Gaussian dip template
    x = np.linspace(-3, 3, int(length_impact))  # x-values for the Gaussian curve
    
    # Generate each Gaussian dip series
    for distance in distances:
        # Calculate the minimum value for this Gaussian dip
        min_value = baseline - (baseline - initial_amplitude) * math.exp(-abs(distance - impact) / 8)
        
        # Scale the Gaussian dip so it dips to the correct minimum value
        gaussian_dip = baseline - (baseline - min_value) * np.exp(-0.5 * x**2)
        
        # Add Gaussian noise to the valid Gaussian region
        noise = np.random.normal(loc=0, scale=noise_std, size=length_impact)
        gaussian_dip_noisy = gaussian_dip + noise
        
        # Create a series with NaNs everywhere
        dip_series = np.full(full_length, np.nan, dtype=float)
        
        # Add the Gaussian dip with noise at the correct time
        start_index = int(distance * 50)
        end_index = start_index + length_impact
        if end_index <= full_length:
            dip_series[start_index:end_index] = gaussian_dip_noisy
        
        # Add this dip series as a new row
        rows.append(dip_series)
    
    # Convert the list of rows into a matrix
    matrix = np.vstack(rows)
    return matrix

# endregion

# region 2. 2D Time-Independent Gaussian Impacts
# Gaussian data of time independent impact on 2D array
def generate_2d_gaussian_matrix_single_impact_time_independent(baseline, initial_amplitude, impact_x, impact_y, grid_size=12, 
    length_impact=200, noise_std=0.3, baseline_noise_std=0.3
):
    """
    Generate a single 2D matrix simulating a static Gaussian impact on a grid.

    This function creates a grid representing a physical surface or field with a 
    localized Gaussian dip (impact) at a specific position. The impact is time-independent 
    (i.e., the same in every use), and noise is added to both the baseline and the impact 
    to simulate measurement variation or physical disturbance.

    Parameters:
        baseline (float): The default value of the grid in the absence of impact.
        initial_amplitude (float): The depth of the Gaussian impact.
        impact_x (int): X-coordinate of the impact center.
        impact_y (int): Y-coordinate of the impact center.
        grid_size (int, optional): Size of the square grid (default is 12).
        length_impact (float, optional): Controls the spread of the Gaussian impact (default is 200).
        noise_std (float, optional): Standard deviation of the noise added to the Gaussian (default is 0.3).
        baseline_noise_std (float, optional): Standard deviation of the baseline noise (default is 0.3).

    Returns:
        np.ndarray: A 2D array of shape (grid_size, grid_size) representing the impacted grid.
    """
    matrix = np.random.normal(loc=baseline, scale=baseline_noise_std, size=(grid_size, grid_size))
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

    gaussian = baseline - (baseline - initial_amplitude) * np.exp(
        -((x - impact_x)**2 + (y - impact_y)**2) / (2 * length_impact)
    )

    noise = np.random.normal(loc=0, scale=noise_std, size=(grid_size, grid_size))
    gaussian_noisy = gaussian + noise

    matrix = np.minimum(matrix, gaussian_noisy)

    return matrix

# endregion

# region 3. 2D Time-Dependent Gaussian Impacts
def generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline,
    initial_amplitude,
    impact_x,
    impact_y,
    snapshots,
    grid_size=12,
    spatial_spread=20,
    noise_std=0,
    baseline_noise_std=0
):
    """
    Parameters:
        baseline: The default baseline value for the grid.
        initial_amplitude: The maximum amplitude of the impact (how deep the Gaussian dip is).
        impact_x: X-coordinate of the impact center.
        impact_y: Y-coordinate of the impact center.
        snapshots: Number of time frames (matrices) to generate.
        grid_size: Size of the square grid (default is 12).
        length_impact: Controls the spread of the Gaussian impact (default is 20).
        noise_std: Standard deviation of noise added to the Gaussian (default is 0).
        baseline_noise_std: Standard deviation of noise added to the baseline (default is 0).

    Returns:
        np.ndarray: A 3D array of shape (snapshots, grid_size, grid_size), containing the sequence of impact matrices.
    """
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    spatial_gaussian = np.exp(
        -((x - impact_x) ** 2 + (y - impact_y) ** 2) / (2 * spatial_spread)
    )

    all_matrices = []

    for t in range(snapshots):
        # Temporal amplitude (Gaussian in time)
        impact_amplitude = initial_amplitude * np.exp(
            -0.5 * ((t - (snapshots - 1)/2) / ((snapshots - 1)/4))**2
        )

        # Baseline (with baseline flicker if baseline_noise_std > 0)
        baseline_frame = baseline + np.random.normal(
            0, baseline_noise_std, size=(grid_size, grid_size)
        )

        # Apply impact to baseline
        frame = baseline_frame - impact_amplitude * spatial_gaussian

        # Add final measurement noise
        frame_noisy = frame + np.random.normal(
            0, noise_std, size=(grid_size, grid_size)
        )

        all_matrices.append(frame_noisy)

    return np.stack(all_matrices)
# endregion

# region 4. 2D Delta time dependent impacts
def generate_2d_time_dependent_delta_impact(
    baseline,
    initial_drop,
    impact_x,
    impact_y,
    snapshots,
    grid_size=12,
    spatial_spread=2.0,
    time_decay=0.05,
    noise_std=0,
    baseline_noise_std=0
):
    """
    Parameters:
        baseline (float): Baseline value of the grid.
        initial_drop (float): Maximum depth of the impact at t=0.
        impact_x (float): X-coordinate of the impact center.
        impact_y (float): Y-coordinate of the impact center.
        snapshots (int): Number of time steps to simulate.
        grid_size (int, optional): Size of the square grid.
        spatial_spread (float, optional): Spatial spread (σ) of the impact.
        time_decay (float, optional): Controls the exponential recovery rate.
        noise_std (float, optional): Std dev of impact-related noise.
        baseline_noise_std (float, optional): Std dev of baseline noise.

    Returns:
        np.ndarray: Tensor of shape (snapshots, grid_size, grid_size)
    """
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    spatial_gaussian = np.exp(-((x - impact_x)**2 + (y - impact_y)**2) / (2 * spatial_spread**2))

    all_matrices = []

    for t in range(snapshots):
        # Temporal amplitude (Delta in time)
        impact_amplitude = initial_drop * np.exp(-t * time_decay)

        # Baseline (with baseline flicker if baseline_noise_std > 0)
        baseline_frame = baseline + np.random.normal(
            0, baseline_noise_std, size=(grid_size, grid_size)
        )

        # Apply impact to baseline
        frame = baseline_frame - impact_amplitude * spatial_gaussian


        # Gaussian spatial impact shape 
        spatial_decay = np.exp(-((x - impact_x)**2 + (y - impact_y)**2) / (2 * spatial_spread**2))
        delta_impact = -impact_amplitude * spatial_decay

        # Add final measurement noise
        frame_noisy = frame + np.random.normal(
            0, noise_std, size=(grid_size, grid_size)
        )

        all_matrices.append(frame_noisy)

    return np.stack(all_matrices)
# endregion

# region 5. Generate 2D time dependent impacts of variable function
def generate_2d_switching_rate_tensor_from_gamma(
    Gamma_in,
    grid_size,
    impact_x=None,
    impact_y=None,
    spatial_spread=2.0,
    downsample_factor=1  # Set to e.g. 10 to take every 10th timestep
):
    """
    Generate a 3D tensor of switching rates over time and space, using Gamma_in(t) at the impact center
    and applying a Gaussian spatial decay to surrounding qubits.

    Parameters:
        Gamma_in (np.ndarray): 1D array of tunneling rates (Hz) over time.
        grid_size (int): Size of the 2D grid (grid_size x grid_size).
        impact_x (int): X-coordinate of impact center (defaults to center).
        impact_y (int): Y-coordinate of impact center (defaults to center).
        spatial_spread (float): Standard deviation of the spatial Gaussian decay.
        downsample_factor (int): How much to downsample the time series. Default = 1 (no downsampling).

    Returns:
        np.ndarray: 3D tensor of shape (new_time, grid_size, grid_size) representing switching rates.
    """
    import numpy as np

    if impact_x is None:
        impact_x = grid_size // 2
    if impact_y is None:
        impact_y = grid_size // 2

    # Downsample Gamma_in
    Gamma_in = Gamma_in[::downsample_factor]
    time_steps = len(Gamma_in)

    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

    # Precompute the spatial decay (same for all time steps)
    spatial_mask = np.exp(-((x - impact_x) ** 2 + (y - impact_y) ** 2) / (2 * spatial_spread ** 2))

    # Broadcast multiplication over time
    tensor = Gamma_in[:, np.newaxis, np.newaxis] * spatial_mask[np.newaxis, :, :]


    return tensor
#endregion

# region 6. 2D Real time dependent impacts
def generate_2d_time_dependent_real_impact(
    baseline,
    impact_x,
    impact_y,
    snapshots,
    grid_size=12,
    spatial_spread=2.0,
    noise_std=0,         
    baseline_noise_std=0 
):
    """
    Real-impact generator using gamma-like temporal profile:
      - per-frame noisy baseline
      - impact field = baseline + Gamma(t) * spatial_Gaussian
      - final frame = min(baseline_frame, impact_frame_noisy) so dips win
    """
    E_dep = 0.4
    eta_ph = 0.3
    Delta = 190e-6
    V_um3 = 100
    tau_qp = 1e-3
    tau_abs = 2e-3
    K = 5

    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    spatial_gaussian = np.exp(
            -((x - impact_x) ** 2 + (y - impact_y) ** 2) / (2 * spatial_spread)
        )

    # Scaling t values
    t_total = 0.01
    t_values = np.linspace(0.0, t_total, snapshots, endpoint=False)
    # Temporal amplitude (real function in time)
    N_r_qp = eta_ph * E_dep / Delta
    delta_nqp_t = (N_r_qp / V_um3) * (tau_qp / (tau_abs - tau_qp)) * (
            np.exp(-t_values / tau_abs) - np.exp(-t_values / tau_qp)
        )
    amplitude_t = -K * delta_nqp_t 

    all_matrices = []

    for t in range(snapshots):
            # Baseline with baseline flicker
            baseline_frame = baseline + np.random.normal(
                0, baseline_noise_std, size=(grid_size, grid_size)
            )

            # Apply impact to baseline
            frame = baseline_frame + amplitude_t[t] * spatial_gaussian

            # Apply final noise
            frame_noisy = frame + np.random.normal(
                0, noise_std, size=(grid_size, grid_size)
            )
            all_matrices.append(frame_noisy)

    return np.stack(all_matrices)

# endregion
