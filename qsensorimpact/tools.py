import numpy as np
import math

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

def generate_2d_gaussian_matrix_single_impact_time_independent(baseline, initial_amplitude, impact_x, impact_y, grid_size=12, 
    length_impact=200, noise_std=0.3, baseline_noise_std=0.3
):
    matrix = np.random.normal(loc=baseline, scale=baseline_noise_std, size=(grid_size, grid_size))
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

    gaussian = baseline - (baseline - initial_amplitude) * np.exp(
        -((x - impact_x)**2 + (y - impact_y)**2) / (2 * length_impact)
    )

    noise = np.random.normal(loc=0, scale=noise_std, size=(grid_size, grid_size))
    gaussian_noisy = gaussian + noise

    matrix = np.minimum(matrix, gaussian_noisy)

    return matrix