import numpy as np
from hmmlearn import hmm

def find_static_switching_rate_clean_series(series):
    """
    Finds the switching rate of a clean parity series

    Parameters:
    series: the parrity series
    """

    # Count transitions
    n_01 = np.sum((np.array(series[:-1]) == 0) & (np.array(series[1:]) == 1))  # Transitions from 0 to 1
    n_00 = np.sum((np.array(series[:-1]) == 0) & (np.array(series[1:]) == 0))  # Transitions from 0 to 0
    n_10 = np.sum((np.array(series[:-1]) == 1) & (np.array(series[1:]) == 0))  # Transitions from 1 to 0
    n_11 = np.sum((np.array(series[:-1]) == 1) & (np.array(series[1:]) == 1))  # Transitions from 1 to 1

    print(n_01, n_00, n_10, n_11)

    total_0 = np.sum(np.array(series) == 0)
    total_1 = np.sum(np.array(series) == 1)

    P = np.array([[n_00 / total_0, n_01 / total_0],
                [n_10 / total_1, n_11 / total_1]])

    switch_0_to_1 = P[0, 1]
    switch_1_to_0 = P[1, 0]

    print("Transition Matrix:")
    print(P)
    print(f"Switching Rate 0 -> 1: {switch_0_to_1}")
    print(f"Switching Rate 1 -> 0: {switch_1_to_0}")

    p_0 = total_0/(total_0 + total_1)
    p_1 = 1 - p_0
    N = len(np.array(series))
    sigma_0 = np.sqrt(p_0 * (1 - p_0) / N)
    sigma_P = np.zeros_like(P)
    if total_0 > 0:
        sigma_P[0, 0] = np.sqrt(P[0, 0] * (1 - P[0, 0]) / total_0)
        sigma_P[0, 1] = np.sqrt(P[0, 1] * (1 - P[0, 1]) / total_0)
    if total_1 > 0:
        sigma_P[1, 0] = np.sqrt(P[1, 0] * (1 - P[1, 0]) / total_1)
        sigma_P[1, 1] = np.sqrt(P[1, 1] * (1 - P[1, 1]) / total_1)

    total_switching_probability = p_0 * switch_0_to_1 + p_1 * switch_1_to_0
    total_switching_probability_std = np.sqrt(
        (switch_0_to_1 - switch_1_to_0) ** 2 * sigma_0 ** 2 +
        (p_0 * sigma_P[0, 1]) ** 2 +
        (p_1 * sigma_P[1, 0]) ** 2
    )

    total_switching_rate = 1 / total_switching_probability
    total_switching_rate_std = total_switching_probability_std / (total_switching_probability ** 2)

    print(f"Total switching probability: {total_switching_probability}")
    print(f"Total switching probability error: {total_switching_probability_std}")
    print(f"Total switching rate: {total_switching_rate}")
    print(f"Total switching rate error: {total_switching_rate_std}")

    return total_switching_rate, total_switching_rate_std

def find_static_switching_rate_noisy_series(noisy_series):
    """
    Finds the switching rate of a noisy series using HMM

    Parameters:
    noisy_series: the noisy parity series

    Returns:
    total_switching_rate: the computed switching rate
    """

    # Reshape the series for hmmlearn
    noisy_series = np.array(noisy_series).reshape(-1, 1)

    # Use CategoricalHMM
    model = hmm.CategoricalHMM(n_components=2, init_params='tm', algorithm="MAP", n_iter=20000, random_state=123)
    model.startprob_ = np.array([1.0, 0.0])

    model.fit(noisy_series)

    # Extract transition matrix and switching rates
    transmat = model.transmat_

    # Calculate switching rates directly from the transition matrix
    switching_freq_0_to_1 = transmat[0, 1]
    switching_freq_1_to_0 = transmat[1, 0]

    # Compute stationary distribution
    total_switching = switching_freq_0_to_1 + switching_freq_1_to_0
    pi_0 = switching_freq_1_to_0 / total_switching
    pi_1 = switching_freq_0_to_1 / total_switching

    # Calculate total switching rate
    total_switching_rate = 1/(pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0)
    
    # Error propagation for switching rate
    # Partial derivatives for error propagation
    dR_dP01 = -pi_0 / (pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0) ** 2
    dR_dP10 = -pi_1 / (pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0) ** 2
    dR_dPi0 = -switching_freq_0_to_1 / (pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0) ** 2
    dR_dPi1 = -switching_freq_1_to_0 / (pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0) ** 2

    # Assuming errors in transition probabilities (you may need to estimate these):
    sigma_P01 = np.sqrt(switching_freq_0_to_1 * (1 - switching_freq_0_to_1) / len(noisy_series))
    sigma_P10 = np.sqrt(switching_freq_1_to_0 * (1 - switching_freq_1_to_0) / len(noisy_series))
    sigma_Pi0 = 0  # Pi0 is derived, its error is dependent on P01 and P10
    sigma_Pi1 = 0  # Similarly, Pi1 is derived from P01 and P10

    # Total error using propagation formula
    switching_rate_error = np.sqrt(
        (dR_dP01 * sigma_P01) ** 2 +
        (dR_dP10 * sigma_P10) ** 2 +
        (dR_dPi0 * sigma_Pi0) ** 2 +
        (dR_dPi1 * sigma_Pi1) ** 2
    )

    return total_switching_rate, switching_rate_error
    
def find_dynamic_switching_rates_noisy_series(parity_series, segment_length):
    """
     Finds the switching rate of a noisy series by segmenting it.

    Parameters:
    series: the parrity series
    segment_length: how long each segment is
    """

    # Segment the parity series into chunks of segment_length
    num_segments = len(parity_series) // segment_length
    switching_rates = []
    switching_rate_errors = []

    for i in range(num_segments):
        # Extract segment
        segment = parity_series[i * segment_length:(i + 1) * segment_length]
        if np.isnan(segment).any():
            rate, error = np.nan, np.nan
        else:
            # Compute switching rate using the provided model function
            rate, error = find_static_switching_rate_noisy_series(segment)
        switching_rates.append(rate)
        switching_rate_errors.append(error)

    return switching_rates, switching_rate_errors

def estimate_switching_rate_with_resampling(generator_function, num_trials=50, iqr_filter=True):
    """
    Estimates switching rate by resampling and removing outliers.

    Parameters:
    generator_function: a function that returns a new parity series each time it's called
    num_trials: how many parity series to generate
    iqr_filter: whether to apply IQR filtering to remove outliers

    Returns:
    mean_rate, std_rate, filtered_rates
    """

    rates = []
    
    for _ in range(num_trials):
        series = generator_function()
        try:
            rate, _ = find_static_switching_rate_clean_series(series)
            rates.append(rate)
        except Exception as e:
            continue  # skip if bad input or edge case
    
    rates = np.array(rates)

    if iqr_filter:
        q1 = np.percentile(rates, 25)
        q3 = np.percentile(rates, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rates = rates[(rates >= lower) & (rates <= upper)]

    return np.mean(rates), np.std(rates), rates.tolist()


def estimate_switching_rate_with_resampling_hmm(generator_function, num_trials=50, iqr_filter=True):
    """
    Estimates switching rate using HMM by resampling and removing outliers.

    Parameters:
    generator_function: a function that returns a new noisy parity series each time it's called
    num_trials: how many parity series to generate
    iqr_filter: whether to apply IQR filtering to remove outliers

    Returns:
    mean_rate: mean switching rate across filtered trials
    std_rate: standard deviation (used as empirical error bar)
    filtered_rates: list of switching rates after filtering
    avg_model_error: mean of the model-predicted error values
    """

    rates = []
    model_errors = []

    for _ in range(num_trials):
        series = generator_function()
        try:
            rate, error = find_static_switching_rate_noisy_series(series)
            rates.append(rate)
            model_errors.append(error)
        except Exception:
            continue

    rates = np.array(rates)
    model_errors = np.array(model_errors)

    if iqr_filter and len(rates) >= 5:
        q1 = np.percentile(rates, 25)
        q3 = np.percentile(rates, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (rates >= lower) & (rates <= upper)
        rates = rates[mask]
        model_errors = model_errors[mask]

    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    avg_model_error = np.mean(model_errors)

    return mean_rate, std_rate, rates.tolist(), avg_model_error
