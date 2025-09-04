import numpy as np
from hmmlearn import hmm

def find_static_switching_rate_clean_series(series):

    n_01 = np.sum((np.array(series[:-1]) == 0) & (np.array(series[1:]) == 1)) 
    n_00 = np.sum((np.array(series[:-1]) == 0) & (np.array(series[1:]) == 0))  
    n_10 = np.sum((np.array(series[:-1]) == 1) & (np.array(series[1:]) == 0))  
    n_11 = np.sum((np.array(series[:-1]) == 1) & (np.array(series[1:]) == 1)) 

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
    noisy_series = np.array(noisy_series).reshape(-1, 1)

    model = hmm.CategoricalHMM(n_components=2, init_params='tm', algorithm="MAP", n_iter=20000, random_state=123)
    model.startprob_ = np.array([1.0, 0.0])

    model.fit(noisy_series)

    transmat = model.transmat_

    switching_freq_0_to_1 = transmat[0, 1]
    switching_freq_1_to_0 = transmat[1, 0]

    total_switching = switching_freq_0_to_1 + switching_freq_1_to_0
    pi_0 = switching_freq_1_to_0 / total_switching
    pi_1 = switching_freq_0_to_1 / total_switching

    total_switching_rate = 1/(pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0)
    
    dR_dP01 = -pi_0 / (pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0) ** 2
    dR_dP10 = -pi_1 / (pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0) ** 2
    dR_dPi0 = -switching_freq_0_to_1 / (pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0) ** 2
    dR_dPi1 = -switching_freq_1_to_0 / (pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0) ** 2

    sigma_P01 = np.sqrt(switching_freq_0_to_1 * (1 - switching_freq_0_to_1) / len(noisy_series))
    sigma_P10 = np.sqrt(switching_freq_1_to_0 * (1 - switching_freq_1_to_0) / len(noisy_series))
    sigma_Pi0 = 0 
    sigma_Pi1 = 0

    switching_rate_error = np.sqrt(
        (dR_dP01 * sigma_P01) ** 2 +
        (dR_dP10 * sigma_P10) ** 2 +
        (dR_dPi0 * sigma_Pi0) ** 2 +
        (dR_dPi1 * sigma_Pi1) ** 2
    )

    return total_switching_rate, switching_rate_error
    
def find_dynamic_switching_rates_noisy_series(parity_series, segment_length):
    num_segments = len(parity_series) // segment_length
    switching_rates = []
    switching_rate_errors = []

    for i in range(num_segments):
        segment = parity_series[i * segment_length:(i + 1) * segment_length]
        if np.isnan(segment).any():
            rate, error = np.nan, np.nan
        else:
            rate, error = find_static_switching_rate_noisy_series(segment)
        switching_rates.append(rate)
        switching_rate_errors.append(error)

    return switching_rates, switching_rate_errors

def estimate_switching_rate_with_resampling(generator_function, num_trials=50, iqr_filter=True):
    rates = []
    
    for _ in range(num_trials):
        series = generator_function()
        try:
            rate, _ = find_static_switching_rate_clean_series(series)
            rates.append(rate)
        except Exception as e:
            continue 
    
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
