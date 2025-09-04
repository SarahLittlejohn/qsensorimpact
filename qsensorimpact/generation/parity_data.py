import random
import numpy as np

def generate_parity_series(n, switching_rate):
    series = [0]
    p_switch = 1/switching_rate
    for _ in range(1, n):
        if random.random() < p_switch:
            next_digit = 1 - series[-1]
        else:
            next_digit = series[-1]
        series.append(next_digit)

    return series

def generate_parity_series_dynamic(switching_rates, num_per_rate):
    series = [0]
    for rate in switching_rates:
        p_switch = 1 / rate
        for _ in range(num_per_rate):
            if np.isnan(p_switch):
                next_digit = np.nan
            elif np.isnan(series[-1]):
                next_digit = 0
            elif random.random() < p_switch:
                next_digit = 1 - series[-1]
            else:
                next_digit = series[-1]
            series.append(next_digit)

    return series

def generate_parity_series_with_noise(parity_series, p_noise):
    noisy_series = []
    for digit in parity_series:
        if random.random() < p_noise:
            noisy_digit = 1 - digit
        else:
            noisy_digit = digit
        noisy_series.append(noisy_digit)
    
    return noisy_series

def generate_e2e_parity_series_with_noise(n, switching_rate, p_noise):
    series = [0]
    p_switch = 1/switching_rate
    for _ in range(1, n):
        if random.random() < p_switch:
            next_digit = 1 - series[-1]
        else:
            next_digit = series[-1]
        
        series.append(next_digit)

    noisy_series = []
    for digit in series:
        if random.random() < p_noise:
            noisy_digit = 1 - digit
        else:
            noisy_digit = digit
        noisy_series.append(noisy_digit)
    
    return noisy_series