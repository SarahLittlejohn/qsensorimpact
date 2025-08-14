import random
import numpy as np

def generate_parity_series(n, switching_rate):
    """
    Generates parity data

    Parameters:
    n: how many outputs there should be (i.e. how much data)
    switching_rate: the rate at which a 

    Returns:
    series: the series of length n, of a parity series
    """
    series = [0]  # Start with 0 explicitly
    p_switch = 1/switching_rate
    for _ in range(1, n):
        if random.random() < p_switch:
            # Flip the previous with probability p_switch
            next_digit = 1 - series[-1]
        else:
            # Keep the previous
            next_digit = series[-1]
        series.append(next_digit)

    return series

def generate_parity_series_dynamic(switching_rates, num_per_rate):
    """
    Generates dynamic parity data (i.e. with changing switching rates)

    Parameters:
    switching_rates: the series of switching rates
    num_per_rate: the number of pieces of parity data per switching rate

    Returns:
    series: series parity data
    """
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
    """
    Takes in a parity series and adds nosie to it

    Parameters:
    series: the parity series we're adding noise to
    p_noise: the probability of noise

    Returns:
    series: series parity data with noise
    """
    # Introduce noise by flipping with probability p_noise
    noisy_series = []
    for digit in parity_series:
        if random.random() < p_noise:
            # Flip to add noise
            noisy_digit = 1 - digit
        else:
            # Keep the original
            noisy_digit = digit
        noisy_series.append(noisy_digit)
    
    return noisy_series

def generate_e2e_parity_series_with_noise(n, switching_rate, p_noise):
    """
    Takes in a switching rate, generates the parity series and adds nosie to it

    Parameters:
    n: how many outputs there should be (i.e. how much data)
    series: the parity series we're adding noise to
    p_noise: the probability of noise

    Returns:
    series: series parity data with noise
    """
    series = [0]  # Start with 0 explicitly
    p_switch = 1/switching_rate
    for _ in range(1, n):
        if random.random() < p_switch:
            # Flip the previous with probability p_switch
            next_digit = 1 - series[-1]
        else:
            # Keep the previous
            next_digit = series[-1]
        
        series.append(next_digit)

    # Introduce noise by flipping with probability p_noise
    noisy_series = []
    for digit in series:
        if random.random() < p_noise:
            # Flip to add noise
            noisy_digit = 1 - digit
        else:
            # Keep the original
            noisy_digit = digit
        noisy_series.append(noisy_digit)
    
    return noisy_series