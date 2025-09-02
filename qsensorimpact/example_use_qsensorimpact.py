from qsensorimpact.generation.parity_data import generate_parity_series, generate_parity_series_dynamic, generate_parity_series_with_noise, generate_e2e_parity_series_with_noise
from qsensorimpact.generation.simulation_data import find_static_switching_rate_clean_series, find_static_switching_rate_noisy_series, find_dynamic_switching_rates_noisy_series, estimate_switching_rate_with_resampling, estimate_switching_rate_with_resampling_hmm
from qsensorimpact.generation.tools import generate_gaussian_matrix, generate_gaussian_matrix_variable_impact, generate_2d_gaussian_matrix_single_impact_time_independent, generate_2d_time_dependent_gaussian_matrix_single_impact, generate_2d_time_dependent_delta_impact, generate_2d_switching_rate_tensor_from_gamma, generate_2d_time_dependent_real_impact
from qsensorimpact.analysis.one_d_impact import analyse_one_d_impact
from qsensorimpact.analysis.two_d_impact import analyse_two_d_impact_snapshot, analyse_two_d_impact, analyse_with_detection, analyse_quartiled_qubits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from pathlib import Path
weights = Path(__file__).parent / "qsensorimpact/qsensorimpact/yolo/yolov5/runs/train/impact-model/weights/best.pt"
weights = str(weights.resolve())


# region 1. Using the generating parity series functionality
# 1.1. Generating a parity series 
series_test1 = generate_parity_series(30000, 3)

# 1.2. Generating a dynamic parity series
series_test2 = generate_parity_series_dynamic([1, 10, 1], 100)

# 1.3. Generating a parity series with noise from a parity series
series_test3 = generate_parity_series_with_noise(series_test2, 0.2)

# 1.4. Generating a partiy series with noise from scratch
series_test4 = generate_e2e_parity_series_with_noise(100, 3, 0.1)
# endregion

# region 2. Using the simulation data functionality
# 2.1. Finding static switching rate from clean series
series_test1 = generate_parity_series(30000, 3)
find_static_switching_rate_clean_series(series_test1)

# 2.2. Finding static switching rate from noisy series
static_series = generate_parity_series(30000, 3)
noisy_series = generate_parity_series_with_noise(static_series, 0.2)
switching_rate, switching_error = find_static_switching_rate_noisy_series(noisy_series)
print(switching_rate)
print(switching_error)

# 2.3. Finding dynamic switching rates from noisy series
dynamic_series = generate_parity_series_dynamic([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 200)
noisy_series = generate_parity_series_with_noise(dynamic_series, 0.01)
switching_rates_dirty, switching_rate_errors_dirty = find_dynamic_switching_rates_noisy_series(dynamic_series, 200)
switching_rates_clean, switching_rate_errors_clean = find_static_switching_rate_clean_series(dynamic_series)

# endregion

# region 3. Tools: generating impact signatures
# 3.1 Generate gaussian matrix representing impact at 0
baseline = 7
initial_amplitude = 3 
distances = np.arange(0, 10.5, 0.5)
matrix = generate_gaussian_matrix(baseline, initial_amplitude, distances)

for i, row in enumerate(matrix):
    plt.plot(row, label=f'Row {i}')

plt.title('Line Graph of Matrix Rows')
plt.xlabel('Column Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 3.2 Generate gaussian matrix representing impact at some distance
baseline = 7
initial_amplitude = 3 
distances = np.arange(0, 10.5, 0.5)
noise_std = 0.3
matrix = generate_gaussian_matrix_variable_impact(baseline, initial_amplitude, distances, impact=2)

for i, row in enumerate(matrix):
    plt.plot(row, label=f'Row {i}')

plt.title('Line Graph of Matrix Rows')
plt.xlabel('Column Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 3.3 Generate 2D gaussian matrix representing impact at some distance (time independent)
baseline = 7
initial_amplitude = 3
grid_size = 40 
length_impact = 200
noise_std = 0.3
baseline_noise_std = 0.3
impact_x, impact_y = 22, 26

matrix = generate_2d_gaussian_matrix_single_impact_time_independent(
    baseline, initial_amplitude, impact_x, impact_y, grid_size, 
    length_impact, noise_std, baseline_noise_std
)

for i, row in enumerate(matrix):
    plt.plot(row, label=f'Row {i}')

plt.title('Line Graph of Matrix Rows')
plt.xlabel('Column Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# endregion

# region 4. Generate 1D impacts
# 4.1 Perfect data direct impact
d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
baseline = 7
initial_amplitude = 3
length_impact = 200
d_impact = 10
simulation_segment = 1000
perfect_switching_rates = generate_gaussian_matrix_variable_impact(baseline, initial_amplitude, d, length_impact, impact=d_impact)

# 4.2 Simulated data indirect impact
d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
baseline = 7
initial_amplitude = 3
length_impact = 200
d_impact = 11.5
simulation_segment = 500
perfect_switching_rates = generate_gaussian_matrix_variable_impact(baseline, initial_amplitude, d, length_impact, impact=d_impact)
simulated_switching_rates_rows = []
simulated_errors_rows = []
for row in perfect_switching_rates:
    simulated_parity_rates = generate_parity_series_dynamic(row, simulation_segment)
    switching_rates, switching_rate_errors = find_dynamic_switching_rates_noisy_series(simulated_parity_rates, simulation_segment)
    simulated_switching_rates_rows.append(switching_rates)
    simulated_errors_rows.append(switching_rate_errors)
matrix_switching_rates = np.array(simulated_switching_rates_rows)
matrix_switching_rate_errors = np.array(simulated_errors_rows)

# endregion

# region 5. Generate 2D snapshot impacts
# 5.1 Perfect data impact
baseline = 14
initial_amplitude = 6
grid_size = 15 
length_impact = 2
noise_std = 0.3
baseline_noise_std = 0.3
impact_x, impact_y = 7.5, 7.5
perfect_switching_rates = generate_2d_gaussian_matrix_single_impact_time_independent(baseline, initial_amplitude, impact_x, impact_y, grid_size, length_impact, noise_std, baseline_noise_std)

# 5.2 Simulated data impact
baseline = 7
initial_amplitude = 3
grid_size = 40 
length_impact = 200
noise_std = 0.3
baseline_noise_std = 0.3
impact_x, impact_y = 22, 26
simulation_segment = 1000
perfect_switching_rates = generate_2d_gaussian_matrix_single_impact_time_independent(baseline, initial_amplitude, impact_x, impact_y, grid_size, length_impact, noise_std, baseline_noise_std)
simulated_switching_rates_rows = []
for row in perfect_switching_rates:
    simulated_parity_rates = generate_parity_series_dynamic(row, simulation_segment)
    switching_rates, switching_rate_errors = find_dynamic_switching_rates_noisy_series(simulated_parity_rates, simulation_segment)
    simulated_switching_rates_rows.append(switching_rates)
simulated_matrix = np.array(simulated_switching_rates_rows)

# endregion

# region 6. 2D Tensor analysis with time dependence single impact
# 6.1 Perfect data impact
tensor = generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline=7,
    initial_amplitude=3,
    impact_x=7.5,
    impact_y=7.5,
    snapshots=50,
    grid_size=15
)

analyse_two_d_impact(tensor)
# endregion

# region 7. 2D Tensor analysis with time dependence two impacts
# 7.1 Two impacts (distinct and concatenated)
tensor_first_impact = generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline=7,
    initial_amplitude=3,
    impact_x=22,
    impact_y=22,
    snapshots=50,
    grid_size=40
)
tensor_second_impact = generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline=7,
    initial_amplitude=3,
    impact_x=12,
    impact_y=10,
    snapshots=50,
    grid_size=40
)

total_tensor = np.concatenate((tensor_first_impact, tensor_second_impact), axis=0)
analyse_two_d_impact(total_tensor)

# 7.2 Two impacts (simulataneous)
tensor_first_impact = generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline=15,
    initial_amplitude=3,
    impact_x=60,
    impact_y=25,
    snapshots=50,
    grid_size=100
)
tensor_second_impact = generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline=15,
    initial_amplitude=3,
    impact_x=25,
    impact_y=20,
    snapshots=50,
    grid_size=100
)

baseline = 10

deviation_first = baseline - tensor_first_impact
deviation_second = baseline - tensor_second_impact

total_deviation = deviation_first + deviation_second

total_tensor = baseline - total_deviation

analyse_two_d_impact(total_tensor)

# 7.3 Two impacts (partial overlap)
snapshots = 50
grid_size = 100
baseline = 15

tensor_first_impact = generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline=baseline,
    initial_amplitude=3,
    impact_x=60,
    impact_y=25,
    snapshots=snapshots,
    grid_size=grid_size
)
tensor_second_impact = generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline=baseline,
    initial_amplitude=3,
    impact_x=40,
    impact_y=20,
    snapshots=snapshots,
    grid_size=grid_size
)

total_snapshots = 75

total_tensor = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)

deviation_first = baseline - tensor_first_impact
total_tensor[:snapshots] -= deviation_first 

deviation_second = baseline - tensor_second_impact
total_tensor[25:25+snapshots] -= deviation_second

ani = analyse_two_d_impact(total_tensor)
plt.show(block=True)
# endregion
 
# region 8. Multi-analysis functions on multiple tensors at once
# 8.1 Multiple imapcts at different times
def run_analysis_on_multiple_impacts(
    tensors, start_times, baseline, grid_size,
    weights_path, total_snapshots=130, conf_thres=0.005, csv_output_path="all_centers.csv"
):
    import csv
    centers_all = []

    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Impact Index", "Cluster Label", "X", "Y", "Time"])

        for i, (tensor, start_t) in enumerate(zip(tensors, start_times)):
            full_tensor = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)
            full_tensor[start_t:start_t + tensor.shape[0]] -= (baseline - tensor)

            centers = analyse_with_detection(full_tensor, weights_path, grid_size=grid_size, conf_thres=conf_thres)

            for label, (x, y, t) in centers.items():
                writer.writerow([i, label, round(x, 2), round(y, 2), round(t, 2)])
                centers_all.append((i, label, x, y, t))

    print(f"Saved {len(centers_all)} centers to {csv_output_path}")
    return centers_all

# 8.1 Multiple imapcts at different times
def run_analysis_on_multiple_simul_impacts(
    tensors, start_times, baseline, grid_size,
    weights_path, total_snapshots=130, conf_thres=0.005, csv_output_path="all_centers.csv"
):
    import csv
    assert len(tensors) == len(start_times), "Mismatch between tensors and start times."
    assert len(tensors) % 2 == 0, "Number of tensors must be even to pair them."

    centers_all = []

    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pair Index", "Cluster Label", "X", "Y", "Time"])

        for i in range(0, len(tensors), 2):
            tensor_a, tensor_b = tensors[i], tensors[i+1]
            start_a, start_b = start_times[i], start_times[i+1]

            total_tensor = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)

            print(f"Pair {i//2}: start_a={start_a}, tensor_a.shape={tensor_a.shape}")
            print(f"Pair {i//2}: start_b={start_b}, tensor_b.shape={tensor_b.shape}")
            print(f"type(start_b): {type(start_b)}, type(tensor_b.shape[0]): {type(tensor_b.shape[0])}")
            start_b = int(start_b)
            end_b = int(start_b + tensor_b.shape[0])
            total_tensor[start_a:start_a + tensor_a.shape[0]] -= (baseline - tensor_a)
            total_tensor[start_b:end_b] -= (baseline - tensor_b)


            centers = analyse_with_detection(total_tensor, weights_path, grid_size=grid_size, conf_thres=conf_thres)

            for label, (x, y, t) in centers.items():
                writer.writerow([i // 2, label, round(x, 2), round(y, 2), round(t, 2)])
                centers_all.append((i // 2, label, x, y, t))

    print(f"Saved {len(centers_all)} centers to {csv_output_path}")
    return centers_all
# endregion

# region 9. 2D Tensor YOLO detection analysis with time dependence multi-variable impacts
# 9.1 Four impacts analysis with YOLO
snapshots = 50
grid_size = 15
baseline = 15
total_snapshots = 150

from pathlib import Path
weights = Path(__file__).parent / "qsensorimpact/qsensorimpact/yolo/yolov5/runs/train/impact-model/weights/best.pt"
weights = str(weights.resolve())

tensor_1 = generate_2d_time_dependent_gaussian_matrix_single_impact(baseline, 3, 7.5, 7.5, snapshots, grid_size, 3)
tensor_2 = generate_2d_time_dependent_gaussian_matrix_single_impact(baseline, 3, 40, 30, snapshots, grid_size, 3)
tensor_3 = generate_2d_time_dependent_gaussian_matrix_single_impact(baseline, 3, 20, 17, snapshots, grid_size, 3)
tensor_4 = generate_2d_time_dependent_gaussian_matrix_single_impact(baseline, 3, 15, 35, snapshots, grid_size, 3)

# Start times
t_1 = 25
t_2 = 50
t_3 = 75
t_4 = 100

# Prepare total tensor
total_tensor = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)
total_tensor[t_1:t_1 + snapshots] -= (baseline - tensor_1)
total_tensor[t_2:t_2 + snapshots] -= (baseline - tensor_2)
total_tensor[t_3:t_3 + snapshots] -= (baseline - tensor_3)
total_tensor[t_4:t_4 + snapshots] -= (baseline - tensor_4)

# Add noise
noise_std = 0.1
noise = np.random.normal(loc=0.0, scale=noise_std, size=total_tensor.shape)
tensor_with_noise = total_tensor + noise

# Run YOLO detection
analyse_with_detection(tensor_with_noise, weights, grid_size=grid_size, conf_thres=0.005)

# endregion

# region 10. 2D Tensor Quartiling analysis single delta impact
snapshots = 50
grid_size = 6
baseline = 15
total_snapshots = 130
t_1 = 50
tensor_1 = generate_2d_time_dependent_delta_impact(baseline, 3, 3, 3, snapshots, grid_size, 3)
total_tensor = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)
total_tensor[t_1:100] -= (baseline - tensor_1)
noise_std=0.1
noise = np.random.normal(loc=0.0, scale=noise_std, size=total_tensor.shape)
tensor_with_noise =  total_tensor + noise

analyse_quartiled_qubits(tensor_with_noise, 50, 60)

# endregion

# region 11. QP-Burst reconstruction 1 Qubit
# Detector and physics constants (from the paper Table I and text)
E_dep = 0.4              # energy deposit in eV
eta_ph = 0.3             # phonon-to-QP conversion efficiency
Delta = 190e-6           # superconducting gap energy in eV (for Al)
V_um3 = 100              # absorber volume in μm³
tau_qp = 1e-3            # quasiparticle recombination time in seconds
tau_abs = 2e-3           # phonon absorption time in seconds
K = 20e3                 # tunneling rate coefficient in Hz·μm³

N_r_qp = eta_ph * E_dep / Delta  # total number of QPs generated
t_total = 0.01             # total simulation time (10 ms)
dt = 1e-6                        # timestep (1 μs)
time = np.arange(0, t_total, dt)

delta_nqp = (N_r_qp / V_um3) * (tau_qp / (tau_abs - tau_qp)) * (
    np.exp(-time / tau_abs) - np.exp(-time / tau_qp)
)

Gamma_in = K * delta_nqp

parity = [1]
current_parity = 1

for i in range(1, len(time)):
    p_flip = Gamma_in[i] * dt 
    print(1/p_flip)
    if np.random.rand() < p_flip:
        current_parity *= -1
    parity.append(current_parity)

parity = np.array(parity)
parity_binary = (parity + 1) // 2

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax[0].plot(time * 1e3, parity, drawstyle='steps-post')
ax[0].set_ylabel('Parity State')
ax[0].set_title('Simulated Parity Telegraph Signal')
ax[0].grid(True)

ax[1].plot(time * 1e3, Gamma_in, label=r'$\Gamma_{\rm in}(t)$', color='orange')
ax[1].set_ylabel('Tunneling Rate (Hz)')
ax[1].set_xlabel('Time (ms)')
ax[1].set_title('Quasiparticle Tunneling Rate Over Time')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# endregion

# region 12. QP-Burst reconstruction Qubit Array 
E_dep = 0.4              # energy deposit in eV
eta_ph = 0.3             # phonon-to-QP conversion efficiency
Delta = 190e-6           # superconducting gap energy in eV (for Al)
V_um3 = 100              # absorber volume in μm³
tau_qp = 1e-3            # quasiparticle recombination time in seconds
tau_abs = 2e-3           # phonon absorption time in seconds
K = 20e3                 # tunneling rate coefficient in Hz·μm³

N_r_qp = eta_ph * E_dep / Delta  # total number of QPs generated
t_total = 0.01             # total simulation time (10 ms)
dt = 1e-6                        # timestep (1 μs)
time = np.arange(0, t_total, dt)

delta_nqp = (N_r_qp / V_um3) * (tau_qp / (tau_abs - tau_qp)) * (
    np.exp(-time / tau_abs) - np.exp(-time / tau_qp)
)

switching_rate = -K * delta_nqp

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

tensor = generate_2d_switching_rate_tensor_from_gamma(switching_rate, 30, downsample_factor=200)

analyse_with_detection(tensor, weights, grid_size=30, conf_thres=0.005)
# endregion

# region 13. Delta function through YOLO
snapshots = 50
grid_size = 20
baseline = 15
total_snapshots = 130
t_1 = 50
tensor_1 = generate_2d_time_dependent_delta_impact(baseline, 3, 10, 10, snapshots, grid_size, 3)
total_tensor = np.full((total_snapshots, grid_size, grid_size), baseline, dtype=np.float64)
total_tensor[t_1:100] -= (baseline - tensor_1)
noise_std=0.01
noise = np.random.normal(loc=0.0, scale=noise_std, size=total_tensor.shape)
tensor_with_noise =  total_tensor + noise

analyse_with_detection(tensor_with_noise, weights, grid_size, conf_thres=0.005)
# endregion

# region 14. QP-Burst analysis

E_dep = 0.4              # energy deposit in eV
eta_ph = 0.3             # phonon-to-QP conversion efficiency
Delta = 190e-6           # superconducting gap energy in eV (for Al)
V_um3 = 100              # absorber volume in μm³
tau_qp = 1e-3            # quasiparticle recombination time in seconds
tau_abs = 2e-3           # phonon absorption time in seconds
K = 20e3                 # tunneling rate coefficient in Hz·μm³

N_r_qp = eta_ph * E_dep / Delta  # total number of QPs generated
t_total = 0.01             # total simulation time (10 ms)
dt = 1e-6                        # timestep (1 μs)
time = np.arange(0, t_total, dt)

delta_nqp = (N_r_qp / V_um3) * (tau_qp / (tau_abs - tau_qp)) * (
    np.exp(-time / tau_abs) - np.exp(-time / tau_qp)
)

switching_rate = -K * delta_nqp

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

tensor = generate_2d_switching_rate_tensor_from_gamma(switching_rate, 6, downsample_factor=200)

analyse_quartiled_qubits(tensor, 5, 10)

# endregion

# region 15. Guassian Quartiling analysis
tensor = generate_2d_time_dependent_gaussian_matrix_single_impact(
    baseline=16,
    initial_amplitude=12,
    impact_x=2,
    impact_y=3,
    snapshots=50,
    grid_size=6,
    noise_std = 0.2,
    baseline_noise_std = 0.2,
    spatial_spread=3
)

total_snapshots = 130

total_tensor = np.full((total_snapshots, 6, 6), 16, dtype=np.float64)

deviation_first = 16 - tensor
total_tensor[50:100] -= deviation_first
noise_std = 0.7
noise = np.random.normal(loc=0.0, scale=noise_std, size=total_tensor.shape)
tensor_with_noise = total_tensor + noise

analyse_quartiled_qubits(tensor_with_noise, 65, 85)
# endregion

# region 16. Graphs of signatures

# 16.1 Gaussian
# Parameters
baseline = 14
initial_amplitude = 6
length_impact = 200
mu = 100 
sigma = 20 

t = np.arange(length_impact)

gaussian_dip = baseline - (baseline - initial_amplitude) * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

plt.figure(figsize=(8, 4))
plt.plot(t, gaussian_dip, color='navy')
plt.axhline(baseline, linestyle='--', color='gray', label='Baseline')
plt.xlabel(r'$t$', fontsize=16, labelpad=6)
plt.ylabel(r'$T_0 + T(t)_{time}$', fontsize=16, labelpad=6)
plt.title("Gaussian Impact Signature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 16.2 Delta
# Parameters
baseline = 14
E_dep = 0.4              # energy deposit in eV
eta_ph = 0.3             # phonon-to-QP conversion efficiency
Delta = 190e-6           # superconducting gap energy in eV (for Al)
V_um3 = 100              # absorber volume in μm³
tau_qp = 1e-3            # quasiparticle recombination time in seconds
tau_abs = 2e-3           # phonon absorption time in seconds
K = 5                    # tunneling rate coefficient

N_r_qp = eta_ph * E_dep / Delta  
t_total = 10e-3             
num_timesteps = 200
dt = t_total / num_timesteps                 
time = np.arange(0, t_total, dt)

delta_nqp = (N_r_qp / V_um3) * (tau_qp / (tau_abs - tau_qp)) * (
    np.exp(-time / tau_abs) - np.exp(-time / tau_qp)
)

switching_rate = baseline - K * delta_nqp

timesteps = np.arange(len(time))

plt.figure(figsize=(8, 4))
plt.plot(timesteps, switching_rate, color='teal')
plt.axhline(baseline, linestyle='--', color='gray', label='Baseline')

plt.xlabel(r'$t$', fontsize=16, labelpad=6)
plt.ylabel(r'$T_0 + T(t)_{time}$', fontsize=16, labelpad=6)
plt.title("QP-Burst Impact Signature")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# endregion

# region 17. Accuracy of SRs

# Graph 1: accuracy of clean switching rates
input_rates = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
output_rates = []
error_rates = []

for i in input_rates:
    gen_fn = lambda: generate_parity_series(200, i)
    mean, std, cleaned_samples = estimate_switching_rate_with_resampling(gen_fn, num_trials=100)
    output_rates.append(mean)
    error_rates.append(std)


plt.figure(figsize=(8, 5))
plt.errorbar(input_rates, output_rates, yerr=error_rates, fmt='o', capsize=5)
plt.plot(input_rates, input_rates, 'k--', label='Input = Output')
plt.xlabel(r'$T_{\mathrm{input}}$', fontsize=16, labelpad=6)
plt.ylabel(r'$T_{\mathrm{output}}$', fontsize=16, labelpad=6)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 2: accuracy of noisy switching rates
input_rates = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
output_rates = []
error_rates = []

for i in input_rates:
    clean = generate_parity_series(400, i)
    gen_fn = lambda: generate_parity_series_with_noise(clean, 0.08)
    mean, std, cleaned_samples, model_avg = estimate_switching_rate_with_resampling_hmm(gen_fn, num_trials=100)
    combined_error = np.sqrt(std**2 + model_avg**2)
    output_rates.append(mean)
    error_rates.append(combined_error)

    print(f"Input_rate: {i}, Std: {std:.4f}, Model Avg Error: {model_avg:.4f}, Combined Error: {combined_error:.4f}")


plt.figure(figsize=(8, 5))
plt.errorbar(input_rates, output_rates, yerr=error_rates, fmt='o', capsize=5)
plt.plot(input_rates, input_rates, 'k--', label='Input = Output')
plt.xlabel(r'$T_{\mathrm{input}}$', fontsize=16, labelpad=6)
plt.ylabel(r'$T_{\mathrm{output}}$', fontsize=16, labelpad=6)
plt.legend()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 3: varying noise for SR 7
noise = np.arange(0, 0.5, 0.005)
diff = []
n_seeds = 100  

for i in noise:
    errs = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        clean = generate_parity_series(400, 7)
        gen_fn = lambda: generate_parity_series_with_noise(clean, i)
        mean, std, cleaned_samples, model_avg = estimate_switching_rate_with_resampling_hmm(
            gen_fn, num_trials=100
        )
        errs.append(abs(7 - mean))
    diff.append(np.mean(errs))

plt.figure(figsize=(8, 5))
plt.plot(noise, diff, 'o')
plt.plot(noise, diff, 'o-', alpha=0.3) 
plt.xlabel("P(noise)", fontsize=16)
plt.ylabel(r'$\langle | \Delta_{T_{\mathrm{SW}} - 7} | \rangle$', fontsize=16, labelpad=6)
plt.grid(True)
plt.tight_layout()
plt.show()

# endregion