# Import statements.
import numpy as np
import rescompy.features as features
import rescompy.regressions as regressions
import rc_helpers as rch
import test_systems as tst
import joblib as jlib

run_label = "performance_vs_test_length_noiseless_train"
run_label = "performance_vs_test_length_noisy_train"

method = "async_sm_ri"

if "noisy" in run_label:
    train_noisy = True
elif "noiseless" in run_label:
    train_noisy = False
    
test_noisy = True #Include observational noise in the short signals from new systems
noisy_targets = False #False, so that we still measure performance against noiseless trajectories
fixed_reg = True

save_data = True
safe_save = False
same_RCs = False
reduce_predictions = True
rmse_only = True
use_parallel = True

# Define the library. Also set the minimum length test signals and the number
# of training and testing samples.
lib_size = 9
val_size = 625
lib_seed = 1001
val_seed = lib_seed + lib_size
transient_length = 1000
lib_length = 5000
test_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]
noise_amps = [1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1]

sigma_rng = [7.5, 12.5]
omega_rng = [.75, 1.25]
val_sigma_rng = [7., 13.]
val_omega_rng = [.7, 1.3]

pred_regs = {
    "async_sm" : 1e-6,
    "async_sm_ri" : 1e-6,
    "library_interpolation" : 1e-6,
    "nearest_euclidean" : 1e-6,
    "library_average" : 1e-6,
    "vanilla" : 1e-6,
    "long_vanilla" : 1e-13,
    "batch" : 1e-6
    }
mapper_regs = {
    "async_sm" : 1e-8,
    "async_sm_ri" : 1e-8,
    "library_interpolation" : np.nan,
    "nearest_euclidean" : np.nan,
    "library_average" : np.nan,
    "vanilla" : np.nan,
    "long_vanilla" : np.nan,
    "batch" : np.nan
    }

# Set initial RC hyperparameters
pred_esn_args = {
    "seed" : 1,
    "size" : 500,
    "spectral_radius" : .9,
    "leaking_rate" : .1,
    "input_strength" : .1,
    "bias_strength" : .5,
    "connections" : 3,
    "input_dimension" : 3
    }
if same_RCs:
    map_esn_args = pred_esn_args.copy()
else:
    map_esn_args = {
        "size" : 1000,
        "spectral_radius" : .9,
        "leaking_rate" : .1,
        "input_strength" : .1,
        "bias_strength" : .5,
        "input_dimension" : 3,
        "connections" : 3,
        "seed" : 2
        }
        
# Arguments for signal mapper feature function.
mapper_lookback = 0
mapper_decimation = 30
max_num_states = 3
    
if mapper_lookback > 0:
    async_mapper_feature = features.MixedReservoirStates(
        decimation = mapper_decimation,
        max_num_states = max_num_states
        )
else:
    async_mapper_feature = features.FinalStateOnly()
    
# Set the prediciton RC feature function.
pred_feature = features.StatesAndInputs()
pred_feature = features.StatesOnly()
    
# Establish the regression routine for the prediction RC.
pred_batch_size = 10
pred_accessible_drives = list(np.arange(-5, 0, 1))
    
# Establish the regression routine for the async signal mapper.
async_mapper_batch_size = 100
async_mapper_accessible_drives = list(np.arange(-100, 0, 1))
async_sample_separation = 1
    
# Fetch training and test signals.
lib_seeds = np.arange(lib_seed, lib_seed + lib_size + 1) + 5
test_seeds = np.arange(lib_size + 1, lib_size + val_size + 1) + 5

lib_sigmas = np.random.default_rng(lib_seed).uniform(
    low = sigma_rng[0], high = sigma_rng[1], size = lib_size)
lib_omegas = np.random.default_rng(lib_seed**lib_seed).uniform(
    low = omega_rng[0], high = omega_rng[1], size = lib_size)
test_sigmas = np.random.default_rng(val_seed).uniform(
    low = val_sigma_rng[0], high = val_sigma_rng[1], size = val_size)
test_omegas = np.random.default_rng(val_seed**val_seed).uniform(
    low = val_omega_rng[0], high = val_omega_rng[1], size = val_size)

test_signals = [tst.get_lorenz(
    seed = seed, return_length = 3000, transient_length = transient_length,
    sigma = sigma, omega = omega)
    for seed, sigma, omega in zip(test_seeds, test_sigmas, test_omegas)]

if train_noisy:
    train_noise = noise_amps.copy()
else:
    train_noise = [None] * len(noise_amps)
if test_noisy:
    test_noise = noise_amps.copy()
else:
    test_noise = [None] * len(noise_amps)

def make_and_train_lib_predict(
        method : str,
        test_length : int,
        lib_length : int = 5000,
        async_sample_separation : int = 1,
        fixed_reg : bool = True,
        lib_noise_amp : float = None,
        test_noise_amp : float = None,
        noisy_targets : bool = False,
        pred_reg : float = 1e-6,
        mapper_reg : float = 1e-6
        ):
    
    if fixed_reg:
        num_time_steps = lib_length - 1
        num_short_sigs = (lib_length - test_length) // async_sample_separation
        pred_regularization = pred_reg * num_time_steps
        if not np.isnan(mapper_reg):
            async_mapper_regularization = mapper_reg * num_short_sigs
    else:
        pred_regularization = pred_reg
        if not np.isnan(mapper_reg):
            async_mapper_regularization = mapper_reg
    
    if not np.isnan(mapper_reg):
        async_mapper_regression = regressions.batched_ridge(
            regularization = async_mapper_regularization
            )
        
    pred_regression = regressions.batched_ridge(
        regularization = pred_regularization
        )
    
    train_signals = [tst.get_lorenz(
        seed = seed, return_length = lib_length + transient_length, sigma = sigma,
        omega = omega) for seed, sigma, omega in zip(lib_seeds, lib_sigmas, lib_omegas)]
    
    if lib_noise_amp is not None:
        train_signals = [signal + lib_noise_amp * np.std(signal, axis = 0) * np.random.normal(
            size = signal.shape) for signal in train_signals]
    if test_noise_amp is not None:
        scale_arr = np.copy(train_signals[0])
        for signal in train_signals[1:]:
            scale_arr = np.concatenate((scale_arr, signal), axis = 0)
        scale = np.std(scale_arr, axis = 0)
        if noisy_targets:
            test_sigs = [signal + test_noise_amp * scale * np.random.normal(
                size = signal.shape) for signal in test_signals]
        else:
            test_sigs = [
                np.concatenate((
                    test_noise_amp*scale*np.random.normal(size = signal[:test_length].shape),
                    np.zeros(signal[test_length:].shape)),
                axis = 0) + signal for i, signal in enumerate(test_signals)
                ]
        del(scale_arr)
        del(scale)
    
    def get_async_predictions(seed: int, test_length: int,
                              incl_ri: bool = False, incl_rf: bool = False):
        
        return rch.Async_SM_Train_and_Predict(
            seed = seed,
            run_label = run_label,
            file_name = str(test_length),
            method_label = str(test_noise_amp),
            pred_esn_args = pred_esn_args,
            mapper_esn_args = map_esn_args,
            library_signals = train_signals,
            test_signals = test_sigs,
            test_length = test_length,
            transient_length = transient_length,
            sample_separation = async_sample_separation,
            incl_ri = incl_ri,
            incl_rf = incl_rf,
            same_seed = same_RCs,
            predict_length = None,
            pred_feature = pred_feature,
            mapper_feature = async_mapper_feature,
            pred_regression = pred_regression,
            mapper_regression = async_mapper_regression,
            mapper_accessible_drives = async_mapper_accessible_drives,
            pred_batch_size = pred_batch_size,
            mapper_batch_size = async_mapper_batch_size,
            save = save_data,
            safe_save = safe_save,
            reduce_predictions = reduce_predictions,
            rmse_only = rmse_only
            )
    
    if method == "async_sm":
        get_async_predictions(
            seed = pred_esn_args["seed"],
            test_length = test_length
            )
            
    if method == "async_sm_ri":
        get_async_predictions(
            seed = pred_esn_args["seed"],
            test_length = test_length,
            incl_ri = True
            )

if use_parallel:
    jlib.Parallel(n_jobs = -1)(jlib.delayed(
        make_and_train_lib_predict)(
            method = method,
            test_length = test_length,
            lib_length = lib_length,
            lib_noise_amp = train_noise[noise_ind],
            test_noise_amp = test_noise[noise_ind],
            noisy_targets = noisy_targets,
            fixed_reg = fixed_reg,
            async_sample_separation = async_sample_separation,
            pred_reg = pred_regs[method],
            mapper_reg = mapper_regs[method]
            )
            for test_length in test_lengths
            for noise_ind in range(len(noise_amps))
            )
else:
    for test_length in test_lengths:
        for noise_ind in range(len(noise_amps)):
            make_and_train_lib_predict(
                method = method,
                test_length = test_length,
                lib_length = lib_length,
                lib_noise_amp = train_noise[noise_ind],
                test_noise_amp = test_noise[noise_ind],
                noisy_targets = noisy_targets,
                fixed_reg = fixed_reg,
                async_sample_separation = async_sample_separation,
                pred_reg = pred_regs[method],
                mapper_reg = mapper_regs[method]
                )