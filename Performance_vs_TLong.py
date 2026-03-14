# Generate data for Fig. S3B

# Import statements.
import numpy as np
import rescompy as rc
import rescompy.features as features
import rescompy.regressions as regressions
import os
import rc_helpers as rch
import test_systems as tst
import joblib as jlib
from typing import Union

run_label = "performance_vs_lib_length"

#method = "async_sm"
method = "async_sm_ri" #metafors
#method = "library_interpolation"
#method = "nearest_euclidean"
#method = "library_average"
#method = "vanilla"
#method = "long_vanilla"
#method = "batch" #multitask

train_noisy = False
test_noisy = False
noisy_targets = False
fixed_reg = True
return_dims = [0, 1, 2]

save_data = True
safe_save = False
test_from_lib = False
same_RCs = False
reduce_predictions = True
rmse_only = True
use_parallel = True
extrapolation_function = rch.get_extrapolation_function(
        duration = 150,
        method = "constant",
        compiled = False
        )
extrapolation_function = None

sync_sm_label = "Sync_SM_Const_Extrap"
sync_sm_label = "Sync_SM_State_Match"

# Define the library. Also set the minimum length test signals and the number
# of training and testing samples.
grid_lib = False
grid_val = True
lib_size = 3**2
val_size = 25**2
lib_seed = 1001
val_seed = lib_seed + lib_size
transient_length = 1000
lib_lengths = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000,
               7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000]
val_length = 3000
test_length = 200
noise_amp = 1e-7

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
    "input_dimension" : len(return_dims)
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
        "input_dimension" : len(return_dims),
        "connections" : 3,
        "seed" : 2
        }
        
# Arguments for signal mapper feature function.
mapper_lookback = 0
mapper_decimation = 30
max_num_states = 3
    
if mapper_lookback > 0:
    mapper_feature = features.StatesOnlyTimeShifted(
        states_lookback_length = mapper_lookback,
        states_decimation = mapper_decimation #dec_time
        )
    async_mapper_feature = features.MixedReservoirStates(
        decimation = mapper_decimation,
        max_num_states = max_num_states
        )
else:
    mapper_feature = features.StatesOnly()
    async_mapper_feature = features.FinalStateOnly()
    
# Set the prediciton RC feature function.
pred_feature = features.StatesAndInputs()
pred_feature = features.StatesOnly()
    
# Establish the regression routine for the prediction RC.
pred_batch_size = 10
pred_accessible_drives = list(np.arange(-5, 0, 1))
       
# Establish the regression routine for the signal mapper.
async_mapper_batch_size = 100
async_mapper_accessible_drives = list(np.arange(-100, 0, 1))
async_sample_separation = 1

if train_noisy:
    train_noise = noise_amp
else:
    train_noise = None
if test_noisy:
    test_noise = noise_amp
else:
    test_noise = None

def train_and_get_predictions(
        method : str,
        file_name : str,
        lib_length : int,
        async_sample_separation : int = 1,
        fixed_reg : bool = True,
        pred_reg : float = 1e-6,
        mapper_reg : float = 1e-6,
        ):
    
    # Fetch training and test signals.
    if grid_lib:
        train_library = rch.Library(
            data = None,
            parameters = None,
            parameter_labels = ["sigma", "omega"],
            data_generator = tst.get_lorenz,
            generator_args = {"transient_length" : transient_length,
                              "return_length" : lib_length + transient_length,
                              "return_dims" : return_dims},
            seed = None
            )
        train_library.generate_grid(
            ranges = [sigma_rng + [int(np.sqrt(lib_size))], omega_rng + [int(np.sqrt(lib_size))]],
            seed = lib_seed
            )
    else:
        lib_sigmas = np.random.default_rng(lib_seed).uniform(
            low = sigma_rng[0], high = sigma_rng[1], size = lib_size)
        lib_omegas = np.random.default_rng(lib_seed**lib_seed).uniform(
            low = omega_rng[0], high = omega_rng[1], size = lib_size)
        lib_params = list(zip(lib_sigmas, lib_omegas))
        train_library = rch.Library(
            data = None,
            parameters = lib_params,
            parameter_labels = ["sigma", "omega"],
            data_generator = tst.get_lorenz,
            generator_args = {"transient_length" : transient_length,
                              "return_length" : lib_length + transient_length,
                              "return_dims" : return_dims},
            seed = lib_seed
            )
        train_library.generate_data()
        
    if grid_val:
        test_library = rch.Library(
            data = None,
            parameters = None,
            parameter_labels = ["sigma", "omega"],
            data_generator = tst.get_lorenz,
            generator_args = {"transient_length" : transient_length,
                              "return_length" : val_length,
                              "return_dims" : return_dims},
            seed = None
            )
        test_library.generate_grid(
            ranges = [val_sigma_rng + [int(np.sqrt(val_size))],
                      val_omega_rng + [int(np.sqrt(val_size))]],
            seed = val_seed
            )    
    else:
        test_sigmas = np.random.default_rng(val_seed).uniform(
            low = val_sigma_rng[0], high = val_sigma_rng[1], size = val_size)
        test_omegas = np.random.default_rng(val_seed**val_seed).uniform(
            low = val_omega_rng[0], high = val_omega_rng[1], size = val_size)
        test_params = list(zip(test_sigmas, test_omegas))
        test_library = rch.Library(
            data = None,
            parameters = test_params,
            parameter_labels = ["sigma", "omega"],
            data_generator = tst.get_lorenz,
            generator_args = {"transient_length" : transient_length,
                              "return_length" : val_length,
                              "return_dims" : return_dims},
            seed = val_seed
            )
        test_library.generate_data()

    print("Train: ", len(train_library.data))
    print("Test: ", len(test_library.data))

    noiseless_train_library = train_library.copy()
    noiseless_test_library = test_library.copy()

    if train_noise is not None:
        train_library.data = [signal + train_noise * np.std(signal, axis = 0) * np.random.normal(
            size = signal.shape) for signal in noiseless_train_library.data]
    if test_noise is not None:
        scale_arr = np.copy(noiseless_train_library.data[0])
        for signal in noiseless_train_library.data[1:]:
            scale_arr = np.concatenate((scale_arr, signal), axis = 0)
        scale = np.std(scale_arr, axis = 0)
        if noisy_targets:
            test_library.data = [signal + test_noise * scale * np.random.normal(
                size = signal.shape) for signal in noiseless_test_library.data]
        else:
            test_library.data = [
                np.concatenate((
                    test_noise*scale*np.random.normal(size = signal[:test_length].shape),
                    np.zeros(signal[test_length:].shape)),
                axis = 0) + signal for i, signal in enumerate(noiseless_test_library.data)
                ]
        del(scale_arr)
        del(scale)
    
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
    
    extra_train_args = {"batch_size" : pred_batch_size, "accessible_drives" : pred_accessible_drives}
    _ = train_library.set_library_RCs(
        pred_esn = rc.ESN(**pred_esn_args),
        transient_length = transient_length,
        train_args = {"regression" : pred_regression, "feature_function" : pred_feature,
                      "batch_size" : pred_batch_size, "accessible_drives" : pred_accessible_drives}
        )    
    
    def save_predictions(
            predictions:        list,
            method_label:       str,
            file_name:          str,
            transient_length:   Union[int, list[int]]
            ):
        
        # save_loc is relative to current working directory
        
        if same_RCs:
            map_esn_args["seed"] =  pred_esn_args["seed"]
        else:
            map_esn_args["seed"] = 10000 -  pred_esn_args["seed"]
            
        experimental_parameters = {
            "test_signal_length" : test_length,
            "transient_length" : transient_length,
            "seed" : pred_esn_args["seed"],
            "map_seed" : map_esn_args["seed"],
            "open_loop" : False,
            "pred_esn_args" : pred_esn_args,
            "map_esn_args" : map_esn_args,
            }
        optimization_info = {}
        prediction_methods = [method_label]
        saved_predictions = {method_label : predictions}
        feature_functions = {
            "pred_feature" : pred_feature,
            "map_feature" : mapper_feature
            }
        
        data = rch.Run_Result(
            run_label = run_label,
            experimental_parameters = experimental_parameters,
            optimization_info = optimization_info,
            prediction_methods = prediction_methods,
            predictions = saved_predictions,
            feature_functions = feature_functions,
            pred_regularizations = None,
            map_regularizations = None,
            )
        
        run_directory = os.path.join(os.getcwd(), run_label)
        save_loc = os.path.join(run_directory, method_label)
        save_loc = os.path.join(save_loc, str(pred_esn_args["seed"]))
        data.save(save_loc = save_loc, safe_save = safe_save,
                  file_name = file_name)
        print("Saved")
    
    
    def get_async_predictions(seed: int, test_length: int,
                              incl_ri: bool = False, incl_rf: bool = False):
        
        return rch.Async_SM_Train_and_Predict(
            seed = seed,
            run_label = run_label,
            file_name = file_name,
            pred_esn_args = pred_esn_args,
            mapper_esn_args = map_esn_args,
            library_signals = train_library.data,
            test_signals = test_library.data,
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
        
    if method == "vanilla":
        
        predictions = rch.get_vanilla_predictions(
                test_library = test_library,
                test_length = test_length,
                predict_length = None,
                pred_esn = rc.ESN(**pred_esn_args),
                transient_length = None,
                pred_regression = pred_regression,
                pred_feature = pred_feature,
                extra_train_args = extra_train_args,
                rmse_only = rmse_only,
                reduce_predictions = reduce_predictions
                )
        
        if save_data:
            save_predictions(
                predictions = predictions,
                file_name = file_name,
                method_label = "vanilla",
                transient_length = min(10, test_length // 10)
                )        
        
    if method == "long_vanilla":
        
        alt_library = rch.Library(
            data = None,
            parameters = test_library.parameters,
            parameter_labels = test_library.parameter_labels,
            data_generator = test_library.data_generator,
            generator_args = {"transient_length" : transient_length,
                              "return_length" : lib_length + transient_length,
                              "return_dims" : return_dims},
            seed = val_seed + val_size
            )
        alt_library.generate_data()
        
        predictions = rch.train_same_dynamics_and_predict(
                library = alt_library,
                test_library = test_library,
                test_length = test_length,
                predict_length = None,
                pred_esn = rc.ESN(**pred_esn_args),
                transient_length = transient_length,
                pred_regression = pred_regression,
                pred_feature = pred_feature,
                extra_train_args = extra_train_args,
                rmse_only = rmse_only,
                reduce_predictions = reduce_predictions
                )
        
        if save_data:
            save_predictions(
                predictions = predictions,
                file_name = file_name,
                method_label = "long_vanilla",
                transient_length = transient_length
                )
        
    if method == "library_interpolation":
        
        predictions = rch.library_interpolate_and_predict(
                library = train_library,
                test_library = test_library,
                test_length = test_length,
                predict_length = None,
                transient_length = transient_length,
                pred_regression = pred_regression,
                pred_feature = pred_feature,
                extra_train_args = extra_train_args,
                rmse_only = rmse_only,
                reduce_predictions = reduce_predictions,
                interp_type = "linear",
                rescale_axes = True
                )
        
        if save_data:
            save_predictions(
                predictions = predictions,
                file_name = file_name,
                method_label = "library_interpolation",
                transient_length = transient_length
                )
    
    if method == "batch":
        
        predictions = rch.train_batch_and_predict(
                library = train_library,
                test_library = test_library,
                test_length = test_length,
                predict_length = None,
                transient_length = transient_length,
                pred_regression = pred_regression,
                pred_feature = pred_feature,
                extra_train_args = extra_train_args,
                rmse_only = rmse_only,
                reduce_predictions = reduce_predictions
                )
        
        if save_data:
            save_predictions(
                predictions = predictions,
                file_name = file_name,
                method_label = "batch",
                transient_length = transient_length
                )
    
    if method == "nearest_euclidean":
        
        predictions = rch.library_interpolate_and_predict(
                library = train_library,
                test_library = test_library,
                test_length = test_length,
                predict_length = None,
                transient_length = transient_length,
                pred_regression = pred_regression,
                pred_feature = pred_feature,
                extra_train_args = extra_train_args,
                rmse_only = rmse_only,
                reduce_predictions = reduce_predictions,
                interp_type = "nearest",
                rescale_axes = True
                )
        
        if save_data:
            save_predictions(
                predictions = predictions,
                file_name = file_name,
                method_label = "nearest_euclidean",
                transient_length = transient_length
                )
            
    if method == "library_average":
        
        predictions = rch.average_library_weights_predict(
                library = train_library,
                test_library = test_library,
                test_length = test_length,
                predict_length = None,
                pred_feature = pred_feature,
                rmse_only = rmse_only,
                reduce_predictions = reduce_predictions
                )
        
        if save_data:
            save_predictions(
                predictions = predictions,
                file_name = file_name,
                method_label = "library_average",
                transient_length = transient_length
                )

if use_parallel:
    jlib.Parallel(n_jobs = -1)(jlib.delayed(
        train_and_get_predictions)(
            method = method,
            lib_length = lib_length,
            file_name = str(lib_length),
            fixed_reg = fixed_reg,
            pred_reg = pred_regs[method],
            mapper_reg = mapper_regs[method],
            async_sample_separation = async_sample_separation
            )
            for lib_length in lib_lengths
            )
else:
    for lib_length in lib_lengths:
        train_and_get_predictions(
            method = method,
            lib_length = lib_length,
            file_name = str(lib_length),
            fixed_reg = fixed_reg,
            pred_reg = pred_regs[method],
            mapper_reg = mapper_regs[method],
            async_sample_separation = async_sample_separation
            )