import test_systems as tst
import numpy as np
import rescompy as rc
import rescompy.features as features
import rescompy.regressions as regressions
import rc_helpers as rch
import os 
import pickle
from typing import Union, Literal, Generator
import sys
import logging
import shutil

method = sys.argv[2]

standardize = False
shift_map2 = True
uniform_lib = False
exclude_param_ranges = True
lib_param_seed = 3333
num_train = 5
file_name = None
test_length = int(sys.argv[1])

system_map = tst.get_logistic_map

def orbit_generator(
    r:                          Union[float, np.ndarray],
    r_power:                    Union[int, float]                    = 1,
    x0:                         Union[Literal['random'], np.ndarray] = 'random',
    transient_length:           int                                  = 5000,
    return_length:              int                                  = 100000,
    dynamical_noise:            float                                = 0,
    observational_noise:        float                                = 0,
    seed:                       Union[int, None, Generator]          = None
    ):
    
    generator = system_map(
        r = r,
        r_power = r_power,
        x0 = x0,
        transient_length = transient_length,
        return_length = return_length,
        dynamical_noise = dynamical_noise,
        observational_noise = observational_noise,
        IC_seed = seed,
        dynamical_noise_seed = seed + 1,
        observational_noise_seed = 2 * seed
        )
    
    return generator

b = -.5
parameter_name = "r"
r_power = 1

focus_rs = [3.5, 3.675, 3.9]
num_vals = 500

focus_rs = [3.65, 3.8]
focus_rs = [3.8, 3.92]
focus_rs = [3.8, 3.92]
focus_rs = [3.62, 3.92]
focus_rs = [3.61, 3.92]

lib_bounds = np.array([3.7, 3.8]) ** (1./r_power)
val_bounds = np.array([2.9, 4]) ** (1./r_power)
exclusions = [(3.625, 3.636), (3.661, 3.663), (3.736, 3.746), (3.828, 3.86)]

lm_transient = 1000
rc_transient = 50
pred_discard = 500
fit_length = 950
lib_length = fit_length + rc_transient
val_length = pred_discard + 500

seed = int(sys.argv[3])
lib_seed =  seed * 2
val_seed =  seed * 3
prc_seed = seed
sm_seed = 1000 - seed

train_dynamical_noise = 0
train_observational_noise = 0
test_dynamical_noise = 0
test_observational_noise = 0

run_label = "pseed"+ str(lib_param_seed) + "_seed" + str(seed)\
    + "_ntrain" + str(num_train) + "_nfit" + str(fit_length) \
    + "_noise" + str(train_observational_noise)
save_loc = os.path.join(os.getcwd(), "SDist_Seeds_Data")
save_loc = os.path.join(save_loc, run_label)
libraries_loc = os.path.join(save_loc, "libraries")
pred_save_loc = os.path.join(save_loc, method)

train_generator_args = {"transient_length" : lm_transient,
                         "return_length" : lib_length,
                         "r_power" : r_power,
                         "dynamical_noise" : train_dynamical_noise,
                         "observational_noise" : train_observational_noise}
val_generator_args = {"transient_length" : lm_transient,
                       "return_length" : val_length + test_length,
                       "r_power" : r_power,
                       "dynamical_noise" : test_dynamical_noise,
                       "observational_noise" : test_observational_noise}

save_data = True
safe_save = False
reduce_predictions = True
rmse_only = False

# Set RC hyperparameters
pred_esn_args = {
    'seed': prc_seed,
    'size': 500,
    'spectral_radius': .2,
    'leaking_rate': .2,
    'input_strength': 2.5,
    'bias_strength': .5,
    'connections': 3,
    'input_dimension': 1
    }

map_esn_args = { #Like other sections of paper
    'seed': sm_seed,
    'size': 1000,
    'spectral_radius': .9,
    'leaking_rate': .1,
    'input_strength': 2.5,
    'bias_strength': .5,
    'connections': 3,
    'input_dimension': 1
    }

pred_regs = {
    "async_sm" : 1e-6,
    "async_sm_ri" : 1e-6,
    "library_interpolation" : 1e-6,
    "nearest_euclidean" : 1e-6,
    "vanilla" : 1e-6,
    "multitask" : 1e-6
    }

mapper_regs = {
    "async_sm" : 1e-8,
    "async_sm_ri" : 1e-8,
    "library_interpolation" : np.nan,
    "nearest_euclidean" : np.nan,
    "vanilla" : np.nan,
    "multitask" : np.nan
    }

# Set the prediciton RC feature function.
pred_feature = features.StatesOnly()

# Establish the regression routine for the prediction RC.
pred_batch_size = 10
pred_accessible_drives = -1
    
# Establish the regression routine for the signal mapper.
mapper_batch_size = 100
mapper_batch_length = 1000
mapper_accessible_drives = -1
    
# Establish the regression routine for the async signal mapper.
async_mapper_batch_size = 100
async_mapper_accessible_drives = list(np.arange(-100, 0, 1))
async_sample_separation = 1
mapper_feature = features.FinalStateOnly()

if uniform_lib:
    lib_rs = np.linspace(lib_bounds[0], lib_bounds[1], num_train)[:, None]
else:
    if exclude_param_ranges:
        seed_counter = 0
        lib_rs = []
        while len(lib_rs) < num_train:
            valid = True
            trial_r = np.random.default_rng(lib_param_seed + seed_counter).uniform(
                lib_bounds[0], lib_bounds[1], 1)[0]
            for bounds in exclusions:
                if trial_r >  bounds[0] and trial_r < bounds[1]:
                    valid = False
            if valid:
                lib_rs.append(trial_r)
            seed_counter += 1
        lib_rs = np.array(lib_rs)[:, None]
    else:
        lib_rs = np.random.default_rng(lib_param_seed).uniform(
            lib_bounds[0], lib_bounds[1], num_train)[:, None]
    
train_library = rch.Library(
    data = None,
    parameters = list(lib_rs),
    parameter_labels = [parameter_name],
    data_generator = orbit_generator,
    generator_args = train_generator_args,
    seed = lib_seed,
    standardize = standardize
    )
train_library.generate_data()

val_library = rch.Library(
    data = None,
    parameters = list(np.linspace(val_bounds[0], val_bounds[1], num_vals)[:, None]),
    parameter_labels = [parameter_name],
    data_generator = orbit_generator,
    generator_args = val_generator_args,
    seed = val_seed,
    standardize = standardize,
    standardizer = train_library.standardizer
    )
val_library.generate_data()

focus_library = rch.Library(
    data = None,
    parameters = list(np.array(focus_rs)[:, None]),
    parameter_labels = [parameter_name],
    data_generator = orbit_generator,
    generator_args = val_generator_args,
    seed = val_seed,
    standardize = standardize,
    standardizer = train_library.standardizer
    )
focus_library.generate_data()

mapper_regression = regressions.batched_ridge(
        regularization = mapper_regs[method]
        )
    
pred_regression = regressions.batched_ridge(
    regularization = pred_regs[method]
    )

extra_train_args = {"batch_size" : pred_batch_size, "accessible_drives" : pred_accessible_drives}
_ = train_library.set_library_RCs(
    pred_esn = rc.ESN(**pred_esn_args),
    transient_length = rc_transient,
    train_args = {"regression" : pred_regression, "feature_function" : pred_feature,
                  "batch_size" : pred_batch_size, "accessible_drives" : pred_accessible_drives}
    )

if method == "vanilla":
    if test_length == 2:
        vanilla_transient = 0
    else:
        vanilla_transient = min(5, int(.5 * test_length))
    predictions = rch.get_vanilla_predictions(
        test_library = val_library,
        test_length = test_length,
        predict_length = None,
        pred_esn = rc.ESN(**pred_esn_args),
        transient_length = vanilla_transient,
        pred_regression = pred_regression,
        pred_feature = pred_feature,
        extra_train_args = extra_train_args,
        rmse_only = rmse_only,
        reduce_predictions = reduce_predictions
        )
    
if method == "async_sm_ri":

    mapper_train_result = rch.Async_SM_Train(
        pred_esn_args = pred_esn_args,
        mapper_esn_args = map_esn_args,
        library_signals = train_library.data,
        test_length = test_length,
        transient_length = rc_transient,
        sample_separation = async_sample_separation,
        incl_ri = True,
        incl_rf = False,
        incl_weights = True,
        pred_feature = pred_feature,
        mapper_feature = mapper_feature,
        pred_regression = pred_regression,
        mapper_regression = mapper_regression,
        mapper_accessible_drives = mapper_accessible_drives,
        pred_batch_size = pred_batch_size,
        mapper_batch_size = mapper_batch_size,
        )
    
    predictions = rch.Async_SM_Predict(
        mapper_train_result = mapper_train_result,
        pred_esn_args = pred_esn_args,
        mapper_esn_args = map_esn_args,
        test_signals = val_library.data,
        test_length = test_length,
        incl_ri = True,
        incl_rf = False,
        predict_length = None,
        pred_feature = pred_feature,
        mapper_feature = mapper_feature,
        reduce_predictions = reduce_predictions,
        rmse_only = rmse_only,
        )
    
if method == "async_sm":

    mapper_train_result = rch.Async_SM_Train(
        pred_esn_args = pred_esn_args,
        mapper_esn_args = map_esn_args,
        library_signals = train_library.data,
        test_length = test_length,
        transient_length = rc_transient,
        sample_separation = async_sample_separation,
        incl_ri = False,
        incl_rf = False,
        incl_weights = True,
        pred_feature = pred_feature,
        mapper_feature = mapper_feature,
        pred_regression = pred_regression,
        mapper_regression = mapper_regression,
        mapper_accessible_drives = mapper_accessible_drives,
        pred_batch_size = pred_batch_size,
        mapper_batch_size = mapper_batch_size,
        )
    
    predictions = rch.Async_SM_Predict(
        mapper_train_result = mapper_train_result,
        pred_esn_args = pred_esn_args,
        mapper_esn_args = map_esn_args,
        test_signals = val_library.data,
        test_length = test_length,
        incl_ri = False,
        incl_rf = False,
        predict_length = None,
        pred_feature = pred_feature,
        mapper_feature = mapper_feature,
        reduce_predictions = reduce_predictions,
        rmse_only = rmse_only,
        )

if method == "library_interpolation":
    
    predictions = rch.library_interpolate_and_predict(
            library = train_library,
            test_library = val_library,
            test_length = test_length,
            predict_length = None,
            transient_length = rc_transient,
            pred_regression = pred_regression,
            pred_feature = pred_feature,
            extra_train_args = extra_train_args,
            rmse_only = rmse_only,
            reduce_predictions = reduce_predictions,
            interp_type = "linear1D",
            rescale_axes = True
            )

if method == "multitask":
    
    predictions = rch.train_batch_and_predict(
            library = train_library,
            test_library = val_library,
            test_length = test_length,
            predict_length = None,
            transient_length = rc_transient,
            pred_regression = pred_regression,
            pred_feature = pred_feature,
            extra_train_args = extra_train_args,
            rmse_only = rmse_only,
            reduce_predictions = reduce_predictions
            )
    
# Save predictions
if save_data:
    
    train_library.save(
        save_loc = libraries_loc,
        safe_save = safe_save,
        file_name = "train_library"
        )
    val_library.save(
        save_loc = libraries_loc,
        safe_save = safe_save,
        file_name = "val_library"
        )
    focus_library.save(
        save_loc = libraries_loc,
        safe_save = safe_save,
        file_name = "focus_library"
        )
    
    data_loc = os.path.join(pred_save_loc, str(test_length))
    if file_name is None:
        # Check if the path exists
        # Overwrite it if safe_save is False; raise Exception if True.
        if os.path.isdir(data_loc):
            if safe_save:
                msg = f"Already folder or file at '{data_loc}' and " \
                    "safe_save is True."
                logging.error(msg)
                raise FileExistsError(msg)
            else:
                shutil.rmtree(data_loc)
                msg = f"Already a folder or file at '{data_loc}' but " \
                    "safe_save is False; deleting the existing " \
                    "files and folders."
                logging.info(msg)
                
        os.makedirs(data_loc)
        with open(os.path.join(data_loc, "predictions.pickle"), 'wb') as temp_file:
            pickle.dump(predictions, temp_file)                
    else:
        # Check if the path exists
        # Overwrite it if safe_save is False; raise Exception if True.
        head, tail = os.path.split(os.path.join(data_loc, file_name))
        if os.path.isdir(head):
            if os.path.exists(os.path.join(data_loc, file_name + ".pickle")):
                if safe_save:
                    msg = f"Already folder or file at '{data_loc}' and " \
                        "safe_save is True."
                    logging.error(msg)
                    raise FileExistsError(msg)
                else:
                    msg = f"Already a folder or file at '{data_loc}' but " \
                          "safe_save is False; deleting the existing " \
                          "files and folders."
                    logging.info(msg)
                    
        else:
            os.makedirs(head)
        
        with open(os.path.join(data_loc, file_name + ".pickle"), 'wb') as temp_file:
            pickle.dump(predictions, temp_file) 
