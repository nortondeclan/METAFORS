import test_systems as tst
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import rescompy as rc
import rescompy.features as features
import rescompy.regressions as regressions
import rc_helpers as rch
import os
import pickle
from typing import Union, Literal, Generator
import climate_helpers as climate

methods = [
    "async_sm_ri",
    "async_sm",
    "library_interpolation",
    "multitask",
    "vanilla"
    ]

ymin, ymax = 0, 1
map_error_ylim = 1
uniform_lib = False
exclude_param_ranges = True
lib_param_seed = 3333
make_figure_3 = True
make_figure_9 = False
if make_figure_3:
    plot_heatmaps = True
    test_lengths = [5]
elif make_figure_9:
    plot_heatmaps = False #True
    test_lengths = [2]

r_power = 1
system_map = tst.get_logistic_map
heat_ymax = 20

exclusions = [(3.625, 3.636), (3.661, 3.663), (3.736, 3.746), (3.828, 3.86)]

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

def get_analytic_map(r, r_power):
    return lambda x : r ** r_power * x * (1 - x)

alpha = .5
marker_size = .05
cumm_dists_ms = 3
label_plots = False
plot_train_lines = True
plot_train_points = False
use_legends = True
highlight_divergence = True
divergence_alpha = 0.25
divergence_color = "yellow"

font_size = 16.
num_train = 5
lib_bounds = np.array([3.75, 3.8]) ** (1./r_power)
lib_bounds = np.array([3.7, 3.8]) ** (1./r_power)
lib_rs = np.linspace(lib_bounds[0], lib_bounds[1], num_train)[:, None]

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

focus_rs = [3.61]
num_vals = 500
val_bounds = np.array([2.75, 4]) ** (1./r_power)
val_bounds = np.array([2.9, 4]) ** (1./r_power)
lm_transient = 1000
rc_transient = 50
pred_discard = 500
fit_length = 950
lib_length = fit_length + rc_transient
val_length = pred_discard + 500
hline_lengths = test_lengths.copy()
seed = 1000
lib_seed = 10
val_seed = 11
train_dynamical_noise = 0
train_observational_noise = 0
test_dynamical_noise = 0
test_observational_noise = 0
train_label = "Training Systems\n(" + str(lib_length) + " Iterations)"

save_data = False
safe_save = False
reduce_predictions = False
rmse_only = False
method_colors = {
    "library_interpolation" : "tab:pink",
    "async_sm_ri" : "tab:red",
    "async_sm" : "tab:blue",
    "multitask" : "tab:orange",
    "vanilla" : "tab:green"
    }
lib_color =  "black"
truth_color = "black"
lib_linewidth = None
method_labels = {
    "library_interpolation" : "Interpolation/Extrapolation\n(Typically Infeasible)",
    "async_sm_ri" : "METAFORS",
    "async_sm" : "METAFORS, Parameters Only",
    "multitask" : "Multi-task Learning",
    "vanilla" : "Training on the Test Signal"
    }
plot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# Set RC hyperparameters
pred_esn_args = {
    'seed': 1,
    'size': 500,
    'spectral_radius': .2,
    'leaking_rate': .2,
    'input_strength': 2.5,
    'bias_strength': .5,
    'connections': 3,
    'input_dimension': 1
    }
map_esn_args = {
    'seed': 9999,
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
pred_feature = features.StatesAndInputs()
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
    
train_library = rch.Library(
    data = None,
    parameters = list(lib_rs),
    parameter_labels = ["r"],
    data_generator = orbit_generator,
    generator_args = {"transient_length" : lm_transient,
                      "return_length" : lib_length,
                      "r_power" : r_power,
                      "dynamical_noise" : train_dynamical_noise,
                      "observational_noise" : train_observational_noise},
    seed = lib_seed
    )
train_library.generate_data()

def plot_prediction(
        prediction:     rc.PredictResult,
        parameter:      float,
        max_horizon:    int,
        fig:            mpl.figure.Figure = None,
        ax:             mpl.axes._axes.Axes = None,
        label_lines:    bool = True,
        frame_legend:   bool = False,
        ax_legend:      bool = True
        ):
    
    if ax is None:
        fig, ax = plt.subplots(
            1, 1,
            figsize = (15, 3),
            constrained_layout = True
            )
    
    ax.plot(
        np.arange(prediction.resync_inputs.shape[0]),
        prediction.resync_inputs[:, 0],
        color = "k",
        )
    if label_lines:
        label = "Truth"
    else:
        label = None
    l1 = ax.plot(
        np.arange(prediction.resync_inputs.shape[0] - 1,
                  prediction.resync_inputs.shape[0] + prediction.target_outputs.shape[0]),
        np.concatenate((prediction.resync_inputs[-1, 0].reshape((1)),
                        prediction.target_outputs[:, 0])),
        color = "k",
        label = label
        )
    if label_lines:
        label = "Prediction"
    else:
        label = None
    l2 = ax.plot(
        np.arange(prediction.resync_inputs.shape[0],
                  prediction.resync_inputs.shape[0] + prediction.reservoir_outputs.shape[0]),
        prediction.reservoir_outputs[:, 0],
        color = "tab:red",
        label = label,
        linestyle = "dotted"
        )
    if label_lines:
        label = "Loop Closed"
    else:
        label = None
    l3 = ax.axvline(x = prediction.resync_inputs.shape[0] - 1,
               linestyle = "--", color = "k",
               label = label)
    ax.set_ylabel("$x_n$")
    if prediction.reservoir_outputs.shape[0] > max_horizon:
        ax.set_xlim(right = max_horizon, left = -1)
        
    if ax_legend:
        leg_1 = ax.legend(handles = [l1, l2], labels = ["Truth", "Prediction"], loc = "upper right", frameon = False, ncols = 2)
        leg_2 = ax.legend(handles = [l3], labels = ["Loop Closed"], loc = "lower right", frameon = False)
        ax.add_artist(leg_1)
        ax.add_artist(leg_2)
        
    ax.set_xlabel("Iteration, $n$")
    fig.patch.set_alpha(0)

with mpl.rc_context({"font.size" : font_size}):
    if plot_heatmaps:
        bfig, baxs = plt.subplots(len(test_lengths) + 2, len(methods),
                                  constrained_layout = True,
                                  figsize = ((6 * len(methods), 12)))
    else:
        bfig, baxs = plt.subplots(len(test_lengths) + 1, len(methods),
                                  constrained_layout = True,
                                  figsize = ((5 * len(methods), 8)))
    
    for tli, test_length in enumerate(test_lengths):
        
        val_library = rch.Library(
            data = None,
            parameters = list(np.linspace(val_bounds[0], val_bounds[1], num_vals)[:, None]),
            parameter_labels = ["r"],
            data_generator = orbit_generator,
            generator_args = {"transient_length" : lm_transient,
                              "return_length" : val_length + test_length,
                              "r_power" : r_power,
                              "dynamical_noise" : test_dynamical_noise,
                              "observational_noise" : test_observational_noise},
            seed = val_seed
            )
        val_library.generate_data()
        
        focus_library = rch.Library(
            data = None,
            parameters = list(np.array(focus_rs)[:, None]),
            parameter_labels = ["r"],
            data_generator = orbit_generator,
            generator_args = {"transient_length" : lm_transient,
                              "return_length" : val_length + test_length,
                              "r_power" : r_power,
                              "dynamical_noise" : test_dynamical_noise,
                              "observational_noise" : test_observational_noise},
            seed = val_seed
            )
        focus_library.generate_data()
        
        for mi, method in enumerate(methods):
            
            if method == "vanilla":
                plot_train_lines = False
            else:
                plot_train_lines = True
            
            if mi == 0:
                ax_legend = True
                label_lines = True
            else:
                ax_legend = False
                label_lines = False
            
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
            
            def get_async_predictions(seed: int, test_length: int,
                                      test_library: rch.Library,
                                      incl_ri: bool = False, incl_rf: bool = False):
                
                return rch.Async_SM_Train_and_Predict(
                    seed = seed,
                    file_name = str(test_length),
                    run_label = None,
                    pred_esn_args = pred_esn_args,
                    mapper_esn_args = map_esn_args,
                    library_signals = train_library.data,
                    test_signals = test_library.data,
                    test_length = test_length,
                    transient_length = rc_transient,
                    sample_separation = async_sample_separation,
                    incl_ri = incl_ri,
                    incl_rf = incl_rf,
                    same_seed = False,
                    predict_length = None,
                    pred_feature = pred_feature,
                    mapper_feature = mapper_feature,
                    pred_regression = pred_regression,
                    mapper_regression = mapper_regression,
                    mapper_accessible_drives = async_mapper_accessible_drives,
                    pred_batch_size = pred_batch_size,
                    mapper_batch_size = async_mapper_batch_size,
                    save = save_data,
                    safe_save = safe_save,
                    reduce_predictions = reduce_predictions,
                    rmse_only = rmse_only
                    )
            
            if method == "async_sm_ri":
                predictions = get_async_predictions(
                    seed = pred_esn_args["seed"],
                    test_length = test_length,
                    test_library = val_library,
                    incl_ri = True
                    )
                focus_predictions = get_async_predictions(
                    seed = pred_esn_args["seed"],
                    test_length = test_length,
                    test_library = focus_library,
                    incl_ri = True
                    )
                
            if method == "async_sm":
                predictions = get_async_predictions(
                    seed = pred_esn_args["seed"],
                    test_length = test_length,
                    test_library = val_library,
                    incl_ri = False
                    )
                focus_predictions = get_async_predictions(
                    seed = pred_esn_args["seed"],
                    test_length = test_length,
                    test_library = focus_library,
                    incl_ri = False
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
                
                focus_predictions = rch.train_batch_and_predict(
                        library = train_library,
                        test_library = focus_library,
                        test_length = test_length,
                        predict_length = None,
                        transient_length = rc_transient,
                        pred_regression = pred_regression,
                        pred_feature = pred_feature,
                        extra_train_args = extra_train_args,
                        rmse_only = rmse_only,
                        reduce_predictions = reduce_predictions
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
                
                focus_predictions = rch.library_interpolate_and_predict(
                        library = train_library,
                        test_library = focus_library,
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
                
                focus_predictions = rch.get_vanilla_predictions(
                    test_library = focus_library,
                    test_length = test_length,
                    predict_length = None,
                    pred_esn = rc.ESN(**pred_esn_args),
                    transient_length = vanilla_transient,
                    pred_regression = pred_regression,
                    pred_feature = pred_feature,
                    extra_train_args = extra_train_args,
                    rmse_only = rmse_only,
                    reduce_predictions = reduce_predictions,
                    append_resync_inputs = True
                    )
                
            plot_prediction(
                prediction = focus_predictions[0],
                parameter = focus_library.parameters[0],
                max_horizon = 30,
                fig = bfig,
                ax = baxs[0, mi],
                label_lines = label_lines,
                ax_legend = ax_legend
                )
            
            if mi == 0:
                ylabel = "$x$"
            else:
                ylabel = None
            remove_yticks = False
            train_label_i = None
            label_truth = True
            
            if tli == len(test_lengths) - 1:
                remove_xticks = False
            else:
                remove_xticks = False
                
            # Overlay the predictions on the true bifurcation diagram.
            climate.bifurcation_diagram(
                predictions = predictions,
                pred_discard = pred_discard,
                truth_library = val_library,
                train_library = train_library,
                highlight_divergence = highlight_divergence,
                divergence_alpha = divergence_alpha,
                divergence_color = divergence_color,
                marker_size = marker_size,
                truth_color = truth_color,
                train_color = lib_color,
                pred_color = method_colors[method],
                ymin = ymin,
                ymax = ymax,
                font_size = font_size,
                alpha = alpha,
                xlabel = None,
                ylabel = ylabel,
                train_linewidth = lib_linewidth,
                plot_train_lines = plot_train_lines,
                plot_train_points = plot_train_points,
                use_legend = use_legends,
                legend_loc = "lower left",
                ax = baxs[tli + 1, mi],
                remove_xticks = remove_xticks,
                remove_yticks = remove_yticks,
                train_label = train_label_i,
                label_truth = label_truth,
                add_text = None
                )
    
    for ax in baxs[0, :]:
        ax.set_ylabel("$x_n$")
    for ax in baxs[1, :]:
        ax.set_ylabel("$x$")
    if plot_heatmaps:
        for ax in baxs[2, :]:
            ax.set_ylabel("Test Length, $N_{test}$")
    
    baxs[0, 0].set_ylabel("$x_n$")
    for i in range(len(methods)):
        baxs[0, i].set_ylim(0, 1.2)

    for ax_i in baxs[1, :]:
        ax_i.set_xlabel("Logistic Parameter, $r$")
    
    for ax_i in baxs[-1, :]:
        ax_i.set_xlabel("Logistic Parameter, $r$")

if plot_heatmaps:
    alpha = .5
    marker_size = .1
    label_plots = False
    divergence_alpha = 0.25
    colormap_logy = False
    colormap_logc = False
    
    font_size = 15.
    ymin, ymax = None, None
    ymin, ymax = None, None
    map_error_ylim = .2
    
    
    lib_param_seed = 3333
    num_train = 5
    fit_length = 950
    noise = 0
    rc_transient = 50
    lib_length = rc_transient + fit_length
    focus_lengths = []
    pred_discard = 500
    normalize_wasserstein = False
    manual_cmax = True
    manual_cmin = True
    fixed_cmax = 1e1
    fixed_cmin = 1e-3
    colormap = "RdBu_r"
    color_over = mpl.colormaps["Reds_r"](0)
    normalize_maperror = False
    avg_type = "Mean"
    avg_type = "Median"
    
    colorbar_label = "Autonomous One-step Error"
    add_method_label = False
    
    folder_loc = os.path.join(os.getcwd(), "SDist_Seeds_Data")
    run_labels = os.listdir(folder_loc)
    
    font_size = 15.
    lib_color = "black"
    truth_color = "black"
    method_labels = {
        "library_interpolation" : "Interpolation/Extrapolation\n   (Typically Infeasible)",
        "lib_interp" : "Interpolation/Extrapolation\n    (Typically Infeasible)",
        "async_sm_ri" : "METAFORS",
        "async_sm" : "METAFORS, Parameters Only",
        "multitask" : "Multi-task Learning",
        "batch" : "Multi-task Learning",
        "vanilla" : "Training on the Test Signal"
        }
    train_label = "Training Systems\n(" + str(lib_length) + " Iterations)"
    legend_locs = ["center left", "best"]
    legend_locs = [None, "right"]
    plot_labels = ["(a) Logistic Map", "(b) Gauss Iterated Map",
                   "(c) Learned Maps", "(d) Cumulative Probabilities"]
    
    alpha = .5
    marker_size = .1
    label_plots = False
    plot_train_lines = True
    plot_train_points = False
    use_legends = True
    highlight_divergence = True
    divergence_alpha = 0.25
    divergence_color = "yellow"
    colormap_logy = False
    colormap_logc = False
    
    font_size = 15.
    ymin, ymax = None, None
    ymin, ymax = None, None
    map_error_ylim = .2
    
    methods = ["async_sm_ri", "async_sm", "lib_interp", "multitask", "vanilla"]
    methods_performance = {method: np.nan * np.ones((10, 40, 500)) for method in methods}
    found_vals = False
    try:
        with open("Single_Dist_10" + avg_type + "s.pickle", "rb") as tmp_file:
            methods_avgs = pickle.load(tmp_file)
        loaded_avgs = True
    except:
        loaded_avgs = False
        for si, run_label in enumerate(run_labels):
            save_loc = os.path.join(folder_loc, run_label)
            
            methods_2 = os.listdir(save_loc)
            if "libraries" in methods_2:
                methods_2.remove("libraries")
                libraries_loc = os.path.join(save_loc, "libraries")
            else:
                libraries_loc = None
            
            if not found_vals:
                if libraries_loc is None:
                    train_library = None
                    val_library = None
                    focus_library = None 
                else:
                    with open(os.path.join(libraries_loc, "val_library.pickle"), 'rb') as temp_file:
                        val_library = pickle.load(temp_file)
                        found_vals = True
                        
            for method_ind, method in enumerate(methods):
                method_loc = os.path.join(save_loc, method)
                if method in methods_2:
                    test_lengths = sorted([int(length) for length in os.listdir(method_loc)])
                    
                    for ti, test_length in enumerate(test_lengths):
                        try:
                            test_length_loc = os.path.join(method_loc, str(test_length))
                            with open(os.path.join(test_length_loc, "predictions.pickle"), 'rb') as temp_file:
                                predictions = pickle.load(temp_file)
                                
                            methods_performance[method][si, test_length - 1] = np.array([climate.get_map_error( #ti] = np.array([climate.get_map_error(
                                predictions = prediction,
                                analytic_map = get_analytic_map(parameter),
                                discard = pred_discard,
                                normalize = normalize_maperror
                                )[0]
                                for parameter, prediction in zip(
                                        val_library.parameters, predictions[:len(val_library.data)])
                                ])
                        
                        except:
                            print("Not found: " + method + " s" + str(si + 1) + " t" + str(test_length))
                        
                else:
                    print("Folder not found: " + method + " s" + str(si + 1))
        
        methods_avgs = {method: None for method in methods}
        for method in methods:
            if avg_type == "Median":
                methods_avgs[method] = np.nanmedian(methods_performance[method], axis = 0)
            elif avg_type == "Mean":
                methods_avgs[method] = np.nanmean(methods_performance[method], axis = 0)
    
    cmin = np.min([methods_avgs[method].min() for method in methods])
    cmax = np.max([methods_avgs[method].max() for method in methods])
    
    extend = "neither"
    if manual_cmax:
        cmax = fixed_cmax
        extend = "max"
    if manual_cmin:
        cmin = fixed_cmin
        extend = "min"
    if manual_cmin and manual_cmax:
        extend = "both"
    
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    cmap = mpl.colormaps[colormap]
    cmap.set_over(color_over)
    with mpl.rc_context({"font.size" : font_size}):
        
        haxs = baxs[-1, :].reshape((1, -1))
        
        for method_ind, method in enumerate(methods):
            
            if loaded_avgs:
                test_lengths = np.arange(1, methods_avgs[method].shape[0] + 1)
            else:
                method_loc = os.path.join(save_loc, method)
                test_lengths = sorted([int(length) for length in os.listdir(method_loc)])
                test_lengths = np.arange(1, 40 + 1)
            
            test_lengths_incl = test_lengths[test_lengths <= heat_ymax]
            x, y = np.meshgrid(np.array(val_library.parameters), np.array(test_lengths_incl))
            pcm = haxs[0, method_ind].pcolormesh(
                x, y, methods_avgs[method][test_lengths <= heat_ymax, :], cmap = cmap,
                norm = mpl.colors.LogNorm(vmin = cmin, vmax = cmax, clip = False)
                )
            
            if add_method_label:
                haxs[0, method_ind].text(
                    .01, .99,
                    "(" + alphabet[method_ind] + ") " + method_labels[method],
                    ha = 'left', va = 'top',
                    weight = "bold",
                    transform = haxs[0, method_ind].transAxes
                    )
            
        for ax in haxs[:, 0]:
            ax.set_ylim(0)
            if colormap_logy:
                ax.set_yscale("log")
        for ax in haxs[0, :]:
            ax.set_facecolor("black")
            ax.set_xlabel("Logistic Parameter, $r$")  
            ax.set_ylim(0)
            for hline in hline_lengths:
                ax.axhline(y = hline, linestyle = "--", linewidth = 2, c = "tab:green")
        bfig.colorbar(pcm, ax = baxs[-1], label = colorbar_label, extend = extend)
        bfig.patch.set_alpha(0)