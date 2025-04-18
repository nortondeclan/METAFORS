import test_systems as tst
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import rescompy as rc
import rescompy.features as features
import rescompy.regressions as regressions
import rc_helpers as rch
from typing import Union, Literal, Generator
import climate_helpers as climate

method = "async_sm_ri"

standardize = False
shift_map2 = True
uniform_lib = False
exclude_param_ranges = True
lib_param_seed = 111
num_train = 5

system_map1 = tst.get_logistic_map
system_map2 = tst.get_gauss_map

def orbit_generator1(
    r:                          Union[float, np.ndarray],
    r_power:                    Union[int, float]                    = 1,
    x0:                         Union[Literal['random'], np.ndarray] = 'random',
    transient_length:           int                                  = 5000,
    return_length:              int                                  = 100000,
    dynamical_noise:            float                                = 0,
    observational_noise:        float                                = 0,
    seed:                       Union[int, None, Generator]          = None
    ):
    
    generator = system_map1(
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

def orbit_generator2(
        a:                          Union[float, np.ndarray]             = 5,
        b:                          Union[float, np.ndarray]             = -.5,
        x0:                         Union[Literal['random'], np.ndarray] = 'random',
        transient_length:           int                                  = 5000,
        return_length:              int                                  = 100000,
        dynamical_noise:            float                                = 0,
        observational_noise:        float                                = 0,
        seed:                       Union[int, None, Generator]          = None
        ):
    
    generator = system_map2(
        a = a,
        b = b,
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

def add_libraries(
        library1:       rch.Library,
        library2:       rch.Library
        ):
    
    return rch.Library(
        data = library1.data + library2.data,
        parameters = library1.parameters + library2.parameters
        )

alpha = .5
marker_size = .05
label_plots = False
plot_train_lines = True
plot_train_points = False
use_legends = True
highlight_divergence = True
divergence_alpha = 0.25
divergence_color = "yellow"

font_size = 15.
ymin, ymax = None, None
ymin, ymax = None, None
map_error_ylim = .2
b = -.5
parameter_name1 = "r"
parameter_name2 = "a"
r_power = 1

if shift_map2:
    map2_standardizer = rc.Standardizer(u = np.zeros((1, 1)))
    map2_standardizer.scale = 1
    map2_standardizer.shift = -b
else:
    map2_standardizer = None

def get_analytic_map1(p):
    return lambda x : p ** r_power * x * (1 - x)

if shift_map2:
    lib_bounds1 = np.array([3.6, 3.9]) ** (1./r_power)
    lib_bounds2 = np.array([6, 12]) #12])
    exclusions1 = [(3.625, 3.636), (3.661, 3.663), (3.736, 3.746), (3.828, 3.86)]
    exclusions2 = [(6.725, 7.49), (8.1, 8.2), (9.21, 9.47), (10.67, 10.79), (11.57, 11.725)]
    if uniform_lib:
        lib_rs1 = np.linspace(lib_bounds1[0], lib_bounds1[1], num_train)[:, None]
        lib_rs2 = np.linspace(lib_bounds2[0], lib_bounds2[1], num_train)[:, None]
    else:
        if exclude_param_ranges:
            seed_counter = 0
            lib_rs1 = []
            while len(lib_rs1) < num_train:
                valid = True
                trial_r1 = np.random.default_rng(lib_param_seed + seed_counter).uniform(
                    lib_bounds1[0], lib_bounds1[1], 1)[0]
                for bounds in exclusions1:
                    if trial_r1 >  bounds[0] and trial_r1 < bounds[1]:
                        valid = False
                if valid:
                    lib_rs1.append(trial_r1)
                seed_counter += 1
                    
            seed_counter = 0
            lib_rs2 = []
            while len(lib_rs2) < num_train:
                valid = True
                trial_r2 = np.random.default_rng(lib_param_seed + seed_counter).uniform(
                    lib_bounds2[0], lib_bounds2[1], 1)[0]
                for bounds in exclusions2:
                    if trial_r2 >  bounds[0] and trial_r2 < bounds[1]:
                        valid = False
                if valid:
                    lib_rs2.append(trial_r2)
                seed_counter += 1
                    
            lib_rs1 = np.array(lib_rs1)[:, None]
            lib_rs2 = np.array(lib_rs2)[:, None]
        else:
            lib_rs1 = np.random.default_rng(lib_param_seed).uniform(
                lib_bounds1[0], lib_bounds1[1], num_train)[:, None]
            lib_rs2 = np.random.default_rng(lib_param_seed).uniform(
                lib_bounds2[0], lib_bounds2[1], num_train)[:, None]
        
    val_bounds1 = np.array([3.4, 4]) ** (1./r_power)
    val_bounds2 = np.array([4, 14])
    
    def get_analytic_map2(p):
        return lambda x : np.exp(- p * (x + b)**2)
    
else:
    lib_bounds1 = np.array([3.6, 3.9]) ** (1./r_power)
    lib_bounds2 = np.array([6, 12])
    if uniform_lib:
        lib_rs1 = np.linspace(lib_bounds1[0], lib_bounds1[1], num_train)[:, None]
        lib_rs2 = np.linspace(lib_bounds2[0], lib_bounds2[1], num_train)[:, None]
    else:
        lib_rs1 = np.random.default_rng(lib_param_seed).uniform(
            lib_bounds1[0], lib_bounds1[1], num_train)[:, None]
        lib_rs2 = np.random.default_rng(lib_param_seed).uniform(
            lib_bounds2[0], lib_bounds2[1], num_train)[:, None]
        
    val_bounds1 = np.array([3.4, 4]) ** (1./r_power)
    val_bounds2 = np.array([4, 14])
    
    def get_analytic_map2(p):
        return lambda x : np.exp(- p * x**2) + b

num_vals = 500
focus_rs1 = [3.61, 3.92]
focus_rs2 = [8., 11.]

lm_transient = 1000
rc_transient = 50
pred_discard = 500
fit_length = 950
lib_length = fit_length + rc_transient
val_length = pred_discard + 500
test_length = 10
seed = 1000
lib_seed = 10
val_seed = 11
train_dynamical_noise = 0
train_observational_noise = 0
test_dynamical_noise = 0
test_observational_noise = 0

train_generator_args1 = {"transient_length" : lm_transient,
                         "return_length" : lib_length,
                         "r_power" : r_power,
                         "dynamical_noise" : train_dynamical_noise,
                         "observational_noise" : train_observational_noise}
train_generator_args2 = {"transient_length" : lm_transient,
                         "return_length" : lib_length,
                         "b" : b,
                         "dynamical_noise" : train_dynamical_noise,
                         "observational_noise" : train_observational_noise}
val_generator_args1 = {"transient_length" : lm_transient,
                       "return_length" : val_length + test_length,
                       "r_power" : r_power,
                       "dynamical_noise" : test_dynamical_noise,
                       "observational_noise" : test_observational_noise}
val_generator_args2 = {"transient_length" : lm_transient,
                       "return_length" : val_length + test_length,
                       "b" : b,
                       "dynamical_noise" : test_dynamical_noise,
                       "observational_noise" : test_observational_noise}

save_data = False
safe_save = False
reduce_predictions = True #False
rmse_only = False
method_colors = {
    "library_interpolation" : "tab:pink",
    "async_sm_ri" : "tab:red"
    }
lib_color = "black"
truth_color = "black"
method_labels = {
    "library_interpolation" : "Interpolation/Extrapolation\n(Typically Infeasible)",
    "async_sm_ri" : "METAFORS"
    }
train_label = "Training Systems\n(" + str(lib_length) + " Iterations)"
legend_locs = ["center left", "best"]
legend_locs = [None, "right"]
legend_locs2 = [None, "center left"]
plot_labels = ["(a) Logistic Map", "(b) Gauss Iterated Map",
               "(c) Learned Maps", "(d) Cumulative Probabilities"]

# Set RC hyperparameters
pred_esn_args = {
    'seed': 1,
    'size': 500,
    'spectral_radius': .2,
    'leaking_rate': .2,
    'input_strength': 4,
    'bias_strength': .5,
    'connections': 3,
    'input_dimension': 1
    }

map_esn_args = {
    'seed': 9999,
    'size': 1000,
    'spectral_radius': .9,
    'leaking_rate': .1,
    'input_strength': 4,
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
    "batch" : 1e-6
    }

mapper_regs = {
    "async_sm" : 1e-8,
    "async_sm_ri" : 1e-8,
    "library_interpolation" : np.nan,
    "nearest_euclidean" : np.nan,
    "vanilla" : np.nan,
    "batch" : np.nan
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
    
train_library1 = rch.Library(
    data = None,
    parameters = list(lib_rs1),
    parameter_labels = [parameter_name1],
    data_generator = orbit_generator1,
    generator_args = train_generator_args1,
    seed = lib_seed,
    standardize = standardize
    )
train_library1.generate_data()
train_library2 = rch.Library(
    data = None,
    parameters = list(lib_rs2),
    parameter_labels = [parameter_name2],
    data_generator = orbit_generator2,
    generator_args = train_generator_args2,
    seed = lib_seed,
    standardizer = map2_standardizer,
    standardize = standardize or shift_map2
    )
train_library2.generate_data()    
train_library = add_libraries(library1 = train_library1, library2 = train_library2)

val_library1 = rch.Library(
    data = None,
    parameters = list(np.linspace(val_bounds1[0], val_bounds1[1], num_vals)[:, None]),
    parameter_labels = [parameter_name1],
    data_generator = orbit_generator1,
    generator_args = val_generator_args1,
    seed = val_seed,
    standardize = standardize,
    standardizer = train_library1.standardizer
    )
val_library1.generate_data()
val_library2 = rch.Library(
    data = None,
    parameters = list(np.linspace(val_bounds2[0], val_bounds2[1], num_vals)[:, None]),
    parameter_labels = [parameter_name2],
    data_generator = orbit_generator2,
    generator_args = val_generator_args2,
    seed = val_seed,
    standardize = standardize or shift_map2,
    standardizer = train_library2.standardizer
    )
val_library2.generate_data()
val_library = add_libraries(library1 = val_library1, library2 = val_library2)

focus_library1 = rch.Library(
    data = None,
    parameters = list(np.array(focus_rs1)[:, None]),
    parameter_labels = [parameter_name1],
    data_generator = orbit_generator1,
    generator_args = val_generator_args1,
    seed = val_seed,
    standardize = standardize,
    standardizer = train_library1.standardizer
    )
focus_library1.generate_data()
focus_library2 = rch.Library(
    data = None,
    parameters = list(np.array(focus_rs2)[:, None]),
    parameter_labels = [parameter_name2],
    data_generator = orbit_generator2,
    generator_args = val_generator_args2,
    seed = val_seed,
    standardize = standardize or shift_map2,
    standardizer = train_library2.standardizer
    )
focus_library2.generate_data()

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
    mapper_batch_size = mapper_batch_size
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

with mpl.rc_context({"font.size" : font_size}):
    fig, axs = plt.subplots(1, 2, constrained_layout = True, figsize = ((14, 6)))
    fig2, axs2 = plt.subplots(1, 2, constrained_layout = False, figsize = ((16, 8)))
    axs = axs.reshape((1, -1))
    axs2 = axs2.reshape((1, -1))
    start_id = 0
    
    for libi, library in enumerate([val_library1, val_library2]):
        
        if libi == 1:
            l_train_label = train_label
            l_method_label = method_labels[method]
            label_truth = True
        else:
            l_train_label = None
            l_method_label = None
            label_truth = False
        
        get_analytic_map = [get_analytic_map1, get_analytic_map2][libi]
        xlabel = ["Logistic Parameter, r", "Gauss Parameter, a"][libi]
        focus_p_label = ["r", "a"][libi]
        incl_legends = [False, use_legends][libi]
        legend_bbox1 = (1.55, .5)
        legend_bbox2 = (1.5, .5)
        legend_bbox3 = (1.5, .5)
        focus_rs = [focus_rs1, focus_rs2][libi]
        tlibrary = [train_library1, train_library2][libi]
        focus_libraries = [focus_library1, focus_library2]
        focus_library = focus_libraries[libi]
        ghostfocus_library = focus_libraries[libi-1]
        cumulative_probs_linewidth = 3
        cumm_dists_ms = 5
        
        focus_pcolors = np.array([["tab:orange", "tab:pink"], ["tab:cyan", "tab:olive"]])
        focus_tcolors = np.array([["tab:brown", "tab:purple"], ["tab:blue", "tab:green"]])
        
        focus_pred_colors = focus_pcolors[libi]
        focus_truth_colors = focus_tcolors[libi]
        ghostfocus_pred_colors = focus_pcolors[libi-1]
        ghostfocus_truth_colors = focus_tcolors[libi-1]
        
        preds = predictions[start_id: start_id + len(library.data)]
        
        start_id = len(library.data)
        
        if standardize:
            ymin, ymax = None, None
        elif shift_map2:
            ymin, ymax = -.1, 1.1
            dymin, dymax = .1, -.1
            statsmin, statsmax = ymin, ymax
        else:
            ymin, ymax = [0, b][libi] - .15, [1, -b][libi] + .1
            statsmin, statsmax = min(0, b), max(1, -b)
            dymin, dymax = [0, 0][libi], [0, 0][libi]
        remove_yticks = False
            
        # Overlay the predictions on the true bifurcation diagram.
        climate.bifurcation_diagram(
            predictions = preds,
            pred_discard = pred_discard,
            truth_library = library,
            train_library = tlibrary,
            focus_library = focus_library,
            focus_colors = focus_pred_colors,
            focus_label = focus_p_label,
            ghostfocus_library = None,
            ghostfocus_colors = ghostfocus_pred_colors,
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
            xlabel = xlabel,
            ylabel = "x",
            plot_train_lines = plot_train_lines,
            plot_train_points = plot_train_points,
            use_legend = False,
            legend_loc = legend_locs[libi],
            legend_bbox = legend_bbox1,
            prediction_method = l_method_label,
            ax = axs[0, libi],
            remove_yticks = remove_yticks,
            remove_xticks = False,
            train_label = l_train_label,
            add_text = None,
            label_truth = label_truth
            )
        
        focus_predictions = rch.Async_SM_Predict(
            mapper_train_result = mapper_train_result,
            pred_esn_args = pred_esn_args,
            mapper_esn_args = map_esn_args,
            test_signals = focus_library.data,
            test_length = test_length,
            incl_ri = True,
            incl_rf = False,
            predict_length = None,
            pred_feature = pred_feature,
            mapper_feature = mapper_feature,
            reduce_predictions = reduce_predictions,
            rmse_only = rmse_only,
            )
    
        for fi, r in enumerate(focus_library.parameters):
            
            frame_legend = False
            
            if fi == 0:
                text_to_add = plot_labels[2]
            else:
                text_to_add = None
            climate.plot_one_step_map(
                prediction = focus_predictions[fi],
                marker_size = 15 * cumm_dists_ms,
                analytic_xs = None,
                analytic_map = get_analytic_map(p = r),
                discard = pred_discard,
                truth_color = focus_truth_colors[fi],
                pred_color = focus_pred_colors[fi],
                plot_truth = True,
                scatter_truth = False,
                truth_linewidth = cumulative_probs_linewidth,
                ax = axs2[0, 0],
                add_text = None,
                label_lines = True,
                xmin = statsmin + dymin,
                xmax = statsmax + dymax,
                ymin = statsmin + dymin,
                ymax = statsmax,
                alpha = .25 * alpha,
                legend_loc = legend_locs[libi],
                legend_bbox = legend_bbox3,
                make_ax_legend = False,
                parameter_label = focus_p_label + f"$^*_{fi+1}$"
                )
        
        climate.plot_cumulative_probabilities(
            predictions = focus_predictions,
            truth_library = focus_library,
            interest_parameters = focus_library.parameters,
            truth_colors = focus_truth_colors,
            predicted_colors = focus_pred_colors,
            val_min = statsmin + dymin,
            val_max = statsmax + dymax,
            num_val_samples = 100,
            marker_size = cumm_dists_ms,
            pred_discard = pred_discard,
            parameter_label = focus_p_label,
            xlabel = "x",
            pred_marker = None,
            truth_marker = None,
            truth_linestyle = "--",
            pred_linestyle = "-",
            linewidth = 1.5 * cumulative_probs_linewidth,
            ax = axs2[0, 1],
            remove_yticks = remove_yticks,
            remove_xticks = False,
            legend_loc = legend_locs[libi],
            legend_bbox = legend_bbox2,
            make_ax_legend = False,
            add_text = None,
            alpha = 1.5 * alpha
            )
    
    fig_legend = fig.legend(loc = "outside lower center", frameon = False, ncols = 7)#, loc = "outside right", frameon = False, ncols = 1)
    for j in [2, 3]:
        fig_legend.legendHandles[j]._sizes = [30]
    for j in [0, 1, 4, 5, 6]:
        fig_legend.legendHandles[j].set_linestyle("none")
        fig_legend.legendHandles[j].set_marker("^")    
        
    fig.patch.set_alpha(0)
    fig2.patch.set_alpha(0)

with mpl.rc_context({"font.size" : font_size + 5}):
    for extra_seeds in [(0, 4)]:
        extra_seed1 = extra_seeds[0]
        extra_seed2 = extra_seeds[1]
        focus_library1b = rch.Library(
            data = None,
            parameters = list(np.array(focus_rs1)[:, None]),
            parameter_labels = [parameter_name1],
            data_generator = orbit_generator1,
            generator_args = val_generator_args1,
            seed = val_seed**2 + extra_seed1**2,
            standardize = standardize,
            standardizer = train_library1.standardizer
            )
        focus_library1b.generate_data()
        focus_library2b = rch.Library(
            data = None,
            parameters = list(np.array(focus_rs2)[:, None]),
            parameter_labels = [parameter_name2],
            data_generator = orbit_generator2,
            generator_args = val_generator_args2,
            seed = val_seed**2 + extra_seed2**2,
            standardize = standardize or shift_map2,
            standardizer = train_library2.standardizer
            )
        focus_library2b.generate_data()
        
        for libi, library in enumerate([val_library1, val_library2]):
            
            focus_libraryb = [focus_library1b, focus_library2b][libi]
        
            focus_predictionsb = rch.Async_SM_Predict(
                mapper_train_result = mapper_train_result,
                pred_esn_args = pred_esn_args,
                mapper_esn_args = map_esn_args,
                test_signals = focus_libraryb.data,
                test_length = test_length,
                incl_ri = True,
                incl_rf = False,
                predict_length = None,
                pred_feature = pred_feature,
                mapper_feature = mapper_feature,
                reduce_predictions = reduce_predictions,
                rmse_only = rmse_only,
                )
            
            for fi, r in enumerate(focus_libraryb.parameters[:-1]):
                
                max_horizon = 50 #100
            
                fig_fi, axs_fi = plt.subplots(
                    1, 1,
                    #figsize = (12, 3 * len(return_dims)),
                    figsize = (15, 3), #(10, 4), #(10, 2 * len(return_dims)),
                    sharex = True, constrained_layout = True
                    )
                if isinstance(axs_fi, mpl.axes._axes.Axes):
                    axs_fi = [axs_fi]
                for xi, fx in enumerate(axs_fi):
                    fx.plot(
                        np.arange(focus_predictionsb[fi].resync_inputs.shape[0]),
                        focus_predictionsb[fi].resync_inputs[:, xi],
                        color = "k",
                        )
                    if xi == 0:
                        label = "Truth"
                    else:
                        label = None
                    fx.plot(
                        np.arange(focus_predictionsb[fi].resync_inputs.shape[0] - 1,
                                  focus_predictionsb[fi].resync_inputs.shape[0] + focus_predictionsb[fi].target_outputs.shape[0]), # + 1),
                        np.concatenate((focus_predictionsb[fi].resync_inputs[-1, xi].reshape((1)),
                                        focus_predictionsb[fi].target_outputs[:, xi])),
                        color = "k",
                        label = label
                        )
                    if xi == 0:
                        label = "Prediction"
                    else:
                        label = None
                    fx.plot(
                        np.arange(focus_predictionsb[fi].resync_inputs.shape[0],
                                  focus_predictionsb[fi].resync_inputs.shape[0] + focus_predictionsb[fi].reservoir_outputs.shape[0]), # + 1),
                        focus_predictionsb[fi].reservoir_outputs[:, xi],
                        color = "tab:red",
                        label = label,
                        linestyle = "dotted"
                        )
                    if xi == 0:
                        label = "Loop Closed"
                    else:
                        label = None
                    fx.axvline(x = focus_predictionsb[fi].resync_inputs.shape[0] - 1,
                               linestyle = "--", color = "k",
                               label = label)
                    fx.set_ylabel("$x_n$")
                    if focus_predictionsb[fi].reservoir_outputs.shape[0] > max_horizon:
                        fx.set_xlim(right = max_horizon, left = -1)
                fig_fi.legend(loc = "outside upper center",
                           ncols = 4,
                           frameon = frame_legend)
                axs_fi[-1].set_xlabel("Iteration, $n$")
                fig_fi.patch.set_alpha(0)