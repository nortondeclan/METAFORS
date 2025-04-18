import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as fx
import os
import pickle
from typing import Union
import test_systems as tst

def join(directory, name):
    return os.path.join(directory, name)

measure_autonomous_onestep_error = True
measure_valid_length = True

if measure_autonomous_onestep_error:
    run_label = "paper_param_climate_fm_nonorm"
    metric_name = "Autonomous One-step Error, $\\epsilon$"
    log_colors = True
elif measure_valid_length:
    run_label = "paper_param_climate_fv"
    metric_name = "Valid Length, $T_{valid}$"
    log_colors = False

methods = [
    'nearest_euclidean',
    'library_interpolation',
    'async_sm_ri',
    'multitask'
    ]

figsize = (10.4, 8.6)
grid_plots = False
row_plots = True

if grid_plots:
    figsize = (10.4, 8.6)
elif row_plots:
    figsize = (20, 5)
    
ex = tst.get_lorenz()
print("Persistance Map Error (Normalization Factor) for Standard Lorenz: ",
      np.mean(np.sqrt(np.sum(np.square(ex[1:]-ex[:-1]), axis = 1))))
del ex

save_figs = False
show_figs = True
alpha = .3
vlength_threshold = 1.
all_methods = False
incl_errorbars = True
colormap = 'RdBu_r'
font_size = 15.
avg_type = 'mean'
avg_type = "median"

outline_text = True
outline_color = "white"
text_color = "black"
text_weight = None
outline_width = 3
use_titles = False
lib_length_varied = False
test_length_varied = False
lib_size_varied = True
library_name = "Lorenz_Lib"

nonstationary_tests = False
test_from_lib = False
lyap_units = False
legend_to_side = False
best_legend = True
side_leg = (.5, -.2)
title_x0, title_y0 = .001, .8
title_x1, title_y1 = .001, .8
ncols_leg = 2
frame_legend = False

auto_color_limits = False
fixed_color_limits = True
v_min, v_max = 0, 1.5

h = .01
Lyap_Exp = 0.9056
Lyap_Time = 1./Lyap_Exp

linestyles = {
    "Unsync_SM" : "-",
    "Unsync_SM_riW" : "--",
    "Unsync_sM_rfW" : "dotted",
    "multitask" : "-",
    "vanilla" : "-",
    }
labels = {
    "Unsync_SM" : "METAFORS, $W_{out}$-only",
    "Unsync_SM_riW" : "(b) METAFORS",
    "async_sm_ri" : "(b) METAFORS",
    "library_interpolation" : "(d) Interpolation\n     (Typically Infeasible)",
    "nearest_euclidean" : "(a) Nearest Library Forecaster\n     (Typically Infeasible)",
    "vanilla" : "Train on Test Signal Directly",
    "multitask" : "(c) Multi-task Learning",
    }
colors = {
    "Unsync_SM" : "tab:red",
    "Unsync_SM_riW" : "black",
    "library_interpolation" : "tab:blue",
    "nearest_euclidean" : "tab:purple",
    "vanilla" : "tab:green",
    "multitask" : "tab:orange",
    }
title_positions = {
    "Unsync_SM_riW" : (.001, .91),
    "async_sm_ri" : (.001, .91),
    "library_interpolation" : (.001, .84),
    "nearest_euclidean" : (.001, .84),
    "multitask" : (.001, .91)
    }

save_loc = join(join(os.getcwd(), "Prediction_Plots"), run_label)

run_directory = join(os.getcwd(), "Prediction_Data")
run_directory = join(run_directory, run_label)
if all_methods:
    methods = os.listdir(run_directory)
    
if "libraries" in methods:
    methods.remove("libraries")

seeds = {method : sorted([int(seed)
                for seed in os.listdir(join(run_directory, method))])
                for method in methods}
seed_strings = {method : [str(seed) for seed in seeds[method]]
                for method in methods}
val_seeds = {method : {int(seed) : sorted([int(val_seed.replace(".pickle", ""))
    for val_seed in os.listdir(join(join(run_directory, method), seed))])
    for seed in seed_strings[method]} for method in methods}
val_seed_strings = {method : {seed : [str(val_seed)
    for val_seed in val_seeds[method][int(seed)]]
    for seed in seed_strings[method]} for method in methods}

libraries_directory = os.path.join(run_directory, "libraries")
with open(os.path.join(libraries_directory, library_name + ".pickle"), "rb") as tmp_file:
    train_library = pickle.load(tmp_file)

pred_only_methods = []
sample_seeds = {}
for method in methods:
    sample_seed = list(val_seeds[method].keys())[0]
    sample_seeds[method] = sample_seed

def get_stats_1D(
        data : Union[list, np.ndarray, dict]
        ):
    
    if isinstance(data, dict):
        keys = list(data.keys())
        array_keys = [key for key in keys if isinstance(data[key], np.ndarray)]
        
        if len(array_keys) == 0:
            return np.nan, np.nan
        
        else:
            array = np.zeros((len(keys), data[array_keys[0]].shape[0]))
            for ind, key in enumerate(keys):
                array[ind] = data[key]
            data = array
    
    elif isinstance(data, list):
        data = np.array(data)
        
    if avg_type == "median":
        medians =  np.median(data)
        quartiles = np.quantile(data, [.25, .75])
    elif avg_type == "mean":
        medians = np.mean(data)
        quartiles = np.std(data)
    
    return medians, quartiles
      
print("Have Median Predictions")
valid_times = {method : {seed : {val_seed : None
    for val_seed in val_seeds[method][seed]}
    for seed in val_seeds[method]}
    for method in methods}
val_params = {method : {seed : None
    for seed in val_seeds[method]}
    for method in methods}

medians = {}  
for method in methods:
    read_dir = join(run_directory, method)
    for seed in val_seeds[method]:
        read_dir = join(read_dir, str(seed))
        for val_ind, val_seed in enumerate(val_seeds[method][seed]):
            
            pred_dir = join(read_dir, str(val_seed) + ".pickle")
            with open(pred_dir, "rb") as tmp_file:
                predictions = pickle.load(tmp_file).predictions[method]
            if "w" in run_label or "v" in run_label:
                tvalids = np.array(predictions)
            else:
                tvalids = np.array(predictions)[:, 0]
            valid_times[method][seed][val_seed] = tvalids
            
            if val_ind == 0:
                lib_file = os.path.join(libraries_directory,
                                        str(val_seed) + ".pickle")
                with open(lib_file, "rb") as tmp_file:
                    val_lib = pickle.load(tmp_file)
                val_params[method][seed] = [list(tup) for tup in val_lib.parameters]
    
    num_test_points = len(val_params[method][seed])
    medians[method] = np.array([
        get_stats_1D(
            [valid_times[method][seed][val_seed][test_ind]
             for val_seed in list(valid_times[method][seed].keys())]
             )[0]
            for test_ind, _ in enumerate(val_params[method][seed])]
        ).reshape((int(np.sqrt(num_test_points)), int(np.sqrt(num_test_points))))

with mpl.rc_context({'font.size': font_size}):  
    figure, _ = plt.subplots(figsize = figsize, layout = "compressed",
                             sharex = True, sharey = True)
    lib_markers = np.array([list(tup) for tup in train_library.parameters])
    axs = []
    colormeshes = [None] * len(methods)
    for method_ind, method in enumerate(methods):
        if len(methods)%2 == 1: 
            if grid_plots:
                if (2*method_ind)//len(methods) == 1:
                    ax = plt.subplot2grid(shape = (2, len(methods)), colspan = 2, fig = figure,
                        loc = ((2*method_ind)//len(methods), (2*method_ind)%len(methods)+1))
                else:
                    ax = plt.subplot2grid(shape = (2, len(methods)), colspan = 2, fig = figure,
                        loc = ((2*method_ind)//len(methods), (2*method_ind)%len(methods)))
                if (2 * method_ind) // 6 == 1:
                    ax.set_xlabel("Time-scale, $\\omega_t$")
                else:
                    ax.set_xticks([])
                ax.set_ylabel("Lorenz Variable, $v_1$")
            elif row_plots:
                ax = plt.subplot2grid(shape = (1, len(methods)), fig = figure, loc = ((0, method_ind)))
                ax.set_xlabel("Time-scale, $\\omega_t$")
                ax.set_ylabel("Lorenz Variable, $v_1$")
                
        else:
            if grid_plots:
                ax = plt.subplot2grid(shape = (2, len(methods)), colspan = 2, fig = figure,
                    loc = ((2*method_ind)//len(methods), (2*method_ind)%len(methods)))
                ax.set_aspect(.1)
                if (2 * method_ind) % len(methods) == 0:
                    ax.set_ylabel("Lorenz Variable, $v_1$")
                    if not use_titles:
                        text = ax.set_title(labels[method], x = title_positions[method][0],
                                            y = title_positions[method][1], loc = "left",
                                            color = text_color)
                        if outline_text:
                            text.set_path_effects([
                                fx.Stroke(linewidth = 2, foreground = outline_color), fx.Normal()])
                else:
                    ax.set_yticks([])
                    if not use_titles:
                        text = ax.set_title(labels[method], x = title_positions[method][0],
                                            y = title_positions[method][1], loc = "left",
                                            color = text_color)
                        if outline_text:
                            text.set_path_effects([
                                fx.Stroke(linewidth = 2, foreground = outline_color), fx.Normal()])
                if (2*method_ind)//len(methods) == 0:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel("Time-scale, $\\omega_t$")
            elif row_plots:
                ax = plt.subplot2grid(shape = (1, len(methods)), fig = figure, loc = ((0, method_ind)))
                ax.set_aspect(.1)
                ax.set_xlabel("Time-scale, $\\omega_t$")
                ax.set_ylabel("Lorenz Variable, $v_1$")
        
        print("Max: ", np.max(medians[method]))
        print("Min: ", np.min(medians[method]))
        print(method + ", Median: ", np.median(medians[method]))
        print(method + ", Mean: ", np.mean(medians[method]))

        vmin, vmax = 0, np.max(np.array([medians[method] for method in methods]))
        
        mesh_args = {"shading" : "nearest", "cmap" : colormap}
        if "v" in run_label:
            mesh_args["cmap"] = colormap.replace("_r", "")
        if not auto_color_limits and not log_colors:
            mesh_args["vmin"] = vmin
            mesh_args["vmax"] = vmax
        elif log_colors and not "v" in run_label:
            mesh_args["norm"] = mpl.colors.LogNorm(
                vmin = np.array([medians[method] for method in methods]).min(),
                vmax = np.array([medians[method] for method in methods]).max(),
                )
        
        x = np.array(val_params[method][list(val_params[method].keys())[0]])[:,1]
        _, idx = np.unique(x, return_index = True)
        x = x[np.sort(idx)]
        y = np.array(val_params[method][list(val_params[method].keys())[0]])[:,0]
        _, idy = np.unique(y, return_index = True)
        y = y[np.sort(idy)]
        
        colormeshes[method_ind] = ax.pcolormesh(x, y, medians[method], **mesh_args)
        
        if lyap_units:
            medians *= h * Lyap_Exp
        
        if use_titles:
            ax.set_title(labels[method])
        
        ax.plot(lib_markers[:, 1], lib_markers[:, 0],
                color = "k", linestyle = "none", marker = "o")
        axs.append(ax)
    
    colorbar = figure.colorbar(colormeshes[-1], ax = axs[:])
    if avg_type == "median":
        colorbar.set_label("Median " + metric_name)
    elif avg_type == "mean":
        colorbar.set_label("Mean " + metric_name)
    
    figure.patch.set_alpha(0)
    
    if save_figs:
        if not os.path.isdir(save_loc):
            os.makedirs(save_loc)
        file = os.path.join(save_loc, method + "_Valid_Time_vs_Test_Params_HeatMap.png")
        figure.savefig(file)
        if not show_figs: plt.close(figure)
        print("Saved")