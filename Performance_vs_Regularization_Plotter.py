# Plot data for Fig. S6

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import logging
from typing import Union

join = os.path.join

run_label = "performance_vs_regularization"

save_figs = False
show_figs = True
alpha = .3
vlength_threshold = 1.
all_methods = True
incl_errorbars = True
colormap = 'RdBu'
font_size = 15.
avg_type = "mean"
twofigsize = (16, 7)
hmap_method = "Unsync_SMW"#"_riW"

two_leg_below = True
frame_legend = False
ncols_two_leg = 3

lyap_units = False
side_leg = (1.175, .5)
ncols_leg = 3
side_leg = (.5, 1.05)
side_leg = (.5, -.2)
frame_legend = False

h = .01
Lyap_Exp = 0.9056
Lyap_Time = 1./Lyap_Exp

linestyles = {
    "Unsync_SM" : "-",
    "Unsync_SMW" : "-",
    "Unsync_SM_riW" : "--",
    "Unsync_sM_rfW" : "dotted",
    "batch" : "-",
    "vanilla" : "-"
    }
labels = {
    "Unsync_SM" : "METAFORS, No Cold-start",
    "Unsync_SMW" : "METAFORS", #"METAFORS, No Cold-start",
    #"Unsync_SM_riW" : "Signal Mapper, $(\\boldsymbol{r}(-t_{test}),W_{out})$",
    "Unsync_SM_riW" : "METAFORS",
    "library_interpolation" : "Interpolated Forecaster (Typically Infeasible)",
    "nearest_euclidean" : "Nearest Library Forecaster (Typically Infeasible)",
    "library_average" : "Average of Lib Members",
    "vanilla" : "Train on Test Signal",
    "long_vanilla" : "Train on a Long Signal w/ the Test Dynamics",
    "batch" : "Multi-task Learning"
    }
colors = {
    "Unsync_SMW" : "tab:red", #"tab:blue",
    "Unsync_SM" : "tab:blue",
    "Unsync_SM_riW" : "tab:red", #"black",
    "library_interpolation" : "tab:purple",
    "nearest_euclidean" : "tab:gray",
    "library_average" : "tab:gray",
    "vanilla" : "tab:green",
    "long_vanilla" : "black",
    "batch" : "tab:orange",
    "One_Step_SM_riW" : "green",
    "Unsync_SM_ri" : "black"
    }


save_loc = join(join(os.getcwd(), "Prediction_Plots"), run_label)

run_directory = join(os.getcwd(), "Prediction_Data")
run_directory = join(run_directory, run_label)
if all_methods:
    methods = os.listdir(run_directory)

seeds = {method : sorted([int(seed)
                for seed in os.listdir(join(run_directory, method))])
                for method in methods}
seed_strings = {method : [str(seed) for seed in seeds[method]]
                for method in methods}
pred_regs = {method : {int(seed) : sorted([float(reg)
    for reg in os.listdir(join(join(run_directory, method), seed))])
    for seed in seed_strings[method]} for method in methods}
pred_reg_strings = {method : {seed : [str(reg)
    for reg in pred_regs[method][int(seed)]]
    for seed in seed_strings[method]} for method in methods}
mapper_regs = {method : {int(seed) : {float(pred_reg) : sorted([float(mapper_reg.replace('.pickle', ''))
    for mapper_reg in os.listdir(join(join(join(run_directory, method), seed), pred_reg))])
    for pred_reg in pred_reg_strings[method][seed]}
    for seed in seed_strings[method]}
    for method in methods}
mapper_reg_strings = {method : {seed : {pred_reg : [str(mapper_reg)
    for mapper_reg in mapper_regs[method][int(seed)][float(pred_reg)]]
    for pred_reg in pred_reg_strings[method][seed]}
    for seed in seed_strings[method]}
    for method in methods}

pred_only_methods = []
pred_only_seeds = {}
dual_reg_methods = []
dual_reg_seeds = {}
for method in methods:
    sample_seed = list(mapper_regs[method].keys())[0]
    sample_pred_reg = list(mapper_regs[method][sample_seed].keys())[0]
    if np.isnan(mapper_regs[method][sample_seed][sample_pred_reg][0]):
        pred_only_methods.append(method)
        pred_only_seeds[method] = sample_seed
    else:
        dual_reg_methods.append(method)
        dual_reg_seeds[method] = sample_seed

def get_valid_length(
    errors: np.ndarray,
    threshold:  float = 1.0
    ) -> int:
    
    valid_length = 0
    for error in errors:
        if error <= threshold:
            valid_length += 1
        else:
            break
           
    if valid_length == errors.shape[0]:
        msg = "Root-square-error does not exceed " \
         f" {threshold}; true valid_length may be " \
          "greater than reported."
        logging.warning(msg)

    return valid_length

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
        medians = np.median(data)
        quartiles = np.quantile(data, [.25, .75])
    elif avg_type == "mean":
        medians = np.mean(data)
        quartiles = np.array([np.std(data), np.std(data)]) / np.sqrt(data.shape[0])
    
    return medians, quartiles
        
print("Have Median Predictions")
valid_times = {method : {seed : {pred_reg : {mapper_reg : None
    for mapper_reg in mapper_regs[method][seed][pred_reg]}
    for pred_reg in mapper_regs[method][seed]}
    for seed in mapper_regs[method]}
    for method in methods}

for method in methods:
    read_dir = join(run_directory, method)
    for seed in mapper_regs[method]:
        read_dir = join(read_dir, str(seed))
        for pred_reg in mapper_regs[method][seed]:
            pred_dir = join(read_dir, str(pred_reg))
            for mapper_reg in mapper_regs[method][seed][pred_reg]:
                file = join(pred_dir, str(mapper_reg) + ".pickle")
                with open(file, "rb") as tmp_file:
                    predictions = pickle.load(tmp_file).predictions[method]
                tvalids = [get_valid_length(prediction, vlength_threshold)
                           for prediction in predictions]
                valid_times[method][seed][pred_reg][mapper_reg] = get_stats_1D(tvalids)
   
with mpl.rc_context({'font.size': font_size}):
    
    figure, ax = plt.subplots(1, 2, figsize = twofigsize, constrained_layout = True)
        
    sample_seed = list(mapper_regs[hmap_method].keys())[0]
    pred_regs = list(mapper_regs[hmap_method][sample_seed])
    map_regs = list(mapper_regs[hmap_method][sample_seed][pred_regs[0]])
    
    avgs = np.zeros((len(pred_regs), len(map_regs)))
    
    x, y = np.meshgrid(
        np.array(pred_regs),
        np.array(map_regs)
        )
    
    for index, pred_reg in enumerate(pred_regs):
        avgs[index, :] = np.array(
            [valid_times[hmap_method][sample_seed][pred_reg][mapper_reg][0]
             for mapper_reg in valid_times[hmap_method][sample_seed][pred_reg]]
            )
        
    for index, mapper_reg in enumerate(map_regs):
        if np.max(avgs) in [valid_times[hmap_method][sample_seed][pred_reg][mapper_reg][0]
                            for pred_reg in pred_regs]:
            best_sm_reg = mapper_reg
            
    print("Max: ", np.max(avgs))
    print("Min: ", np.min(avgs))
    
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].set_xlim(min(pred_regs), max(pred_regs))
    ax[1].set_ylim(min(map_regs), max(map_regs))
    mesh_args = {"shading" : "nearest", "cmap" : colormap}
        
    colormesh = ax[1].pcolormesh(x, y, avgs.T, **mesh_args)
    colorbar = figure.colorbar(colormesh, ax = ax[1])

    if lyap_units:
        avgs *= h * Lyap_Exp
    
    if lyap_units:
        if avg_type == "median":
            colorbar.set_label("Median Valid Time, $T_{valid}$ (Lyapunov Times, $\\tau_{Lyap}$)")
        elif avg_type == "mean":
            colorbar.set_label("Mean Valid Time, $T_{valid}$ (Lyapunov Times, $\\tau_{Lyap}$)")
    else:
        if avg_type == "median":
            colorbar.set_label("Median Valid Time, $T_{valid}$ (Time Steps, $\\Delta t$)")
        elif avg_type == "mean":
            colorbar.set_label("Mean Valid Time, $T_{valid}$ (Time Steps, $\\Delta t$)")
    
    ax[1].set_ylabel("Signal Mapper RC Regularization, $\\alpha_{SM}$")
    ax[1].set_xlabel("Forecaster RC Regularization, $\\alpha_F$")
    
    for index, method in enumerate(pred_only_methods):
        seed = pred_only_seeds[method]
        medians = np.array([valid_times[method][seed][pred_reg][list(valid_times[method][seed][pred_reg].keys())[0]][0]
                                for pred_reg in valid_times[method][seed].keys()])
        iqrs = np.array([valid_times[method][seed][pred_reg][list(valid_times[method][seed][pred_reg].keys())[0]][1]
                              for pred_reg in valid_times[method][seed].keys()])
        regs = np.array(list(valid_times[method][seed].keys()), dtype = np.float64)
            
        if lyap_units:
            medians *= h * Lyap_Exp
            iqrs *= h * Lyap_Exp
    
        yerrs = iqrs.T
        if incl_errorbars:
            ax[0].errorbar(regs, medians, yerr = yerrs,
                           label = labels[method], linestyle = "--", marker = "o",
                           capsize = 5, capthick = 2, fillstyle = "none",
                           color = colors[method])
        else:
            ax[0].plot(regs, medians,
                       label = labels[method], linestyle = "--", marker = "o",
                       color = colors[method])
            
    seed = dual_reg_seeds[hmap_method]
    medians = np.array([valid_times[hmap_method][seed][pred_reg][best_sm_reg][0]
                        for pred_reg in valid_times[hmap_method][seed].keys()])
    iqrs = np.array([valid_times[hmap_method][seed][pred_reg][best_sm_reg][1]
                     for pred_reg in valid_times[hmap_method][seed].keys()])
    regs = np.array(list(valid_times[hmap_method][seed].keys()), dtype = np.float64)
        
    if lyap_units:
        medians *= h * Lyap_Exp
        iqrs *= h * Lyap_Exp

    yerrs = iqrs.T
    if incl_errorbars:
        ax[0].errorbar(regs, medians, yerr = yerrs,
                       label = labels[hmap_method], linestyle = "--", marker = "o",
                       capsize = 5, capthick = 2, fillstyle = "none",
                       color = colors[hmap_method])
    else:
        ax[0].plot(regs, medians,
                   label = labels[hmap_method], linestyle = "--", marker = "o",
                   color = colors[hmap_method])  
        
    ax[0].set_xscale("log")
    ax[0].set_ylim(0)
    if lyap_units:
        if avg_type == "median":
            ax[0].set_ylabel("Median Valid Time, $T_{valid}$ (Lyapunov Times, $\\tau_{Lyap}$)")
        elif avg_type == "mean":
            ax[0].set_ylabel("Mean Valid Time, $T_{valid}$ (Lyapunov Times, $\\tau_{Lyap}$)")
    else:
        if avg_type == "median":
            ax[0].set_ylabel("Median Valid Time, $T_{valid}$ (Time Steps, $\\Delta t$)")
        elif avg_type == "mean":
            ax[0].set_ylabel("Mean Valid Time, $T_{valid}$ (Time Steps, $\\Delta t$)")
    
    ax[0].set_xlabel("Forecaster RC Regularization, $\\alpha_F$")

    leg = figure.legend(loc = "outside lower center", ncol = ncols_two_leg, frameon = frame_legend)
        
    if save_figs:
        if not os.path.isdir(save_loc):
            os.makedirs(save_loc)
        file = os.path.join(save_loc, "Valid_Time_vs_Test_Length.png")
        figure.savefig(file)
        if not show_figs: plt.close(figure)
        print("Saved")