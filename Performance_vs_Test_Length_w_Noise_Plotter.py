import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import logging
from typing import Union

join = os.path.join

run_labels = [
    "performance_vs_test_length_noiseless_train",
    "performance_vs_test_length_noisy_train"
    ]

save_figs = False
show_figs = True
vlength_threshold = 1.
all_methods = True
avg_type = "mean"

lyap_units = False
font_size = 15.0
colormap = "RdBu"

h = .01
Lyap_Exp = 0.9056
Lyap_Time = 1./Lyap_Exp
plot_noises = [1e-3, 1e-2, 1e-1, 2e-1]

colors = {
    "Unsync_SM" : "black",
    "Unsync_SM_riW" : "black",
    "Unsync_SM_rfW" : "black",
    "batch" : "orange",
    "vanilla" : "purple",
    "Sync_SM" : "green",
    "Sync_SM_riW" : "limegreen",
    "Sync_SM_rfW" : "green",
    "Sync_SM_Const_Extrap_riW" : "magenta",
    "Sync_SM_State_Match_riW" : "pink",
    "Sync_SM_State_Match" : "cyan",
    "Sync_SM_Const_Extrap" : "grey"
    }
linestyles = {
    "Unsync_SM" : "-",
    "Unsync_SM_riW" : "--",
    "Unsync_sM_rfW" : "dotted",
    "batch" : "-",
    "vanilla" : "-",
    "Sync_SM" : "-",
    "Sync_SM_riW" : "--",
    "Sync_SM_rfW" : "dotted",
    "Sync_SM_Const_Extrap_riW" : "dotted",
    "Sync_SM_State_Match_riW" : "dashdot",
    "Sync_SM_State_Match" : "-",
    "Sync_SM_Const_Extrap" : "-"
    }
labels = {
    "Unsync_SM" : "Async SM $W$-only",
    "Unsync_SM_riW" : "Async SM $(r_{-t_{test}}, W)$",
    "Unsync_SM_rfW" : "Async SM $(r_{0}, W)$",
    "batch" : "Batch",
    "vanilla" : "Vanilla",
    "Sync_SM" : "Sync SM $W$-only",
    "Sync_SM_riW" : "Sync SM $(r_{-t_{test}}, W)$",
    "Sync_SM_rfW" : "Sync SM $(r_{0}, W)$",
    "Sync_SM_Const_Extrap_riW" : "Sync SM $(r_{-t_{test}}, W)$, Const. Extrap.",
    "Sync_SM_State_Match_riW" : "Sync SM $(r_{-t_{test}}, W)$, State Match",
    "Sync_SM_State_Match" : "Sync SM $W$-only, State Match",
    "Sync_SM_Const_Extrap" : "Sync SM $W$-only, Const. Extrap."
    }
labels = {str(noise) : "$\\sigma_{Noise}=$" + str(noise) + "$\\sigma_{Lib}$"
          for noise in plot_noises}
linestyles = ["dotted", "--"]
markers = ["o", "o"]
colors = {
    1e-4 : "tab:red",
    1e-3 : "tab:red",
    1e-2 : "tab:blue",
    1e-1 : "tab:orange",
    2e-1 : "tab:purple"
    }
run_plot_labels = ["Trained without Noise", "Trained with Noise"]

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
        quartiles = np.std(data)/np.sqrt(len(data))
    
    return medians, quartiles

medians_max = 0
medians_dict = {}
figure, ax = plt.subplots(1, 2, figsize = (14,6), sharey = True,
                          constrained_layout = True)
for run_ind, run_label in enumerate(run_labels):

    save_loc = join(join(os.getcwd(), "Prediction_Plots"), run_label)
    
    run_directory = join(os.getcwd(), "Prediction_Data")
    run_directory = join(run_directory, run_label)
    if all_methods:
        methods = os.listdir(run_directory)
    
    sample_seps = [float(method.replace("_riW", "")) for method in methods]
    methods = [str(method) + "_riW" for method in sorted(sample_seps)]
    
    seeds = {method : sorted([int(seed)
                    for seed in os.listdir(join(run_directory, method))])
                    for method in methods}
    seed_strings = {method : [str(seed) for seed in seeds[method]]
                    for method in methods}
    test_lengths = {method : {int(seed) : sorted([int(length.replace('.pickle', ''))
        for length in os.listdir(join(join(run_directory, method), seed))])
        for seed in seed_strings[method]} for method in methods}
    test_length_strings = {method : {seed : [str(length)
        for length in test_lengths[method][int(seed)]]
        for seed in seed_strings[method]} for method in methods}
    
    all_seeds = list(set([seed for method in methods for seed in seeds[method]]))
    all_lengths = sorted(list(set([test_length for method in methods
                                for seed in seeds[method]
                                for test_length in test_lengths[method][seed]])))
    num_seeds = len(all_seeds)
    num_lengths = len(all_lengths)
            
    print("Have Median Predictions")
    valid_times = {method : {seed : {test_length : None
        for test_length in test_lengths[method][seed]}
        for seed in seeds[method]}
        for method in methods}
        
    for method in methods:
        read_dir = join(run_directory, method)
        for seed in seeds[method]:
            read_dir = join(read_dir, str(seed))
            for test_length in test_lengths[method][seed]:
                file = join(read_dir, str(test_length) + ".pickle")
                dict_method = method
                with open(file, "rb") as tmp_file:
                    predictions = pickle.load(tmp_file).predictions[dict_method]
                tvalids = [get_valid_length(prediction, vlength_threshold)
                               for prediction in predictions]
                valid_times[method][seed][test_length] = get_stats_1D(tvalids)
    
    seed = all_seeds[0]    
    
    medians = np.zeros((len(methods), len(all_lengths)))
    x, y = np.meshgrid(
        np.array(all_lengths),
        np.array([float(method.replace("_riW", "")) for method in methods])
        )
        
    for index, method in enumerate(methods):
        medians[index, :] = np.array([valid_times[method][seed][test_length][0]
                                      for test_length in valid_times[method][seed].keys()])
        
    medians_dict[run_label] = medians
    medians_max = max(np.max(medians), medians_max)
    print("Label: ", run_label)
    print("Max: ", np.max(medians))
    print("Min: ", np.min(medians))

    with mpl.rc_context({'font.size': font_size}):   
        
        for index, method in enumerate(methods):
            medians[index, :] = np.array([valid_times[method][seed][test_length][0]
                                          for test_length in valid_times[method][seed].keys()])
        
        vmin, vmax = 0, np.max(medians)
        
        ax[run_ind].set_yscale("log")
        ax[run_ind].set_ylim(np.min([float(method.replace("_riW", "")) for method in methods]),
                             np.max([float(method.replace("_riW", "")) for method in methods]))            
        
        colormesh = ax[run_ind].pcolormesh(x, y, medians, shading = 'nearest', 
                      cmap=colormap, vmin = vmin, vmax = vmax)
        
        if run_ind == len(run_labels) - 1:
            colorbar = figure.colorbar(colormesh, ax = ax)
    
        if lyap_units:
            medians *= h * Lyap_Exp
            
        ax[run_ind].tick_params(axis = 'both', which = 'major', labelsize = font_size)
        ax[run_ind].tick_params(axis = 'both', which = 'minor', labelsize = font_size)    
        if lyap_units:
            ax[run_ind].set_xlabel("Test Signal Length ($\\tau_{Lyap}$)",
                                   fontsize = font_size)
            if run_ind == 0:
                ax[run_ind].set_title("Trained without Noise")
                ax[run_ind].set_ylabel("Standard Deviation of Noise, $\\sigma_{Noise}$ ($\\sigma_{Lib}$)",
                                       fontsize = font_size)
            if run_ind == len(run_labels) - 1:
                ax[run_ind].set_title("Trained with Noise")
                if avg_type == "median":
                    colorbar.set_label("Median Valid Time ($\\tau_{Lyap}$)")
                elif avg_type == "mean":
                    colorbar.set_label("Mean Valid Time ($\\tau_{Lyap}$)")
        else:
            ax[run_ind].set_xlabel("Test Signal Length (Time Steps)",
                                   fontsize = font_size)
            if run_ind == 0:
                ax[run_ind].set_title("Trained without Noise")
                ax[run_ind].set_ylabel("Standard Deviation of Noise, $\\sigma_{Noise}$ ($\\sigma_{Lib}$)",
                                       fontsize = font_size)
            if run_ind == len(run_labels) - 1:
                ax[run_ind].set_title("Trained with Noise")
                if avg_type == "median":
                    colorbar.set_label("Median Valid Time (Time Steps)")
                elif avg_type == "mean":
                    colorbar.set_label("Mean Valid Time (Time Steps)")
            
        if save_figs:
            if not os.path.isdir(save_loc):
                os.makedirs(save_loc)
            file = os.path.join(save_loc, "NoiseAmp_TShort_TValid_Heatmap.png")
            figure.savefig(file)
            if not show_figs: plt.close(figure)
            print("Saved")
            

figure, ax = plt.subplots(
    1, 1, sharey = True, constrained_layout = True, figsize = (8, 6.5),
    )
for run_ind, run_label in enumerate(run_labels):

    save_loc = join(join(os.getcwd(), "Prediction_Plots"), run_label)
    
    run_directory = join(os.getcwd(), "Prediction_Data")
    run_directory = join(run_directory, run_label)
    if all_methods:
        methods = os.listdir(run_directory)
    methods = [str(method) + "_riW" for method in sorted(plot_noises)]
    
    seeds = {method : sorted([int(seed)
                    for seed in os.listdir(join(run_directory, method))])
                    for method in methods}
    seed_strings = {method : [str(seed) for seed in seeds[method]]
                    for method in methods}
    test_lengths = {method : {int(seed) : sorted([int(length.replace('.pickle', ''))
        for length in os.listdir(join(join(run_directory, method), seed))])
        for seed in seed_strings[method]} for method in methods}
    test_length_strings = {method : {seed : [str(length)
        for length in test_lengths[method][int(seed)]]
        for seed in seed_strings[method]} for method in methods}
    
    all_seeds = list(set([seed for method in methods for seed in seeds[method]]))
    all_lengths = sorted(list(set([test_length for method in methods
                                for seed in seeds[method]
                                for test_length in test_lengths[method][seed]])))
    num_seeds = len(all_seeds)
    num_lengths = len(all_lengths)
            
    print("Have Median Predictions")
    valid_times = {method : {seed : {test_length : None
        for test_length in test_lengths[method][seed]}
        for seed in seeds[method]}
        for method in methods}
        
    for method in methods:
        read_dir = join(run_directory, method)
        for seed in seeds[method]:
            read_dir = join(read_dir, str(seed))
            for test_length in test_lengths[method][seed]:
                file = join(read_dir, str(test_length) + ".pickle")
                dict_method = method
                with open(file, "rb") as tmp_file:
                    predictions = pickle.load(tmp_file).predictions[dict_method]
                tvalids = [get_valid_length(prediction, vlength_threshold)
                               for prediction in predictions]
                valid_times[method][seed][test_length] = get_stats_1D(tvalids)
    
    seed = all_seeds[0]    
    
    for method in methods:
        ax.errorbar(
            np.array(all_lengths),
            [valid_times[method][seed][tlength][0] for tlength in all_lengths],
            yerr = [valid_times[method][seed][tlength][1] for tlength in all_lengths],
            label = labels[method.replace("_riW", "")],
            linestyle = linestyles[run_ind],
            marker = markers[run_ind],
            capsize = 5, capthick = 2, fillstyle = "none",
            color = colors[float(method.replace("_riW", ""))]
            )

with mpl.rc_context({'font.size': font_size}):
    
    color_lines = []
    for noise in plot_noises:
        color_lines.append(ax.plot([0], [0], color = colors[noise],
                                   marker='None', linestyle='-',
                                   label = labels[str(noise)])[0])
    category_lines = []
    for i, _ in enumerate(run_labels):
        category_lines.append(ax.plot([0], [0], linestyle = linestyles[i],
                                      color = "k", marker='None',
                                      label = run_plot_labels[i])[0])
    
    ax.set_ylim(0)
    figure.legend(color_lines, [labels[str(noise)] for noise in plot_noises],
                  loc = "lower right",
                  ncol = len(color_lines)//2,
                  bbox_to_anchor = (1.015, .07),
                  frameon = False)
    figure.legend(category_lines, run_plot_labels,
                  loc = "outside upper center",
                  ncol = 2,
                  frameon = False)
    
    ax.tick_params(axis = 'both', which = 'major', labelsize = font_size)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = font_size)
    
    ax.set_xlabel("Test Signal Length (Time Steps)", fontsize = font_size)
    
    if avg_type == "median":
        ax.set_ylabel("Median Valid Time (Time Steps)", fontsize = font_size)
    elif avg_type == "mean":
        ax.set_ylabel("Mean Valid Time (Time Steps)", fontsize = font_size)