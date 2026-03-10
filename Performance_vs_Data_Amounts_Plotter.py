import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import logging
from typing import Union

join = os.path.join

fig_6cd_s4 = True #False
fig_s5 = False
fig_s3 = False

if fig_6cd_s4:
    run_labels = ["performance_vs_test_length_z_only",
                  "performance_vs_test_length_ri_noW_z_only"]
    separate_legends = True
    twofigsize = (18.5, 7)
elif fig_s5:
    run_labels = ["performance_vs_lib_size", 
                  "performance_vs_lib_length"]
    separate_legends = False
    twofigsize = (16.5, 7)
elif fig_s3:
    run_labels = ["performance_vs_test_length",
                  "performance_vs_test_length_ri_noW"]
    separate_legends = True
    twofigsize = (18.5, 7)

run_label = run_labels[1]

save_figs = False
show_figs = True
alpha = .3
vlength_threshold = 1.
all_methods = True
incl_errorbars = True
colormap = 'RdBu' 
font_size = 16.
x_cutoff = None
avg_type = "mean"

lyap_units = False
two_leg_below = True
ncols = [2, 2]
boxmins = [-.3, -.3]
ncols_leg = 2
frame_legend = False
ncols_two_leg = 4

methods_to_exclude = ["One_Step_SM_riW", "nearest_euclidean"]

h = .01
Lyap_Exp = 0.9056
Lyap_Time = 1./Lyap_Exp
labels = {
    "Unsync_SMW" : "METAFORS, Zero Start",
    "Unsync_SM" : "METAFORS, Zero Start",
    "Unsync_SM_riW" : "METAFORS",
    "library_interpolation" : "Interpolated Forecaster (Typically Infeasible)",
    "nearest_euclidean" : "Nearest Library Forecaster\n(Typically Infeasible)",
    "library_average" : "Average of Lib Members",
    "vanilla" : "Train on Test Signal",
    "long_vanilla" : "Train on Long Signal w/ Same Dynamics",
    "batch" : "Multi-task Learning",
    "One_Step_SM_riW" : "Signal Mapper without Internal Memory",
    "Unsync_SM_ri" : "Resynchronized using METAFORS, $r_{-t_{test}}$-only"
    }
colors = {
    "Unsync_SMW" : "tab:blue",
    "Unsync_SM" : "tab:blue",
    "Unsync_SM_riW" : "tab:red",
    "library_interpolation" : "tab:purple",
    "nearest_euclidean" : "tab:gray",
    "library_average" : "tab:gray",
    "vanilla" : "tab:green",
    "long_vanilla" : "tab:cyan",
    "batch" : "tab:orange",
    "One_Step_SM_riW" : "green",
    "Unsync_SM_ri" : "black"
    }
linestyles = {
    "Unsync_SMW" : "dashdot",
    "Unsync_SM" : "dashdot",
    "Unsync_SM_riW" : "--",
    "library_interpolation" : (0, (3, 1, 1, 1, 1, 1)),
    "nearest_euclidean" : "--",
    "library_average" : "--",
    "vanilla" : (0, (1, 1)),
    "long_vanilla" : "--",
    "batch" : (5, (10, 3)),
    "One_Step_SM_riW" : "--",
    "Unsync_SM_ri" : "--"
    }

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
        medians = np.median(data) #, axis = 0)
        ##quartiles = np.quantile(data, [.25, .75]) #, axis = 0)
        #quartiles = np.sqrt(np.pi/(2 * data.shape[0])) * np.array([.5 * np.std(data), .5 * np.std(data)])
        quartiles = np.sqrt(np.pi/(2 * data.shape[0])) * np.array([np.std(data), np.std(data)])
        
    elif avg_type == "mean":
        medians = np.mean(data)
        quartiles = np.array([np.std(data), np.std(data)]) / np.sqrt(data.shape[0])
    
    return medians, quartiles
        

save_loc = join(join(os.getcwd(), "Prediction_Plots"), run_label)

run_directory = join(os.getcwd(), "Prediction_Data")
run_directory = join(run_directory, run_label)
if all_methods:
    methods = os.listdir(run_directory)
for method in methods_to_exclude:
    if method in methods:
        methods.remove(method)

seeds = {method : sorted([int(seed)
                for seed in os.listdir(join(run_directory, method))])
                for method in methods}
seed_strings = {method : [str(seed) for seed in seeds[method]]
                for method in methods}
test_lens = {method : {int(seed) : sorted([int(length.replace(".pickle", ""))
    for length in os.listdir(join(join(run_directory, method), seed))])
    for seed in seed_strings[method]} for method in methods}
test_len_strings = {method : {seed : [str(length)
    for length in test_lens[method][int(seed)]]
    for seed in seed_strings[method]} for method in methods}

pred_only_methods = []
sample_seeds = {}
for method in methods:
    sample_seed = list(test_lens[method].keys())[0]
    sample_seeds[method] = sample_seed

print("Have Median Predictions")
valid_times = {method : {seed : {length : None
    for length in test_lens[method][seed]}
    for seed in test_lens[method]}
    for method in methods}

for method in methods:
    read_dir = join(run_directory, method)
    for seed in test_lens[method]:
        read_dir = join(read_dir, str(seed))
        for length in test_lens[method][seed]:
            pred_dir = join(read_dir, str(length) + ".pickle")
            with open(pred_dir, "rb") as tmp_file:
                predictions = pickle.load(tmp_file).predictions[method]
            tvalids = [get_valid_length(prediction, vlength_threshold)
                       for prediction in predictions]
            valid_times[method][seed][length] = get_stats_1D(tvalids)
                
best_legend = True
with mpl.rc_context({'font.size': font_size}):
    
    if "lib_length" in run_label:
        lib_length_varied = True
    else:
        lib_length_varied = False
    if "test_length" in run_label:
        test_length_varied = True
    else:
        test_length_varied = False
    if "lib_size" in run_label:
        lib_size_varied = True
    else:
        lib_size_varied = False
    if "SM_size" in run_label:
        SM_size_varied = True
    else:
        SM_size_varied = False
    if "pred_size" in run_label:
        pred_size_varied = True
    else:
        pred_size_varied = False
    
    if "noW" in run_label:
        labels["Unsync_SM_ri"] = "METAFORS, Cold-start Only"
        labels["long_vanilla"] = "Zero Start"
        labels["long_vanilla_state_matching"] = "Search Training Data"
        labels["long_vanilla_extrap_constant"] = "Back-extrapolate as a Constant"
        
        linestyles["long_vanilla_state_matching"] = "--"
        linestyles["long_vanilla_extrap_constant"] = "--"
        colors["long_vanilla_state_matching"] = "tab:olive"
        colors["long_vanilla_extrap_constant"] = "tab:pink"
    
    figure, ax = plt.subplots(1, 1, figsize = (8, 7), constrained_layout = True)
    for index, method in enumerate(methods):
        seed = sample_seeds[method]
        medians = np.array([valid_times[method][seed][length][0]
                            for length in valid_times[method][seed].keys()])
        iqrs = np.array([valid_times[method][seed][length][1]
                            for length in valid_times[method][seed].keys()])
        lengths = np.array(list(valid_times[method][seed].keys()), dtype = int)
            
        if lyap_units:
            medians *= h * Lyap_Exp
            iqrs *= h * Lyap_Exp
            if "length" in run_label and x_cutoff is not None:
                x_cutoff *= h * Lyap_Exp
                
        yerrs = iqrs.T
        if x_cutoff is not None:
            lengths = lengths[lengths <= x_cutoff]
        medians = medians[np.arange(len(lengths))]
        yerrs = yerrs[:, np.arange(len(lengths))]
        
        if incl_errorbars:
            plt.errorbar(lengths, medians, yerr = yerrs,
                 label = labels[method], linestyle = linestyles[method],
                 marker = "o", capsize = 5, capthick = 2, fillstyle = "none",
                 color = colors[method])
        else:
            plt.plot(lengths, medians,
                 label = labels[method], linestyle = linestyles[method],
                 marker = "o", color = colors[method])
    
    ax.set_ylim(0)
    if lyap_units:
        if avg_type == "median":
            ax.set_ylabel("Median Valid Time, $T_{valid}$ (Lyapunov Times, $\\tau_{Lyap}$)")
        elif avg_type == "mean":
            ax.set_ylabel("Mean Valid Time, $T_{valid}$ (Lyapunov Times, $\\tau_{Lyap}$)")
        if test_length_varied: ax.set_xlabel("Test Length, $t_{test}$ (Lyapunov Times, $\\tau_{Lyap}$)")
        elif lib_length_varied: ax.set_xlabel("Library Member Fitting Length, $N_{fit}$ (Lyapunov Times, $\\tau_{Lyap}$)")
        elif lib_size_varied: ax.set_xlabel("Number of Long Signals in the Library, $N_L$")
        elif SM_size_varied: ax.set_xlabel("Signal Mapper Size, $N_{SM}$ (Nodes)")
        elif pred_size_varied: ax.set_xlabel("Forecaster Size, $N_F$ (Nodes)")
    else:
        if avg_type == "median":
            ax.set_ylabel("Median Valid Time, $T_{valid}$ (Time Steps, $\\Delta t$)")
        elif avg_type == "mean":
            ax.set_ylabel("Mean Valid Time, $T_{valid}$ (Time Steps, $\\Delta t$)")
        if test_length_varied: ax.set_xlabel("Test Length, $N_{test}$")
        elif lib_length_varied: ax.set_xlabel("Library Member Fitting Length, $N_{fit}$")
        elif lib_size_varied: ax.set_xlabel("Number of Long Signals in the Library, $N_L$")
        elif SM_size_varied: ax.set_xlabel("Signal Mapper Size, $N_{SM}$ (Nodes)")
        elif pred_size_varied: ax.set_xlabel("Forecaster Size, $N_F$ (Nodes)")
        
    figure.legend(frameon = frame_legend, loc = "outside lower center",
                  ncol = ncols_leg)
    figure.patch.set_alpha(0)
        
    if save_figs:
        if not os.path.isdir(save_loc):
            os.makedirs(save_loc)
        file = os.path.join(save_loc, "Valid_Time_vs_Test_Length.png")
        figure.savefig(file)
        if not show_figs: plt.close(figure)
        print("Saved")
        
best_legend = False
already_labelled = []
with mpl.rc_context({'font.size': font_size}):
    if two_leg_below:
        figure, ax = plt.subplots(1, len(run_labels), figsize = twofigsize, constrained_layout = True)
    else:
        figure, ax = plt.subplots(1, len(run_labels), figsize = twofigsize, constrained_layout = True)
        
    for rli, run_label in enumerate(run_labels):
        
        if "lib_length" in run_label:
            lib_length_varied = True
        else:
            lib_length_varied = False
        if "test_length" in run_label:
            test_length_varied = True
        else:
            test_length_varied = False
        if "lib_size" in run_label:
            lib_size_varied = True
        else:
            lib_size_varied = False
        if "SM_size" in run_label:
            SM_size_varied = True
        else:
            SM_size_varied = False
        if "pred_size" in run_label:
            pred_size_varied = True
        else:
            pred_size_varied = False
        
        if "noW" in run_label:
            labels["Unsync_SM_ri"] = "METAFORS, Cold-start Only"
            labels["long_vanilla"] = "Zero Start"
            labels["long_vanilla_state_matching"] = "Search Training Data"
            labels["long_vanilla_extrap_constant"] = "Back-extrapolate as a Constant"
            
            linestyles["long_vanilla_state_matching"] = "--"
            linestyles["long_vanilla_extrap_constant"] = "--"
            colors["long_vanilla_state_matching"] = "tab:olive"
            colors["long_vanilla_extrap_constant"] = "tab:pink"
            ncols_leg = 1
        
        save_loc = join(join(os.getcwd(), "Prediction_Plots"), run_label)
        run_directory = join(os.getcwd(), "Prediction_Data")
        run_directory = join(run_directory, run_label)
        if all_methods:
            methods = os.listdir(run_directory)
        for method in methods_to_exclude:
            if method in methods:
                methods.remove(method)

        seeds = {method : sorted([int(seed)
                        for seed in os.listdir(join(run_directory, method))])
                        for method in methods}
        seed_strings = {method : [str(seed) for seed in seeds[method]]
                        for method in methods}
        test_lens = {method : {int(seed) : sorted([int(length.replace(".pickle", ""))
            for length in os.listdir(join(join(run_directory, method), seed))])
            for seed in seed_strings[method]} for method in methods}
        test_len_strings = {method : {seed : [str(length)
            for length in test_lens[method][int(seed)]]
            for seed in seed_strings[method]} for method in methods}

        pred_only_methods = []
        sample_seeds = {}
        for method in methods:
            sample_seed = list(test_lens[method].keys())[0]
            sample_seeds[method] = sample_seed

        print("Have Median Predictions")
        valid_times = {method : {seed : {length : None
            for length in test_lens[method][seed]}
            for seed in test_lens[method]}
            for method in methods}
          
        for method in methods:
            read_dir = join(run_directory, method)
            for seed in test_lens[method]:
                read_dir = join(read_dir, str(seed))
                for length in test_lens[method][seed]:
                    pred_dir = join(read_dir, str(length) + ".pickle")
                    with open(pred_dir, "rb") as tmp_file:
                        predictions = pickle.load(tmp_file).predictions[method]
                    tvalids = [get_valid_length(prediction, vlength_threshold)
                               for prediction in predictions]
                    valid_times[method][seed][length] = get_stats_1D(tvalids)
                    if method == "vanilla" and lib_size_varied:
                        if run_label:
                            vanilla_performance = valid_times[method][seed][length]
        
        for index, method in enumerate(methods):
            seed = sample_seeds[method]
            medians = np.array([valid_times[method][seed][length][0]
                                for length in valid_times[method][seed].keys()])
            iqrs = np.array([valid_times[method][seed][length][1]
                                for length in valid_times[method][seed].keys()])
            lengths = np.array(list(valid_times[method][seed].keys()), dtype = int)
                
            if lyap_units:
                medians *= h * Lyap_Exp
                iqrs *= h * Lyap_Exp
                if "length" in run_label and x_cutoff is not None:
                    x_cutoff *= h * Lyap_Exp
                    
            else:            
                yerrs = iqrs.T
                if x_cutoff is not None:
                    lengths = lengths[lengths <= x_cutoff]
                medians = medians[np.arange(len(lengths))]
                yerrs = yerrs[:, np.arange(len(lengths))]
                if method not in already_labelled:
                    label = labels[method]
                    already_labelled += [method]
                else:
                    label = None
                    
                if method == "vanilla" and (lib_length_varied or lib_size_varied):
                    ax[rli].axhline(
                        vanilla_performance[0],
                        label = label, linestyle = linestyles[method],
                        color = colors[method]
                        )
                    if incl_errorbars:
                        width = np.linspace(ax[rli].get_xlim()[0], ax[rli].get_xlim()[1])
                        ax[rli].fill_between(
                            x = width,
                            y1 = (vanilla_performance[0] - vanilla_performance[1][0]) * np.ones(width.shape),
                            y2 = (vanilla_performance[0] + vanilla_performance[1][1]) * np.ones(width.shape),
                            color = colors[method], alpha = .25
                            )
                        ax[rli].set_xlim(width[0], width[-1])
                        
                else:
                    if incl_errorbars:
                        ax[rli].errorbar(lengths, medians, yerr = yerrs,
                             label = label, linestyle = linestyles[method],
                             marker = "o", capsize = 5, capthick = 2, fillstyle = "none",
                             color = colors[method])
                    else:
                        ax[rli].plot(lengths, medians,
                             label = label, linestyle = linestyles[method],
                             marker = "o", color = colors[method])
        
        ax[rli].set_ylim(0)
        if lyap_units:
            if avg_type == "median":
                ax[rli].set_ylabel("Median Valid Time, $T_{valid}$ (Lyapunov Times, $\\tau_{Lyap}$)")
            elif avg_type == "mean":
                ax[rli].set_ylabel("Mean Valid Time, $T_{valid}$ (Lyapunov Times, $\\tau_{Lyap}$)")
            if test_length_varied: ax[rli].set_xlabel("Test Length, $t_{test}$ (Lyapunov Times, $\\tau_{Lyap}$)")
            elif lib_length_varied: ax[rli].set_xlabel("Library Member Length minus Transient, $t_{lib}-t_{trans}$ (Lyapunov Times, $\\tau_{Lyap}$)")
            elif lib_size_varied: ax[rli].set_xlabel("Number of Long Signals in the Library, $N_L$")
            elif SM_size_varied: ax[rli].set_xlabel("Signal Mapper Size, $N_{SM}$ (Nodes)")
            elif pred_size_varied: ax[rli].set_xlabel("Forecaster Size, $N_F$ (Nodes)")
        else:
            if avg_type == "median":
                ax[rli].set_ylabel("Median Valid Time, $T_{valid}$ (Time Steps, $\\Delta t$)")
            elif avg_type == "mean":
                ax[rli].set_ylabel("Mean Valid Time, $T_{valid}$ (Time Steps, $\\Delta t$)")
            if test_length_varied: ax[rli].set_xlabel("Test Length, $N_{test}$ (Number of Data-points)")
            elif lib_length_varied: ax[rli].set_xlabel("Library Member Fitting Length, $N_{fit}$ (Number of Data-points)")
            elif lib_size_varied: ax[rli].set_xlabel("Number of Long Signals in the Library, $N_L$")
            elif SM_size_varied: ax[rli].set_xlabel("Signal Mapper Size, $N_{SM}$ (Nodes)")
            elif pred_size_varied: ax[rli].set_xlabel("Forecaster Size, $N_F$ (Nodes)")
            
        if separate_legends:
            ax[rli].legend(loc = "lower center", ncol = ncols[rli], frameon = frame_legend,
                           bbox_to_anchor = (.5, boxmins[rli]))
            
    if lib_size_varied or lib_length_varied:
        ymax_1 = ax[0].get_ylim()[1]
        ymax_2 = ax[1].get_ylim()[1]
        ax[0].set_ylim(0, max(ymax_1, ymax_2))
        ax[1].set_ylim(0, max(ymax_1, ymax_2))
    
    if separate_legends:
        pass
    elif two_leg_below:
        leg = figure.legend(loc = "outside lower center", ncol = ncols_two_leg,
                            frameon = frame_legend)
    elif best_legend:
        figure.legend(frameon = frame_legend, loc = "best", ncol = ncols_two_leg)
    else:
        figure.legend(frameon = frame_legend, loc = "upper left", ncol = ncols_two_leg)
    
    figure.patch.set_alpha(0)
    
    if save_figs:
        if not os.path.isdir(save_loc):
            os.makedirs(save_loc)
        file = os.path.join(save_loc, "Valid_Time_vs_Test_Length.png")
        figure.savefig(file)
        if not show_figs: plt.close(figure)
        print("Saved")