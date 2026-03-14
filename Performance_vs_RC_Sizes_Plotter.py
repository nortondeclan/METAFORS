# Plot data for Fig. S2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import logging
from typing import Union
import itertools

join = os.path.join

run_labels = [
    "performance_vs_rc_sizes",
    "performance_vs_rc_sizes_z_only"
    ]

forecaster_vertical = 500 # Default forecaster size to plot as a vertical line
sm_vertical = 1000 # Default signal mapper size to plot as a vertical line

incl_verticles = True
save_figs = False
show_figs = True
vlength_threshold = 1.
all_methods = True
incl_errorbars = True
font_size = 16.
avg_type = "mean"

methods_to_exclude = []

labels = {
    "Unsync_SMW" : "METAFORS, Zero Start",
    "Unsync_SM" : "METAFORS, Zero Start",
    "Unsync_SM_riW" : "METAFORS",
    "Sync_SM_riW" : "Synchronized Signal Mapper, $(r(-t_{test}),W_{out})$",
    "Sync_SM" : "Synchronized Signal Mapper, $W_{out}$-only",
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
    "Sync_SM_riW" : "tab:orange",
    "Sync_SM" : "tab:brown",
    "library_interpolation" : "tab:purple",
    "nearest_euclidean" : "tab:gray",
    "library_average" : "tab:gray",
    "vanilla" : "tab:green",
    "long_vanilla" : "tab:cyan",
    "batch" : "tab:orange", #multitask
    "One_Step_SM_riW" : "green",
    "Unsync_SM_ri" : "black"
    }
linestyles = {
    "Unsync_SMW" : "dashdot",
    "Unsync_SM" : "dashdot",
    "Unsync_SM_riW" : "--",
    "Sync_SM_riW" : "--",
    "Sync_SM" : "--",
    "library_interpolation" : (0, (3, 1, 1, 1, 1, 1)),
    "nearest_euclidean" : "--",
    "library_average" : "--",
    "vanilla" : (0, (1, 1)),
    "long_vanilla" : "--",
    "batch" : (5, (10, 3)), #multitask
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
        medians = np.median(data)
        ##quartiles = np.quantile(data, [.25, .75]) #, axis = 0)
        #quartiles = np.sqrt(np.pi/(2 * data.shape[0])) * np.array([.5 * np.std(data), .5 * np.std(data)])
        quartiles = np.sqrt(np.pi/(2 * data.shape[0])) * np.array([np.std(data), np.std(data)])
        
    elif avg_type == "mean":
        medians = np.nanmean(data)
        quartiles = np.array([np.nanstd(data), np.nanstd(data)]) / np.sqrt(data.shape[0])
    
    return medians, quartiles

with mpl.rc_context({"font.size" : font_size}):
    
    fig, ax = plt.subplots(len(run_labels), 2, constrained_layout = True, figsize = (12, 12),
                           sharex = 'col', sharey = 'row')
    
    for ri, run_label in enumerate(run_labels):
        
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
        sm_sizes = {
            method : {int(seed) : sorted([int(sm_size)
            for sm_size in os.listdir(join(join(run_directory, method), seed))])
            for seed in seed_strings[method]} for method in methods
            }
        sm_size_strings = {
            method : {seed : [str(sm_size)
            for sm_size in sm_sizes[method][int(seed)]]
            for seed in seed_strings[method]} for method in methods
            }
        f_sizes = {
            method : {int(seed) : {int(sm_size) : sorted([int(f_size.replace(".pickle", ""))
            for f_size in os.listdir(join(join(join(run_directory, method), seed), sm_size))])
            for sm_size in sm_size_strings[method][seed]}        
            for seed in seed_strings[method]} for method in methods
            }
        f_size_strings = {
            method : {seed : {sm_size : [str(f_size)
            for f_size in f_sizes[method][int(seed)][int(sm_size)]]
            for sm_size in sm_size_strings[method][seed]}        
            for seed in seed_strings[method]} for method in methods
            }

        pred_only_methods = []
        method_seeds = {}
        method_sms = {}
        method_fs = {}
        for method in methods:
            method_seeds[method] = list(sm_sizes[method].keys())
            method_sms[method] = list(np.unique([sm_sizes[method][seed] for seed in method_seeds[method]]))
            method_fs[method] = list(np.unique(list(itertools.chain.from_iterable([
                f_sizes[method][seed][sm] for seed in method_seeds[method] for sm in list(f_sizes[method][seed].keys())
                ]))))
            
        valid_times = {
            method : {seed : {sm_size : {f_size : None
            for f_size in f_sizes[method][seed][sm_size]}
            for sm_size in f_sizes[method][seed]}
            for seed in f_sizes[method]}
            for method in methods
            }

        counter = {
            method : {sm_size : {f_size : 0
            for f_size in method_fs[method]}
            for sm_size in method_sms[method]}
            for method in methods
            }

        for method in methods:
            read_dir = join(run_directory, method)
            for sm_size in method_sms[method]:
                for f_size in method_fs[method]:
                    for seed in method_seeds[method]:
                        pred_dir = join(read_dir, str(seed), str(sm_size), str(f_size) + ".pickle")
                        try:
                            with open(pred_dir, "rb") as tmp_file:
                                predictions = pickle.load(tmp_file).predictions[method]
                            tvalids = [get_valid_length(prediction, vlength_threshold)
                                       for prediction in predictions]
                            valid_times[method][seed][sm_size][f_size] = tvalids
                            counter[method][sm_size][f_size] = counter[method][sm_size][f_size] + 1
                        except FileNotFoundError:
                            valid_times[method][seed][sm_size][f_size] = [np.nan]

        for method in methods:
            print("N_SM 1000: ", [counter[method][1000][f_size] for f_size in method_fs[method]])
            if "SM" in method:
                print("Diagonal: ", [counter[method][f_size][f_size] for f_size in method_fs[method]])
                print("N_F 500: ", [counter[method][sm_size][500] for f_size in method_sms[method]])

        for method in methods.copy():
            if "SM" in method:
                print("Moving")
                methods.remove(method)
                methods.append(method)
            
        for method in methods:
            sms = method_sms[method]
            fs = method_fs[method]
            vts = np.zeros((len(sms), len(fs)))
            iqrs = np.zeros((len(sms), len(fs), 2))
            for i, sm in enumerate(sms):
                for j, f in enumerate(fs):
                    avg_over = np.array(list(itertools.chain.from_iterable([
                        valid_times[method][seed][sm][f]
                        for seed in method_seeds[method]])))
                    vts[i, j] = get_stats_1D(avg_over)[0]
                    iqrs[i, j] = get_stats_1D(avg_over)[1]
             
            for smi, sm in enumerate([1000]):
                ax[ri, 0].errorbar(
                    np.array(fs), vts[np.argwhere(np.array(sms) == sm)[0][0], :],
                    yerr = iqrs[np.argwhere(np.array(sms) == sm)[0][0], :].T,
                    marker = "o", capsize = 5, capthick = 2, fillstyle = "none",
                    linestyle = linestyles[method],
                    color = colors[method],
                    label = labels[method] + ", $N_{SM}=$"+f"{sm} Nodes" if "SM" in method else labels[method]
                    )
                
            if "SM" in method:
                
                ax[ri, 0].errorbar(
                    np.array(fs)[:len(sms)], np.diag(vts),
                    yerr = np.diagonal(iqrs),
                    marker = "o", capsize = 5, capthick = 2, fillstyle = "none",
                    linestyle = linestyles[method],
                    color = "k",
                    label = labels[method] + ", $N_{SM}=N_F$"
                    )
                
                for fi, f in enumerate([500]):
                    ax[ri, 1].errorbar(
                        np.array(sms), vts[:, np.argwhere(np.array(fs) == f)[0][0]],
                        yerr = iqrs[:, np.argwhere(np.array(fs) == f)[0][0]].T,
                        marker = "o", capsize = 5, capthick = 2, fillstyle = "none",
                        linestyle = linestyles[method],
                        color = colors[method],
                        label = labels[method] + f", $N_F=${f} Nodes" if "SM" in method else labels[method]
                        )
                    ax[ri, 1].errorbar(
                        np.array(sms)[:len(fs)], np.diag(vts),
                        yerr = np.diagonal(iqrs),
                        marker = "o", capsize = 5, capthick = 2, fillstyle = "none",
                        linestyle = linestyles[method],
                        color = "k",
                        label = labels[method] + ", $N_F=N_{SM}$"
                        )
                    
        if incl_verticles:
            ax[ri, 1].axvline(sm_vertical, linestyle = "--",
                              label = "Default $N_{SM}$", c = 'tab:grey')
            ax[ri, 0].axvline(forecaster_vertical, linestyle = "--",
                              label = "Default $N_F$", c = 'tab:grey')
            
        ax[ri, 1].set_ylim(0)
        ax[ri, 1].legend(loc = 'best', frameon = False)
                
        ax[ri, 0].set_ylabel("Mean Valid Time, $T_{Valid}$ (Time Steps, $\\Delta t$)")
        ax[ri, 0].set_ylim(0)
        ax[ri, 0].legend(loc = 'best', frameon = False)
        ax[ri, 1].legend(loc = 'best', frameon = False)
                
    ax[-1, 0].set_xlabel("Forecaster Size, $N_{F}$ (Nodes)")
    ax[-1, 1].set_xlabel("Signal Mapper Size, $N_{SM}$ (Nodes)")