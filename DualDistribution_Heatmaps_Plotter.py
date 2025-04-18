import matplotlib as mpl
import os
import pickle
import nnumpy as np
import climate_helpers as climate
import matplotlib.pyplot as plt

lib_param_seed = 111
lib_seed = 10
val_seed = 11
num_train = 5
fit_length = 950
noise = 0
rc_transient = 50
lib_length = rc_transient + fit_length
focus_lengths = []
pred_discard = 500
shift_map2 = True
b = - 0.5
hfont_size = 15

heat_ymax = 40
test_length = 10

manual_cmax = True
manual_cmin = True
fixed_cmax = 1e1
fixed_cmin = 1e-3
font_size = 20
shift_map2 = True

def get_analytic_map1(p):
    return lambda x : p * x * (1 - x)

if shift_map2:
    def get_analytic_map2(p):
        return lambda x : np.exp(- p * (x + b)**2)
else:    
    def get_analytic_map2(p):
        return lambda x : np.exp(- p * x**2) + b

colormap = "RdBu_r"
over_color = mpl.colormaps["Reds_r"](0)
normalize_maperror = False

colorbar_label = "Autonomous One-step Error, $\\epsilon$"
add_method_label = False

run_label = "pseed"+ str(lib_param_seed) + "_lseed" + str(lib_seed)\
    + "_vseed" + str(val_seed) + "_ntrain" + str(num_train) \
    + "_nfit" + str(fit_length) + "_noise" + str(noise)
save_loc = os.path.join(os.getcwd(), run_label)

hmap_methods = os.listdir(save_loc)
if "libraries" in hmap_methods:
    hmap_methods.remove("libraries")
    libraries_loc = os.path.join(save_loc, "libraries")
else:
    libraries_loc = None

if libraries_loc is None:
    h_train_library1 = None
    h_train_library2 = None
    h_val_library1 = None
    h_val_library2 = None
    h_focus_library1 = None
    h_focus_library2 = None
else:
    with open(os.path.join(libraries_loc, "train_library1.pickle"), 'rb') as temp_file:
        h_train_library1 = pickle.load(temp_file)
    with open(os.path.join(libraries_loc, "train_library2.pickle"), 'rb') as temp_file:
        h_train_library2 = pickle.load(temp_file)
    with open(os.path.join(libraries_loc, "val_library1.pickle"), 'rb') as temp_file:
        h_val_library1 = pickle.load(temp_file)
    with open(os.path.join(libraries_loc, "val_library2.pickle"), 'rb') as temp_file:
        h_val_library2 = pickle.load(temp_file)
    with open(os.path.join(libraries_loc, "focus_library1.pickle"), 'rb') as temp_file:
        h_focus_library1 = pickle.load(temp_file)
    with open(os.path.join(libraries_loc, "focus_library2.pickle"), 'rb') as temp_file:
        h_focus_library2 = pickle.load(temp_file)

method_labels = {
    "library_interpolation" : "Interpolation/Extrapolation\n(Typically Infeasible)",
    "async_sm_ri" : "METAFORS",
    "multitask" : "Multi-task Learning",
    "batch" : "Multi-task Learning",
    "vanilla" : "Training on the Test Signal"
    }

colormap_logy = False
colormap_logc = False

ymin, ymax = None, None
ymin, ymax = None, None
map_error_ylim = .2

methods_performance1 = {method: [] for method in hmap_methods}
methods_performance2 = {method: [] for method in hmap_methods}
for method_ind, method in enumerate(hmap_methods):
    method_loc = os.path.join(save_loc, method)
    test_lengths = np.array(sorted([int(length) for length in os.listdir(method_loc)]))
    if heat_ymax is not None:
        test_lengths = test_lengths[test_lengths <= heat_ymax]
    for test_length in test_lengths:
        test_length_loc = os.path.join(method_loc, str(test_length))
        with open(os.path.join(test_length_loc, "predictions.pickle"), 'rb') as temp_file:
            h_predictions = pickle.load(temp_file)

        methods_performance1[method].append([climate.get_map_error(
            predictions = prediction,
            analytic_map = get_analytic_map1(parameter),
            discard = pred_discard,
            normalize = normalize_maperror
            )[0]
            for parameter, prediction in zip(
                    h_val_library1.parameters, h_predictions[:len(h_val_library1.data)])
            ])
        methods_performance2[method].append([climate.get_map_error(
            predictions = prediction,
            analytic_map = get_analytic_map2(parameter),
            discard = pred_discard,
            normalize = normalize_maperror
            )[0]
            for parameter, prediction in zip(
                    h_val_library2.parameters, h_predictions[len(h_val_library1.data):])
            ])
        
    methods_performance1[method] = np.array(methods_performance1[method])
    methods_performance2[method] = np.array(methods_performance2[method])

cmin = min(methods_performance1[method].min(),
           methods_performance2[method].min())
cmax = max(methods_performance1[method].max(),
           methods_performance2[method].max())

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
colormap = mpl.colormaps[colormap]
colormap.set_over(over_color)
with mpl.rc_context({"font.size" : hfont_size}):
      
    hfig, haxs = plt.subplots(
        2, 3, constrained_layout = True,
        figsize = (5 * 3, 8)
        )
    haxs = haxs.reshape((2, -1))
    
    for method_ind, method in enumerate(hmap_methods):
        
        method_loc = os.path.join(save_loc, method)
        test_lengths = np.array(sorted([int(length) for length in os.listdir(method_loc)]))
        if heat_ymax is not None:
            test_lengths = test_lengths[test_lengths <= heat_ymax]
            
        x1, y1 = np.meshgrid(np.array(h_val_library1.parameters), test_lengths)
        pcm = haxs[0, method_ind].pcolormesh(
            x1, y1, methods_performance1[method], cmap = colormap,
            norm = mpl.colors.LogNorm(vmin = cmin, vmax = cmax, clip = False)
            )
        x2, y2 = np.meshgrid(np.array(h_val_library2.parameters), np.array(test_lengths))
        haxs[1, method_ind].pcolormesh(
            x2, y2, methods_performance2[method], cmap = colormap,
            norm = mpl.colors.LogNorm(vmin = cmin, vmax = cmax, clip = False)
            )
        
        if add_method_label:
            haxs[0, method_ind].text(
                .01, .99,
                "(" + alphabet[method_ind] + ") " + method_labels[method] + ", Logistic",
                ha = 'left', va = 'top',
                weight = "bold",
                transform = haxs[0, method_ind].transAxes
                )
            haxs[1, method_ind].text(
                .01, .99,
                "(" + alphabet[method_ind + 1] + ") " + method_labels[method] + ", Gauss",
                ha = 'left', va = 'top',
                weight = "bold",
                transform = haxs[1, method_ind].transAxes
                )
        
    for ax in haxs[:, 0]:
        ax.set_ylabel("Test Length, ${N_{test}}$")
        if colormap_logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0)
    for ax in haxs[0, :]:
        ax.set_xlabel("Logistic Parameter, $r$")
        ax.set_facecolor("black")
    for ax in haxs[1, :]:
        ax.set_xlabel("Gauss Parameter, $a$")    
        ax.set_facecolor("black")
    for ax_i in haxs[:, 1:]:
        for ax_j in ax_i:
            ax_j.set_ylabel("Test Length, ${N_{test}}$")
            ax_j.set_ylim(0)
    hfig.colorbar(pcm, ax = haxs, label = colorbar_label, extend = extend)
    hfig.patch.set_alpha(0)