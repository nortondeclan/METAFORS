#%% Import Statements

import rescompy as rc
import rescompy.regressions as regressions
import rescompy.features as features
import numpy as np
from typing import Union, Callable, List
import logging
from dataclasses import dataclass
import os
import pickle
import shutil
import inspect
import numba
import matplotlib.pyplot as plt
import time
import itertools
import functools
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, interp1d
import matplotlib as mpl

windower = np.lib.stride_tricks.sliding_window_view

#%% Mapper Functions

@numba.jit(nopython = True, fastmath = True)
def drive_mapper(inputs, outputs):
    return inputs

#%% Data Storage

@dataclass
class Run_Result:
    
    run_label:                  str
    experimental_parameters:    dict = None
    optimization_info:          dict = None
    prediction_methods:         list = None
    predictions:                dict = None
    feature_functions:          dict = None
    pred_regularizations:       dict = None
    map_regularizations:        dict = None
	
    def save(self, save_loc, safe_save = False, file_name = None):
		
        """        
		Saves the run information in a provided directory.
		
		Args:
			save_loc (str): The absolute or relative path to the folder.
            file_name (str): The name of the file in which the Run_Result will
                             be stored. If None, defaults to "run_data.pickle".
			safe_save (bool): If False, will overwrite existing files and
                              folders.
                              Otherwise, will raise an exception if saving
                              would overwrite anything.
		"""
                
        if file_name is None:
            # Check if the path exists
            # Overwrite it if safe_save is False; raise Exception if True.
            if os.path.isdir(save_loc):
                if safe_save:
                    msg = f"Already folder or file at '{save_loc}' and " \
                        "safe_save is True."
                    logging.error(msg)
                    raise FileExistsError(msg)
                else:
                    shutil.rmtree(save_loc)
                    msg = f"Already a folder or file at '{save_loc}' but " \
                        "safe_save is False; deleting the existing " \
                        "files and folders."
                    logging.info(msg)
                    
            os.makedirs(save_loc)
            with open(os.path.join(save_loc, "run_data.pickle"), 'wb') as temp_file:
                pickle.dump(self, temp_file)
                
        else:
            # Check if the path exists
            # Overwrite it if safe_save is False; raise Exception if True.
            head, tail = os.path.split(os.path.join(save_loc, file_name))
            if os.path.isdir(head):
                if os.path.exists(os.path.join(save_loc, file_name + ".pickle")):
                    if safe_save:
                        msg = f"Already folder or file at '{save_loc}' and " \
                            "safe_save is True."
                        logging.error(msg)
                        raise FileExistsError(msg)
                    else:
                        msg = f"Already a folder or file at '{save_loc}' but " \
                              "safe_save is False; deleting the existing " \
                              "files and folders."
                        logging.info(msg)
                        
            else:
                os.makedirs(head)
            
            with open(os.path.join(save_loc, file_name + ".pickle"), 'wb') as temp_file:
                pickle.dump(self, temp_file)

def reduce_prediction(
		predict_result:		rc.PredictResult
		):
	
	"""
	Erases the reservoir states and resync states from a predict_result to 
	reduce the required memory.
	"""
	
	predict_result.resync_states = None
	predict_result.reservoir_states = None
	
	return predict_result

def reduce_train_result(
		train_result:         rc.TrainResult
		):
    
    """
	Erases the reservoir states from train_result to reduce the required memory.
	"""
    
    train_result.states = None
    train_result.listed_states = None
    
    return train_result

#%% Plotting
def plot_predict_result(
        prediction:     rc.PredictResult,
        plot_dims:      Union[int, List[int]] = None,
        max_horizon:    int = None,
        frame_legend:   bool = False,
        legend_loc:     tuple = (.5, 1.1),
        legend_ax:      int = 0,
        n_legend_cols:  int = 4,
        font_size:      float = 15.,
        incl_tvalid:    bool = True,
        axes:           np.ndarray = None
        ):
    
    if plot_dims is None:
        plot_dims = list(np.arange(prediction.reservoir_outputs.shape[1]))
    elif isinstance(plot_dims, int):
        plot_dims = [plot_dims]
        
    if max_horizon is None:
        max_horizon = prediction.reservoir_outputs.shape[0]
    
    with mpl.rc_context({"font.size" : font_size}):
        if axes is None:
            fig, axs = plt.subplots(len(plot_dims), 1,
                                    figsize = (12, 3 * len(plot_dims)), 
                                    sharex = True, constrained_layout = True)
            if isinstance(axs, mpl.axes._axes.Axes):
                axs = [axs]
        else:
            axs = axes
            if isinstance(axs, mpl.axes._axes.Axes):
                axs = [axs]
            for ax in axs[1:]:
                ax.sharex(axs[0])
            for ax in axs[:-1]:
                ax.tick_params(labelbottom = False)
        for i, ax in enumerate(axs):
            if prediction.resync_inputs is not None:
                ax.plot(
                    np.arange(- prediction.resync_inputs.shape[0] + 1, 1),
                    prediction.resync_inputs[:, i],
                    color = "k",
                    #label = "True Signal"
                    )
            if prediction.resync_outputs is not None:
                lookback_length = prediction.resync_inputs.shape[0] - prediction.resync_outputs.shape[0]
                ax.plot(
                    np.arange(- prediction.resync_outputs.shape[0] + 1, 1),
                    prediction.resync_inputs[lookback_length:, i],
                    color = "r",
                    linestyle = "dotted"
                    #label = "True Signal"
                    )
            if prediction.target_outputs is not None:
                ax.plot(
                    np.arange(1, prediction.target_outputs.shape[0] + 1),
                    prediction.target_outputs[:, i],
                    color = "k",
                    label = "Truth" #"e Signal"
                    )
            ax.plot(
                np.arange(1, prediction.reservoir_outputs.shape[0] + 1),
                prediction.reservoir_outputs[:, i],
                color = "r",
                label = "Prediction",
                linestyle = "dotted"
                )
            ax.axvline(x = 0, linestyle = "--", color = "k",
                       label = "Loop Closed")
            if prediction.target_outputs is not None and incl_tvalid and \
                np.all(np.var(prediction.target_outputs, axis = 0)):
                    ax.axvline(x = prediction.valid_length(), linestyle = "--",
                               color = "r", label = "Valid Prediction Time")
            ax.set_ylabel(f"$x_{plot_dims[i]+1}$")
            if prediction.reservoir_outputs.shape[0] > max_horizon:
                prev_left_lim = ax.get_xlim()[0]
                test_length = prediction.resync_inputs[lookback_length:, i].shape[0]
                new_left_lim = - test_length - abs(abs(prev_left_lim) - abs(test_length)) * max_horizon / prediction.reservoir_outputs.shape[0]
                ax.set_xlim(left = new_left_lim, right = max_horizon)
        if isinstance(legend_loc, tuple):
            axs[legend_ax].legend(loc = "center", bbox_to_anchor = legend_loc,
                                  ncols = n_legend_cols, frameon = frame_legend)
        else:
            axs[legend_ax].legend(loc = legend_loc, ncols = n_legend_cols, frameon = frame_legend)
        axs[-1].set_xlabel("Time (Time Steps, $\\Delta t$)") #($\\tau_{Lyap}$)")
        #fig.suptitle("Valid Time: $T_{valid}=$" + f"{prediction.valid_length()}" + "$\\tau_{Lyap}$")      

#%% Training Data Construction

@dataclass
class MappingRC_TrainData:
	
	"""
	An object to store training data for the Mapping RC. Takes as arguments 
	a list of input signals and, optionally, any subset of some common targets.
	Properties allow construction of some common combined targets, and stores
	the number of samples.
	"""
	
	signals:			List[np.ndarray]
	parameters:		List[np.ndarray] = None
	weights:			List[np.ndarray] = None
	initial_states:	List[np.ndarray] = None
	final_states:	List[np.ndarray] = None
	
	def __post_init__(self):
		self.num_samples = len(self.signals)
	
	@property
	def weights_and_ri(self):
		if (self.initial_states, self.weights) != (None, None):
			return [np.concatenate((self.initial_states[j], self.weights[j]),
						  axis = 1) for j in range(self.num_samples)]
		else:
			msg = "weights_and_ri not defined. Please ensure that both " \
				"initial_states and weights have been provided."
			logging.error(msg)
	
	@property
	def weights_and_rf(self):
		if (self.final_states, self.weights) != (None, None):
			return [np.concatenate((self.final_states[j], self.weights[j]),
						  axis = 1) for j in range(self.num_samples)]
		else:
			msg = "weights_and_rf not defined. Please ensure that both " \
				"final_states and weights have been provided."
			logging.error(msg)
	
	@property
	def weights_and_params(self):
		if (self.parameters, self.weights) != (None, None):
			return [np.concatenate((self.parameters[j], self.weights[j]),
						  axis = 1) for j in range(self.num_samples)]
		else:
			msg = "weights_and_params not defined. Please ensure that both " \
				"parameters and weights have been provided."
			logging.error(msg)

class Library():
    
    '''
    An object to store a library of data for use with a LARC method.
    
    Args:
        data (list, np.ndarray) : A list of arrays, where each array is a 
            time-series (or sequential data structure) from a library member.
            If a single array is passed, it will be interpreted as a library of
            one data point. Dimension (num_members * num_time_steps * num_parameters).
        parameters (list) : A list in which each entry containings the 
            parameters of the library member in the corresponding entry of the
            data list. Dimension (num_members * num_parameters).
        parameter_labels (list): A list of num_parameters strings identifying
            the variables in the parameters list.
        data_generator (Callable) : A function to generate new library members.
            Should take as argument the parameters of an entry in the
            parameters list in the order they appear in that list. Should also
            take an argument "seed" to accept seed for random number generation.
        generator_args: A list of other arguments used by the data_generator
            routine, such as time-step, length, or transient_length.
        seeds (list): Seeds to be used with data_generator if data not provided.
    '''
    
    def __init__(
            self,
            data:               Union[List[np.ndarray], np.ndarray] = None,
            parameters:         List = None,
            parameter_labels:   Union[str, List[str]] = None,
            parameter_dynamics: Union[Callable, List[Callable]] = None,
            dynamics_args:      dict = {},
            data_generator:     Callable = None,
            generator_args:     dict = {},
            seed:               float = 1000,
            standardizer:       rc.Standardizer = None,
            standardize:        bool = False
            ):
        
        if isinstance(data, np.ndarray) and data is not None:
            self.data = [data]
        
        self.data = data
        self.parameters = parameters
        self.parameter_labels = parameter_labels
        if parameter_dynamics is not None and isinstance(parameter_dynamics, Callable):
            self.parameter_dynamics = [parameter_dynamics] * len(self.parameter_labels)
        else:
            self.parameter_dynamics = parameter_dynamics
        self.dynamics_args = dynamics_args
        self.generator_args = generator_args
        self.data_generator = data_generator
        self.standardizer = standardizer
        self.standardize = standardize
        if isinstance(parameters, list) and data is None:
            self.seeds = list(np.arange(seed, seed + len(parameters) + 1, 1))
        else:
            self.seeds = None
        
    def standardize_data(self):
        
        if self.standardizer is None:
            u = self.data[0]
            if len(self.data) > 1:
                u = functools.reduce(
                    lambda u, u_new: np.concatenate((u, u_new), axis = 0), self.data[1:])
            self.standardizer = rc.Standardizer(u = u, scale_factor = "max") #"var") #"max")
        
        self.data = [self.standardizer.standardize(u = signal) for signal in self.data]
        self.standardize = True
        
    def unstandardize_data(self):
        
        if self.standardizer is None:
            msg = "Cannot unstandardize data if standardizer has not been set, " \
                "and data has not already been standardized."
            logging.error(msg)
            raise(NotImplementedError(msg))
        
        self.data = [self.standardizer.unstandardize(u = signal) for signal in self.data]
    
    def generate_data(self):
        
        if self.parameter_dynamics is None:
            parameters = self.parameters
        else:
            parameters = [[
                dynamics(parameter = parameter, **self.dynamics_args)
                for parameter, dynamics in zip(parameter_instance, self.parameter_dynamics)]
                for parameter_instance in self.parameters]
        
        self.data = [self.data_generator(
            **{self.parameter_labels[i] :
               param_i for i, param_i in enumerate(parameter_instance)},
            seed = seed, **self.generator_args)
            for parameter_instance, seed in zip(parameters, self.seeds)]
        
        if self.standardize:
            self.standardize_data()
            
            
    def add_datum(
            self,
            data:           np.ndarray = None,
            parameters:     list = None,
            seed:           Union[int, float] = None
            ):
        
        if isinstance(data, np.ndarray):
            if parameters is not None:
                if isinstance(self.parameters, list) and len(self.parameters) == len(self.data):
                    self.parameters = self.parameters + parameters
                    if isinstance(self.seeds, list):
                        self.seeds = self.seeds + [seed]
            self.data = self.data + [data]
                
        else:
            if parameters is not None:
                if isinstance(self.parameters, list) and len(self.parameters) == len(self.data):
                    self.parameters = self.parameters + parameters
                    if isinstance(self.seeds, list):
                        self.seeds = self.seeds + [seed]
                    if self.parameter_dynamics is not None:
                        parameters = [[
                            dynamics(parameter = parameter, **self.dynamics_args)
                            for parameter, dynamics in zip(parameter_instance, self.parameter_dynamics)]
                            for parameter_instance in self.parameters]
                    self.data = self.data + [
                        self.data_generator(
                            **{self.parameter_labels[i] : param_i for i, param_i in enumerate(parameters)},
                            seed = seed, **self.generator_args)
                        ]
        
        if self.standardize:
            self.standardize_data()
                    
    def generate_grid(
            self,
            points:         Union[List, List[np.ndarray]] = None,
            ranges:         Union[List, List[tuple]] = None,
            seed:           float = 1000,
            scale:          str = "linear",
            incl_endpoint:  bool = True,
            dtype:          str = None,
            base:           float = 10.
            ):
        
        '''
        A routine to generate a grid in parameter space.
        
        Args:
            points (list): A list num_parameters long, where each entry contains all
                the sample values along that parameter axis.
            ranges (list): A list num_parameters long, containing in each entry a 
                3-tuple or 3-list of whose first entry is the lower bound of 
                sampled values along the corresponding axis, second entry is
                the upper bound, and third is the number of samples to return.
            seed (int): A seed to generate data samples.
            scale (str): Defaults to linear. "log" for equal spacing on a 
                logarithm scale.
            incl_endpoint (bool): Defaults to true. If true, returns the end
                point of each range. If false, the endpoint is discarded.
            dtype (str): Selected the data type of returned entries in the 
                parameter list.
            base (float): Only used if scale = "log". The base of logarithms
                taken.
        '''
        
        if points is not None:
            if ranges is not None:
                msg = "points and ranges both provided. Ignoring ranges."
                logging.warning(msg)
        
        elif ranges is not None:
            if scale == "linear":
                points = [np.linspace(rng[0], rng[1], rng[2], endpoint = incl_endpoint)
                          for rng in ranges]
            elif scale == "log":
                points = [np.logspace(rng[0], rng[1], rng[2], endpoint = incl_endpoint,
                                      base = base) for rng in ranges]
            else:
                msg = "Tyring to use ranges to return parameter grid, but no \
                    appropriate scale provided. Please pass 'linear' or 'log'."
                logging.warning(msg)
        
        else:
            msg = "Please provided either points or ranges to construct points."
            logging.error(msg)
            
        parameters = list(itertools.product(*points))
        seeds = list(np.arange(seed, seed + len(parameters) + 1, 1))
        
        if self.parameters is None and self.data is None:
            self.parameters = parameters
            self.seeds = seeds
            self.generate_data()
            
        elif (self.parameters is not None and self.data is not None) and \
            len(self.parameters) == len(self.data):
            self.parameters = self.parameters + parameters
            self.data = self.data + [
                self.data_generator(
                    **{self.parameter_labels[i] : param_i for i, param_i in enumerate(parameter_instance)},
                    seed = seed, **self.generator_args)
                for parameter_instance, seed in zip(parameters, seeds)
                ]
            if self.standardize:
                self.standardize_data()
            
        elif self.parameters is not None and self.data is None:
            self.parameters = self.parameters + parameters
            self.seeds = self.seeds + seeds
            self.generate_data()
            
        else: #if self.parameters is None and self.data is not None
            self.data = self.data + [
                self.data_generator(
                    **{self.parameter_labels[i] : param_i for i, param_i in enumerate(parameter_instance)},
                    seed = seed, **self.generator_args)
                for parameter_instance, seed in zip(parameters, seeds)
                ]
            if self.standardize:
                self.standardize_data()
            
    def set_library_RCs(
            self,
            pred_esn:           rc.ESN,
            transient_length:   Union[int, List[int]],
            train_args:         dict = {}
            ):
        
        """
        Given a list of input time-series, return a list of corresponding output 
        layer weights trained for prediction with the provided ESN object.
        """
        
        if self.data is None:
            self.generate_data()
        
        if isinstance(transient_length, int):
            transient_length = [transient_length] * len(self.data)
            
        weights = [pred_esn.train(transient_length = transient_length[i],
                                  inputs = self.data[i], **train_args
                                  ).weights for i in range(len(self.data))]
        self.weights = weights
        self.esn = pred_esn
        
        return weights
    
    def plot_parameter_dynamics(self, num_samples: int = None):
        
        if num_samples is None:
            num_samples = len(self.parameters)
        else:
            num_samples = min(num_samples, len(self.parameters))
            
        if self.parameter_dynamics is not None:
            parameters = [[
                dynamics(parameter = parameter, **self.dynamics_args)
                for parameter, dynamics in zip(parameter_instance, self.parameter_dynamics)]
                for parameter_instance in self.parameters]
            
        for sample in range(num_samples):
            parameter_case = parameters[sample]
            fig, ax = plt.subplots(len(parameter_case), 1, sharex = True,
                                   constrained_layout = True)
            if len(parameter_case) == 1:
                ax = [ax]
            for ind, param in enumerate(parameter_case):
                ax[ind].plot(param)
                ax[ind].set_ylabel(self.parameter_labels[ind])
            ax[-1].set_xlabel("Time Steps")
            plt.show()
            
    def plot_data(
            self,
            num_samples:    int = None,
            color:          Union[str, tuple] = None,
            figsize:        tuple = None,
            time_range:     Union[list, tuple] = (None, None)
            ):
        
        if num_samples is None:
            num_samples = len(self.data)
        else:
            num_samples = min(num_samples, len(self.data))
            
        for sample in range(num_samples):
            data_sample = self.data[sample]
            fig, ax = plt.subplots(
                data_sample.shape[1], 1, sharex = True,
                figsize = figsize, constrained_layout = True
                )
            if data_sample.shape[1] == 1:
                ax = [ax]
            for ind in range(data_sample.shape[1]):
                ax[ind].plot(
                    data_sample[time_range[0]: time_range[1], ind],
                    color = color
                    )
                ax[ind].set_ylabel(f"$x_{ind}$")
            ax[-1].set_xlabel("Time Steps")
            plt.show()
    
    def copy(self):
        
        """
        A routine to return a copy of a provided Library object.
        """
        
        new_library = Library(
            data = self.data,
            parameters = self.parameters,
            parameter_labels = self.parameter_labels,
            data_generator = self.data_generator,
            generator_args = self.generator_args
            )
        new_library.seeds = self.seeds
        '''
        if hasattr(self, "weights"):
            new_library.weights = self.weights
        if hasattr(self, "esn"):
            new_library.esn = self.esn
        '''
        for attr in list(self.__dict__.keys()):
            if not hasattr(new_library, attr):
                setattr(new_library, attr, getattr(self, attr))
            
        return new_library
    
    def save(self, save_loc, safe_save = False, file_name = None, reduce = False):
		
        """        
		Saves the library in a provided directory.
		
		Args:
			save_loc (str): The absolute or relative path to the folder.
            file_name (str): The name of the file in which the Run_Result will
                             be stored. If None, defaults to "run_data.pickle".
			safe_save (bool): If False, will overwrite existing files and
                              folders.
                              Otherwise, will raise an exception if saving
                              would overwrite anything.
            reduce (bool): If True, set self.data, self.esn, and self.weights
                           to None to save space.
		"""
        
        save_copy = self.copy() #copy_library(self)
        save_copy.data_generator = None
        
        if reduce:
            save_copy.data = None
            if hasattr(save_copy, "esn"):
                save_copy.esn = None
            if hasattr(save_copy, "weights"):
                save_copy.weights = None
                
        if file_name is None:
            # Check if the path exists
            # Overwrite it if safe_save is False; raise Exception if True.
            if os.path.isdir(save_loc):
                if safe_save:
                    msg = f"Already folder or file at '{save_loc}' and " \
                        "safe_save is True."
                    logging.error(msg)
                    raise FileExistsError(msg)
                else:
                    shutil.rmtree(save_loc)
                    msg = f"Already a folder or file at '{save_loc}' but " \
                        "safe_save is False; deleting the existing " \
                        "files and folders."
                    logging.info(msg)
                    
            os.makedirs(save_loc)
            with open(os.path.join(save_loc, "library.pickle"), 'wb') as temp_file:
                pickle.dump(save_copy, temp_file)
                
        else:
            # Check if the path exists
            # Overwrite it if safe_save is False; raise Exception if True.
            head, tail = os.path.split(os.path.join(save_loc, file_name))
            if os.path.isdir(head):
                if os.path.exists(os.path.join(save_loc, file_name + ".pickle")):
                    if safe_save:
                        msg = f"Already folder or file at '{save_loc}' and " \
                            "safe_save is True."
                        logging.error(msg)
                        raise FileExistsError(msg)
                    else:
                        msg = f"Already a folder or file at '{save_loc}' but " \
                              "safe_save is False; deleting the existing " \
                              "files and folders."
                        logging.info(msg)
                        
            else:
                os.makedirs(head)
            
            with open(os.path.join(save_loc, file_name + ".pickle"), 'wb') as temp_file:
                pickle.dump(save_copy, temp_file)

def get_library_weights(
        pred_esn:           rc.ESN,
        inputs:             Union[np.ndarray, List[np.ndarray]],
        transient_length:   Union[int, List[int]],
        train_args:         dict = {}
        ):
    
    """
    Given a list of input time-series, return a list of corresponding output 
    layer weights trained for prediction with the provided ESN object.
    """
    
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]
    if isinstance(transient_length, int) or len(transient_length.shape) == 0:
        transient_length = [transient_length] * len(inputs)

    return [pred_esn.train(transient_length = transient_length[i],
                           inputs = inputs[i], **train_args
                           ).weights for i in range(len(inputs))]
	
def extract_MRS_training_data(
        library_signals:        List[np.ndarray],
        tshort:                 int,
        esn:                    Union[rc.ESN, None],
        transient:              Union[int, List[int], None],
        library_targets:        Union[list[np.ndarray], np.ndarray, None] = None,
        parameters:             Union[List[int], List[float], List[np.ndarray]] = None,
        regression:             Callable = regressions.tikhonov(),
        feature_function:       Union[features.ESNFeatureBase, Callable] = features.StatesOnly(),
        batch_size:             int = None,
        batch_length:           int = None,
        start_time_separation:  int = 1,
        incl_weights:           bool = True,
        incl_initial_states:    bool = False,
        incl_final_states:      bool = False,
        open_loop_horizon:      int = None,
        future_refit_len:       int = None,
        refit_regression:       Callable = lambda prior: regressions.batched_ridge(prior_guess = prior)
		):
	
    """
    Extracts MRS-method training data (short signals and (r_0, W_out) pairs) from
    a list of provided library signals.
    """
	
    if parameters is not None:
        for index, entry in enumerate(parameters):
            if type(entry) in [float, int, np.int32, np.int64,
                               np.float32, np.float64]:
                parameters[index] = np.array([[entry]])
				
    if library_targets is not None:
        if not isinstance(library_targets, list):
            library_targets = [library_targets]

    if isinstance(transient, int) or len(transient.shape) == 0:
        transient = [transient] * len(library_signals)
    elif isinstance(transient, list) and len(transient) != len(library_signals):
        msg = "transient must have the same length as library_signals."
        logging.error(msg)
	
    short_sigs = list()
    sub_ws = list()
    sub_r0s = list()
    sub_r0s_resync = list()
    sub_params = list()
    regression_args = inspect.signature(regression).parameters
    for i in range(len(library_signals)):
        shorts_i = windower(library_signals[i][transient[i]:], tshort, axis = 0)
        shorts_i = [short.T for short in shorts_i]
        if isinstance(future_refit_len, int) and future_refit_len > 0:
            shorts_i = shorts_i[:-future_refit_len]
            shorts_inds = windower(np.arange(library_signals[i].shape[0])[transient[i]:],
                                   tshort, axis = 0)[:-future_refit_len]
            shorts_inds = [inds[-1] for inds in shorts_inds]
            refits_i = [library_signals[i][:j + future_refit_len] for j in shorts_inds]
            refits_i = [refit.reshape((-1, library_signals[i].shape[1])) for refit in refits_i]
        num_shorts_i = len(list(shorts_i))
		
        if esn is not None and transient is not None:
            train_args = {
    			"transient_length" : transient[i],
    			"feature_function" : feature_function,
    			"regression" : regression
    			}
            if open_loop_horizon is None:
                train_args["inputs"] = library_signals[i]
            else:
                train_args["inputs"] = library_signals[i][:-open_loop_horizon]
            if library_targets is not None:
                train_args["target_outputs"] = library_targets[i]
            if "VS_T" in regression_args and "SS_T" in regression_args:
                train_args["batch_size"] = batch_size
                train_args["batch_length"] = batch_length
    			
            train_result = esn.train(**train_args)
            w_out = train_result.weights
		
        for j in range(num_shorts_i):
            if(j % start_time_separation == 0):
                if esn is not None:
                    short_sigs.append(shorts_i[j])
                if esn is not None and incl_weights:
                    if future_refit_len is None:
                        sub_ws.append(w_out.reshape(1, -1))
                    elif isinstance(future_refit_len, int) and future_refit_len > 0:
                        train_args = {
                			"transient_length" : shorts_inds[j],
                			"feature_function" : feature_function,
                			"regression" : refit_regression(prior = w_out),
                        "batch_size" : batch_size,
                        "batch_length" : batch_length
                			}
                        if open_loop_horizon is None:
                            train_args["inputs"] = refits_i[j]
                        else:
                            train_args["inputs"] = refits_i[j][:-open_loop_horizon]
                        if library_targets is not None:
                            train_args["target_outputs"] = library_targets[i][
                                :shorts_inds[j] + future_refit_len]
                        sub_ws.append(esn.train(**train_args).weights.reshape(1, -1))
                                      #- w_out.reshape((1, -1)))
                if parameters is not None:
                    sub_params.append(parameters[i])
		
        if esn is not None and incl_final_states:
            r0s = train_result.states[transient[i] - 1 + tshort - 1:]
            for j in range(r0s.shape[0]):
                if(j % start_time_separation == 0):
                    sub_r0s.append(r0s[j].reshape(1, -1))
		
        if esn is not None and incl_initial_states:
            if tshort > 1:
                r0s_resync = train_result.states[transient[i] - 1: - tshort + 1]
            else:
                r0s_resync = train_result.states[transient[i] - 1:]
            for j in range(r0s_resync.shape[0]):
                if(j % start_time_separation == 0):
                    sub_r0s_resync.append(r0s_resync[j].reshape(1, -1))
	
    return MappingRC_TrainData(signals = short_sigs, parameters = sub_params,
							weights = sub_ws, initial_states = sub_r0s_resync,
							final_states = sub_r0s)


#%% Async SM Training and Predictions

def Async_SM_Train(
        pred_esn_args : dict,
        mapper_esn_args : dict,
        library_signals : List[np.ndarray],
        test_length : int,
        transient_length : int,
        sample_separation : int = 1,
        incl_weights : bool = True,
        incl_ri : bool = False,
        incl_rf : bool = False,
        pred_feature : Union[Callable, features.ESNFeatureBase] = features.StatesOnly(),
        mapper_feature : Union[Callable, features.ESNFeatureBase] = features.FinalStateOnly(),
        pred_regression : Callable = regressions.batched_ridge(regularization = 1e-2),
        mapper_regression : Callable = regressions.batched_ridge(),
        mapper_accessible_drives : Union[List[int], int, str] = -1,
        pred_batch_size : int = 1,
        mapper_batch_size : int = 100,
        future_refit_len : int = None,
        refit_regression : Callable = lambda prior: regressions.batched_ridge(prior_guess = prior)
        ):
    
    pred_esn = rc.ESN(**pred_esn_args)
    mapper_esn = rc.ESN(**mapper_esn_args)
    
    map_train_data = extract_MRS_training_data(
        library_signals = library_signals,
        tshort = test_length,
        esn = pred_esn,
        transient = transient_length,
        regression = pred_regression,
        batch_size = pred_batch_size,
        feature_function = pred_feature,
        start_time_separation = sample_separation,
        incl_initial_states = incl_ri,
        incl_final_states = incl_rf,
        incl_weights = incl_weights,
        future_refit_len = future_refit_len,
        refit_regression = refit_regression
        )
    print("Extracted Async SM Training Data")
    
    mapper_train_args = {
        "transient_length" : 0,
        "inputs" : map_train_data.signals,
        "feature_function" : mapper_feature,
        "regression" : mapper_regression,
        "batch_size" : mapper_batch_size,
        "accessible_drives" : mapper_accessible_drives
        }
    
    if incl_weights and incl_ri:
        mapper_train_args["target_outputs"] = map_train_data.weights_and_ri
    elif incl_weights and incl_rf:
        mapper_train_args["target_outputs"] = map_train_data.weights_and_rf
    elif incl_ri:
        mapper_train_args["target_outputs"] = map_train_data.initial_states
    elif incl_rf:
        mapper_train_args["target_outputs"] = map_train_data.final_states
    else:
        mapper_train_args["target_outputs"] = map_train_data.weights
    
    mapper_train_result = mapper_esn.train(**mapper_train_args)
    
    return mapper_train_result

def Async_SM_Predict(
        mapper_train_result : Union[np.ndarray, rc.rescompy.TrainResult],
        pred_esn_args : dict,
        mapper_esn_args : dict,
        test_signals : List[np.ndarray],
        test_length : int,
        incl_ri : bool = False,
        incl_rf : bool = False,
        predict_length : int = None,
        pred_feature : Union[Callable, features.ESNFeatureBase] = features.StatesOnly(),
        mapper_feature : Union[Callable, features.ESNFeatureBase] = features.FinalStateOnly(),
        pred_mapper_function : Callable = rc.default_mapper,
        #pred_mapper_function : Callable = None,
        reduce_predictions : bool = False,
        rmse_only : bool = False,
        resync_signals : List[np.ndarray] = None,
        fixed_pred_weights : np.ndarray = None,
        cyclic_predict : bool = False
        ):
    
    pred_esn = rc.ESN(**pred_esn_args)
    mapper_esn = rc.ESN(**mapper_esn_args)
    
    predictions = []
    
    if resync_signals is not None and len(resync_signals) == 1:
        resync_signals = [resync_signals[0]] * len(test_signals)
    
    for test_ind, test in enumerate(test_signals):
        
        if cyclic_predict:
            
            if predict_length is None:
                target_outputs = test[test_length:]
                predict_length = target_outputs.shape[0]
            elif test.shape[0] >= test_length + predict_length:
                target_outputs = test[test_length: test_length + predict_length]
            else:
                target_outputs = None
            
            mapper_inputs = test[:test_length].copy()
            inputs = np.zeros((predict_length, test.shape[1]))
            states = np.zeros((predict_length, pred_esn.size))
            outputs = np.zeros((predict_length, test.shape[1])) #Alter to allow for outputs of different dim to inputs
            
            for step in range(predict_length):
                mapper_predict_args = {
                    "train_result" : mapper_train_result,
                    "inputs" : mapper_inputs,
                    "initial_state" : np.zeros(mapper_esn.size),
                    "mapper" : drive_mapper
                    }
                if isinstance(mapper_train_result, np.ndarray):
                    mapper_predict_args["feature_function"] = mapper_feature
                mapper_prediction = mapper_esn.predict(**mapper_predict_args)
                
                if incl_ri or incl_rf:
                    w_out = mapper_prediction.reservoir_outputs[
                        0, pred_esn.size:].reshape((pred_feature.feature_size(
                            pred_esn.size, pred_esn.input_dimension), -1))
                    if step == 0:
                        r_0 = mapper_prediction.reservoir_outputs[0, :pred_esn.size]
                else:
                    w_out = mapper_prediction.reservoir_outputs[0].reshape(
                        (pred_feature.feature_size(pred_esn.size, pred_esn.input_dimension), -1))
                    
                predict_args = {
                    "train_result" : w_out,
                    "feature_function" : pred_feature,
                    "predict_length" : 1,
                    "mapper" : pred_mapper_function
                    }
                
                if incl_rf or step > 0:
                    predict_args["initial_state"] = r_0
                elif incl_ri:
                    predict_args["initial_state"] = r_0
                    predict_args["resync_signal"] = test[:test_length]
                elif resync_signals is None:
                    predict_args["resync_signal"] = test[:test_length]
                else:
                    predict_args["resync_signal"] = resync_signals[test_ind]
                    
                update = pred_esn.predict(**predict_args)
                
                inputs[step] = update.inputs[-1]
                outputs[step] = update.reservoir_outputs[-1]
                states[step] = update.reservoir_states[-1]
                
                r_0 = states[step]
                mapper_inputs = np.concatenate((mapper_inputs, outputs[step].reshape(1, -1)))[1:]
                    
            prediction = rc.PredictResult(inputs = inputs,
                                          reservoir_outputs = outputs,
                                          reservoir_states = states,
                                          predict_length = predict_length,
                                          resync_inputs = test[:test_length],
                                          target_outputs = target_outputs)
            
        else:
            mapper_predict_args = {
                "train_result" : mapper_train_result,
                "inputs" : test[:test_length],
                "initial_state" : np.zeros(mapper_esn.size),
                "mapper" : drive_mapper
                }
            if isinstance(mapper_train_result, np.ndarray):
                mapper_predict_args["feature_function"] = mapper_feature
            mapper_prediction = mapper_esn.predict(**mapper_predict_args)
            
            if fixed_pred_weights is None and (incl_ri or incl_rf):
                r_0 = mapper_prediction.reservoir_outputs[0, :pred_esn.size]
                w_out = mapper_prediction.reservoir_outputs[
                    0, pred_esn.size:].reshape((pred_feature.feature_size(
                        pred_esn.size, pred_esn.input_dimension), -1))
            elif fixed_pred_weights is None:
                w_out = mapper_prediction.reservoir_outputs[0].reshape(
                    (pred_feature.feature_size(pred_esn.size, pred_esn.input_dimension), -1))
            else:
                w_out = fixed_pred_weights
                r_0 = mapper_prediction.reservoir_outputs[0, :pred_esn.size]
                
            predict_args = {
            		"train_result" : w_out,
            		"feature_function" : pred_feature,
                "mapper" : pred_mapper_function
            		}
                
            if predict_length is None:
                predict_args["target_outputs"] = test[test_length:]
            elif test.shape[0] >= test_length + predict_length:
                predict_args["target_outputs"] = test[test_length:
                                                      test_length + predict_length]
            else:
                predict_args["predict_length"] = predict_length
                
            if incl_ri:
                predict_args["initial_state"] = r_0
                predict_args["resync_signal"] = test[:test_length]
            elif incl_rf:
                predict_args["initial_state"] = r_0
            elif resync_signals is None:
                predict_args["resync_signal"] = test[:test_length]
            else:
                predict_args["resync_signal"] = resync_signals[test_ind]
            prediction = pred_esn.predict(**predict_args)
        
        #rc.plotter.plot_actual(predict_result = pred_esn.predict(**predict_args))
        
        if rmse_only:
            #print(pred_esn.predict(**predict_args).valid_length())
            #predictions.append(pred_esn.predict(**predict_args).nrmse)
            predictions.append(prediction.nrmse)
        elif reduce_predictions:
            #predictions.append(reduce_prediction(pred_esn.predict(**predict_args)))
            predictions.append(reduce_prediction(prediction))
        else:
            #predictions.append(pred_esn.predict(**predict_args))
            predictions.append(prediction)
        
    return predictions

def Async_SM_Train_and_Predict(
        seed : int,
        run_label : str,
        file_name : str,
        pred_esn_args : dict,
        mapper_esn_args : dict,
        library_signals : List[np.ndarray],
        test_signals : List[np.ndarray],
        test_length : int,
        transient_length : int,
        sample_separation : int = 1,
        incl_ri : bool = False,
        incl_rf : bool = False,
        same_seed : bool = False,
        predict_length : int = None,
        pred_feature : Union[Callable, features.ESNFeatureBase] = features.StatesOnly(),
        mapper_feature : Union[Callable, features.ESNFeatureBase] = features.FinalStateOnly(),
        pred_regression : Callable = regressions.batched_ridge(regularization = 1e-2),
        mapper_regression : Callable = regressions.batched_ridge(),
        mapper_accessible_drives : Union[List[int], int, str] = -1,
        pred_mapper_function : Callable = rc.default_mapper,
        pred_batch_size : int = 1,
        mapper_batch_size : int = 100,
        #pred_mapper_function : Callable = None,
        save : bool = True,
        safe_save : bool = False,
        reduce_predictions : bool = False,
        rmse_only : bool = False,
        method_label : str = "Unsync_SM",
        fixed_pred_weights : np.ndarray = None,
        future_refit_len : int = None,
        refit_regression : Callable = lambda prior: regressions.batched_ridge(prior_guess = prior),
        cyclic_predict : bool = False
        ):
    
    if same_seed:
        mapper_seed = seed
    else:
        mapper_seed = 10000 - seed
    pred_esn_args["seed"] = seed
    mapper_esn_args["seed"] = mapper_seed       
    
    if incl_ri:
        method_label += "_ri"
    elif incl_rf:
        method_label += "_rf"
        
    if cyclic_predict:
        method_label += "_cyc"
        
    if fixed_pred_weights is None:
        incl_weights = True
        method_label += "W"
    else:
        incl_weights = False
    
    start_time = time.time()
    
    mapper_train_result = Async_SM_Train(
        pred_esn_args = pred_esn_args,
        mapper_esn_args = mapper_esn_args,
        library_signals = library_signals,
        test_length = test_length,
        transient_length = transient_length,
        sample_separation = sample_separation,
        incl_ri = incl_ri,
        incl_rf = incl_rf,
        incl_weights = incl_weights,
        pred_feature = pred_feature,
        mapper_feature = mapper_feature,
        pred_regression = pred_regression,
        mapper_regression = mapper_regression,
        mapper_accessible_drives = mapper_accessible_drives,
        pred_batch_size = pred_batch_size,
        mapper_batch_size = mapper_batch_size,
        future_refit_len = future_refit_len,
        refit_regression = refit_regression
        )
    
    train_time = time.time() - start_time
    print("Trained Async SM")
    
    predictions = Async_SM_Predict(
        mapper_train_result = mapper_train_result,
        pred_esn_args = pred_esn_args,
        mapper_esn_args = mapper_esn_args,
        test_signals = test_signals,
        test_length = test_length,
        incl_ri = incl_ri,
        incl_rf = incl_rf,
        predict_length = predict_length,
        pred_feature = pred_feature,
        pred_mapper_function = rc.default_mapper,
        mapper_feature = mapper_feature,
        #pred_mapper_function = pred_mapper_function,
        reduce_predictions = reduce_predictions,
        rmse_only = rmse_only,
        fixed_pred_weights = fixed_pred_weights,
        cyclic_predict = cyclic_predict
        )
        
    if save:
        current_time = time.time() - start_time
            
        experimental_parameters = {
            "test_signal_length" : test_length,
            "sample_separation" : sample_separation,
            "transient_length" : transient_length,
            "predict_length" : predict_length,
            "seed" : seed,
            "map_seed" : mapper_seed,
            "run_time" : current_time,
            "train_timers" : train_time,
            "open_loop" : False,
            "pred_esn_args" : pred_esn_args,
            "map_esn_args" : mapper_esn_args
            }
        optimization_info = {}
        prediction_methods = [method_label]
        saved_predictions = {method_label : predictions}
        feature_functions = {
            "pred_feature" : pred_feature,
            "map_feature" : mapper_feature
            }
        #pred_regularizations = {method_label : pred_regression}
        #map_regularizations = {method_label : mapper_regression}
        
        data = Run_Result(
            run_label = run_label,
            experimental_parameters = experimental_parameters,
            optimization_info = optimization_info,
            prediction_methods = prediction_methods,
            predictions = saved_predictions,
            feature_functions = feature_functions,
            pred_regularizations = None, #pred_regularizations,
            map_regularizations = None, #map_regularizations
            )
        
        run_directory = os.path.join(os.getcwd(), run_label)
        save_loc = os.path.join(run_directory, method_label)
        save_loc = os.path.join(save_loc, str(seed))
        data.save(save_loc = save_loc, safe_save = safe_save,
                  file_name = file_name)
        print("Saved")
        
    else:
        return predictions

#%% Extrapolation and State Look-up

def get_extrapolation_function(
        duration : int,
        method : str = "constant",
        compiled : bool = True,
        extrap_func : Callable = None
        ):
    
    """
    Returns a function that extrapolates a time-series backwards in time
    for a fixed duration.
    """
    
    if method == "constant":
        def extrap_function(signal : np.ndarray):
            return np.concatenate((signal[0][None].repeat(duration, axis = 0),
                                   signal), axis = 0)
    
    #elif method == "linear":
    
    elif extrap_func is not None:
        def extrap_function(signal : np.ndarray):
            return extrap_func(signal, duration)
    
        
    if compiled:
        try:
            extrap_function = numba.jit(nopython = True, fastmath = True)(extrap_function)
        except:
            pass
        
    return extrap_function

#@numba.jit(nopython = True, fastmath = True)
def match_segments(
        test_segment : np.ndarray,
        lookup_signals : List[np.ndarray] #Union[np.ndarray, List[np.ndarray]]
        ):
    
    """
    Finds the segment that best matches a provided test segment among a list of
    of provided lookup signals. The best match is chosen by minimizing the rmse
    over the duration of the test segment, with the error in each component
    normalized by the variance of that component over the lookup signals.
    
    Returns a list whose:
        first entry is the index of the lookup signal containing the best match,
        second entry is the time step within that lookup signal corresponding
            to the start of the best match,
        third entry is the segment from the lookup signals which best matches
            the test segment.
    """    
    
    test_length = test_segment.shape[0]
    if isinstance(lookup_signals, np.ndarray):
        lookup_signals = [lookup_signals]
    
    '''    
    norm = np.var(self.target_outputs, axis=0)
    if np.min(norm) == 0.0:
        msg = "NRMSE is not defined when a component of the " \
                  "target output has 0 variance."
        logging.error(msg)

    else:
        se = np.square(self.reservoir_outputs - self.target_outputs)
        return np.sqrt(np.mean(se/norm, axis = 1))
    '''
    
    matches = np.zeros(len(lookup_signals), dtype = int)
    min_errors = np.zeros(len(lookup_signals))
    for i, signal in enumerate(lookup_signals):
        errors = np.zeros(signal.shape[0] - test_length)
        for step in range(signal.shape[0] - test_length):
            errors[step] = np.sum(np.sqrt(np.mean(np.square(
                signal[step: step + test_length] - test_segment), axis = 0)))
        matches[i] = np.argmin(errors)
        min_errors[i] = np.min(errors)
        
    closest_match = [np.argmin(min_errors), matches[np.argmin(min_errors)],
                     lookup_signals[np.argmin(min_errors)][
                         matches[np.argmin(min_errors)]:
                                 matches[np.argmin(min_errors)] + test_length]]
    
    return closest_match

def get_reservoir_state_matcher(
        lookup_signals : np.ndarray,
        lookup_states : np.ndarray
        ):
    
    """
    Returns a function that takes as argument a test signal and returns the
    reservoir state corresponding to the start of the segment from a 
    collection of lookup signals which best matches the test signal.
    """
    
    #@numba.jit(nopython = True, fastmath = True)
    def get_state(test_segment : np.ndarray):
        
        closest_match = match_segments(
            test_segment = test_segment,
            lookup_signals = lookup_signals
            )
        
        return lookup_states[closest_match[0]][closest_match[1]]
    
    return get_state

#%% Baseline Prediction Methods

def euclidean_distance_predict(
        library:                Library,
        test_library:           Library,
        test_length:            int,
        predict_length:         int = None,
        pred_esn:               rc.ESN = None,
        transient_length:       Union[int, List[int]] = None,
        pred_regression:        Callable = regressions.tikhonov(),
        pred_feature:           Union[Callable, features.ESNFeatureBase] = features.StatesOnly(),
        extra_train_args:       dict = {},
        mapper_function:        Callable = rc.default_mapper,
        rmse_only:              bool = False,
        reduce_predictions:     bool = False
        ):
    
    if hasattr(library, "weights") and pred_esn is None:
        pred_esn = library.esn
    else:
        train_args = {"regression" : pred_regression, "feature_function" : pred_feature}
        for key in extra_train_args.keys():
            train_args[key] = extra_train_args[key]
            
        _ = library.set_library_RCs(
            pred_esn = pred_esn,
            transient_length = transient_length,
            train_args = train_args         
            )
    
    predictions = []

    for test_ind, test in enumerate(test_library.data):
        lib_params = np.array([np.array(parameters) for parameters in
                               library.parameters])
        test_params = np.array(test_library.parameters[test_ind])
        distances = np.sqrt(np.sum(np.square(
            (lib_params - test_params) / np.ptp(lib_params, axis = 0))))
        
        predict_args = {
        		"train_result" : library.weights[np.argmin(distances)],
        		"feature_function" : pred_feature,
            "mapper" : mapper_function
        		}
            
        if predict_length is None:
            predict_args["target_outputs"] = test[test_length:]
        else:
            predict_args["target_outputs"] = test[test_length:
                                                  test_length + predict_length]
        predict_args["resync_signal"] = test[:test_length]
        
        if rmse_only:
            predictions.append(pred_esn.predict(**predict_args).nrmse)
        elif reduce_predictions:
            predictions.append(reduce_prediction(pred_esn.predict(**predict_args)))
        else:
            predictions.append(pred_esn.predict(**predict_args))    
    
    return predictions 
    
def train_batch_and_predict(
        library:                Library,
        test_library:           Library,
        test_length:            int,
        predict_length:         int = None,
        pred_esn:               rc.ESN = None,
        transient_length:       Union[int, List[int]] = None,
        pred_regression:        Callable = regressions.tikhonov(),
        pred_feature:           Union[Callable, features.ESNFeatureBase] = features.StatesOnly(),
        mapper_function:        Callable = rc.default_mapper,
        extra_train_args:       dict = {},
        rmse_only:              bool = False,
        reduce_predictions:     bool = False
        ):
    
    if hasattr(library, "esn") and pred_esn is None:
        pred_esn = library.esn
        
    train_args = {"regression" : pred_regression, "feature_function" : pred_feature}
    for key in extra_train_args.keys():
        train_args[key] = extra_train_args[key]

    train_result = pred_esn.train(
        transient_length = transient_length,
        inputs = library.data,
        **train_args
        )
    
    predictions = []

    for test_ind, test in enumerate(test_library.data):
        
        predict_args = {
        		"train_result" : train_result,
            "mapper" : mapper_function
        		}
            
        if predict_length is None:
            predict_args["target_outputs"] = test[test_length:]
        else:
            predict_args["target_outputs"] = test[test_length:
                                                  test_length + predict_length]
        predict_args["resync_signal"] = test[:test_length]
        
        if rmse_only:
            predictions.append(pred_esn.predict(**predict_args).nrmse)
        elif reduce_predictions:
            predictions.append(reduce_prediction(pred_esn.predict(**predict_args)))
        else:
            predictions.append(pred_esn.predict(**predict_args))    
    
    return predictions 

def train_same_dynamics_and_predict(
        library:                Library,
        test_library:           Library,
        test_length:            int,
        predict_length:         int = None,
        pred_esn:               rc.ESN = None,
        transient_length:       Union[int, List[int]] = None,
        pred_regression:        Callable = None,
        pred_feature:           Union[Callable, features.ESNFeatureBase] = features.StatesOnly(),
        mapper_function:        Callable = rc.default_mapper,
        extra_train_args:       dict = {},
        rmse_only:              bool = False,
        reduce_predictions:     bool = False,
        extrapolation_func:     Callable = None,
        state_matcher:          Callable = None,
        drive_prediction:       bool = True
        ):
    
    if hasattr(library, "esn") and pred_esn is None:
        pred_esn = library.esn
    
    predictions = []
    
    train_args = {"regression" : pred_regression, "feature_function" : pred_feature}
    for key in extra_train_args.keys():
        train_args[key] = extra_train_args[key]

    for test_ind, test in enumerate(test_library.data):
        
        train_result = pred_esn.train(
            transient_length = transient_length,
            inputs = library.data[test_ind],
            **train_args
            )
        
        predict_args = {
            "train_result" : train_result,
            "mapper" : mapper_function
        		}
        
        if drive_prediction:
            predict_args["inputs"] = test[test_length:]
            
        if predict_length is None:
            predict_args["target_outputs"] = test[test_length:]
        else:
            predict_args["target_outputs"] = test[test_length:
                                                  test_length + predict_length]
        
        if extrapolation_func is not None:
            predict_args["resync_signal"] = extrapolation_func(test[:test_length])
            predict_args["initial_state"] = np.zeros(pred_esn.size)
        elif state_matcher is not None:
            predict_args["resync_signal"] = test[:test_length]
            predict_args["initial_state"] = state_matcher(test[:test_length])    
        else:
            predict_args["resync_signal"] = test[:test_length]
            predict_args["initial_state"] = np.zeros(pred_esn.size)
        
        if rmse_only:
            predictions.append(pred_esn.predict(**predict_args).nrmse)
        elif reduce_predictions:
            predictions.append(reduce_prediction(pred_esn.predict(**predict_args)))
        else:
            predictions.append(pred_esn.predict(**predict_args))    
    
    return predictions

def get_vanilla_predictions(
        test_library:           Library,
        test_length:            int,
        predict_length:         int = None,
        pred_esn:               rc.ESN = None,
        transient_length:       Union[int, List[int]] = None,
        pred_regression:        Callable = None,
        pred_feature:           Union[Callable, features.ESNFeatureBase] = features.StatesOnly(),
        mapper_function:        Callable = rc.default_mapper,
        extra_train_args:       dict = {},
        rmse_only:              bool = False,
        reduce_predictions:     bool = False,
        append_resync_inputs:   bool = False
        ):
    
    predictions = []
    
    if transient_length is None:
        transient_length = min(10, test_length // 10)
        
    train_args = {"regression" : pred_regression, "feature_function" : pred_feature}
    for key in extra_train_args.keys():
        train_args[key] = extra_train_args[key]

    for test_ind, test in enumerate(test_library.data):
        
        train_result = pred_esn.train(
            transient_length = transient_length,
            inputs = test[:test_length],
            **train_args
            )
        
        predict_args = {
            "train_result" : train_result,
            "mapper" : mapper_function
        		}
            
        if predict_length is None:
            predict_args["target_outputs"] = test[test_length:]
        else:
            predict_args["target_outputs"] = test[test_length:
                                                  test_length + predict_length]
        
        if rmse_only:
            predictions.append(pred_esn.predict(**predict_args).nrmse)
        elif reduce_predictions:
            predictions.append(reduce_prediction(pred_esn.predict(**predict_args)))
        else:
            predictions.append(pred_esn.predict(**predict_args))    
            
        if append_resync_inputs:
            predictions[-1].resync_inputs = test[:test_length]
    
    return predictions

@numba.jit(nopython = True, fastmath = True)
def idw(
        test_loc:       np.ndarray,
        lib_locs:       np.ndarray,
        rescalings:     np.ndarray
        ):
    
    distances = np.sqrt(np.sum(np.square((lib_locs - test_loc) / rescalings), axis = 1))
    
    return (1. / distances) / np.sum(1./distances)

class LinearNDInterpNearestNDExtrap():
    
    """
    An object to allow for linear interpolation inside the convex hull of 
    points, and nearest-neighbour extrapolation outside the convex hull of
    points.
    """
    
    def __init__(self, points, values, rescale):
        self.funcinterp = LinearNDInterpolator(
            points = points, values = values, rescale = rescale)
        self.funcnearest = NearestNDInterpolator(
            x = points, y = values, rescale = rescale)
        
    def __call__(self, point):
        z = self.funcinterp(point)
        chk = np.isnan(z)
        if chk.any():
            return self.funcnearest(point)
        else:
            return z

def library_interpolate_and_predict(
        library:            Library,
        test_library:       Library,
        test_length:        int,
        predict_length:     int = None,
        pred_esn:           rc.ESN = None,
        transient_length:   Union[int, List[int]] = None,
        pred_regression:    Callable = None,
        pred_feature:       Union[Callable, features.ESNFeatureBase] = features.StatesOnly(),
        extra_train_args:   dict = {},
        mapper_function:    Callable = rc.default_mapper,
        rmse_only:          bool = False,
        reduce_predictions: bool = False,
        interp_type:        str = "linear",
        rescale_axes:       bool = True,
        allow_extrap:       bool = True
        ):
    
    if hasattr(library, "weights") and pred_esn is None:
        pred_esn = library.esn
    else:
        train_args = {"regression" : pred_regression, "feature_function" : pred_feature}
        for key in extra_train_args.keys():
            train_args[key] = extra_train_args[key]
        
        _ = library.set_library_RCs(
            pred_esn = pred_esn,
            transient_length = transient_length,
            train_args = train_args
            )
    
    predictions = []
    
    lib_params = np.array([np.array(parameters) for parameters in
                           library.parameters])
    lib_weights = np.array(library.weights)
    
    if interp_type == "linear":
        if allow_extrap:
            interpolation = LinearNDInterpNearestNDExtrap(
                points = lib_params, values = lib_weights, rescale = rescale_axes)
        else:
            interpolation = LinearNDInterpolator(
                points = lib_params, values = lib_weights, rescale = rescale_axes)
    elif interp_type == "nearest":
        interpolation = NearestNDInterpolator(
            x = lib_params, y = lib_weights, rescale = rescale_axes)
    elif interp_type == "inverse_distance":
        def interpolation(
                test_params : np.ndarray
                ):
            
            if rescale_axes:
                rescalings = np.ptp(lib_params, axis = 0)
            else:
                rescalings = np.ones(lib_params.shape[1])
            
            idw_weights = idw(test_params, lib_params, rescalings)
            
            return np.sum(np.array(
                [lib_weights[i] * idw_weight for i, idw_weight in enumerate(idw_weights)]
                ), axis = 0).reshape((1,) + lib_weights[0].shape)
    elif interp_type == "linear1D":
        interpolation = interp1d(x = lib_params.flatten(), y = lib_weights, axis = 0,
                                 bounds_error = False,
                                 fill_value = "extrapolate"
                                 )       
            
    for test_ind, test in enumerate(test_library.data):
        
        test_params = np.array(test_library.parameters[test_ind])        
        interp_weights = np.squeeze(interpolation(test_params), axis = 0)
        
        predict_args = {
        		"train_result" : interp_weights,
        		"feature_function" : pred_feature,
            "mapper" : mapper_function
        		}
            
        if predict_length is None:
            predict_args["target_outputs"] = test[test_length:]
        else:
            predict_args["target_outputs"] = test[test_length:
                                                  test_length + predict_length]
        predict_args["resync_signal"] = test[:test_length]
        
        if rmse_only:
            predictions.append(pred_esn.predict(**predict_args).nrmse)
        elif reduce_predictions:
            predictions.append(reduce_prediction(pred_esn.predict(**predict_args)))
        else:
            predictions.append(pred_esn.predict(**predict_args))    
    
    return predictions 