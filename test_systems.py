#%% Import Statements
import numpy as np
from typing import Literal, Union, Generator, List
import numba

from numpy.random import default_rng

#%% Lorenz_63
@numba.jit(nopython = True, fastmath = True)
def _lorenz_deriv(x, sigma, beta, rho, omega):
    
    """
    Returns the derivatives for each component of the Lorenz system.
    """
    
    x_prime = np.zeros((3))
    x_prime[0] = omega * (sigma*(x[1] - x[0]))
    x_prime[1] = omega * (x[0]*(rho - x[2]) - x[1])
    x_prime[2] = omega * (x[0]*x[1] - beta*x[2])

    return x_prime

@numba.jit(nopython=True, fastmath=True)
def _lorenz(sigma, beta, rho, omega, x0, integrate_length, h):
    
    """
    Applies Runge-Kutta integration to the Lorenz system.
    """
    
    x = np.zeros((integrate_length, 3))
    x[0] = x0

    for t in range(integrate_length - 1):
        
        k1 = _lorenz_deriv(x[t], sigma[t], beta[t], rho[t], omega[t])
        k2 = _lorenz_deriv(x[t] + (h/2)*k1, sigma[t], beta[t], rho[t], omega[t])
        k3 = _lorenz_deriv(x[t] + (h/2)*k2, sigma[t], beta[t], rho[t], omega[t])
        k4 = _lorenz_deriv(x[t] + h*k3, sigma[t], beta[t], rho[t], omega[t])

        x[t+1] = x[t] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x

def get_lorenz(
    sigma:            Union[float, np.ndarray]             = 10.0,
    beta:             Union[float, np.ndarray]             = 8/3,
    rho:              Union[float, np.ndarray]             = 28.0,
    omega:            Union[float, np.ndarray]             = 1.,
    x0:               Union[Literal['random'], np.ndarray] = 'random',
    transient_length: int                                  = 5000,
    return_length:    int                                  = 100000,
    h:                float                                = 0.01,
    seed:             Union[int, None, Generator]          = None,
    return_dims:      Union[int, List[int]]                = [0, 1, 2]
    ) -> np.ndarray:
    
    """The Lorenz-getting function.
    
    This function integrates and returns a solution to the Lorenz system,
    obtained with a Runge-Kutta integration scheme.
    
    Args:
        sigma (float, np.ndarray): The first Lorenz parameter.
        beta (float, np.ndarray): The second Lorenz parameter.
        rho (float, np.ndarray): The third Lorenz parameter.
        omega (float, np.ndarray): The time-scale of the Lorenz system.
        x0 (np.ndarray): The initial condition of the Lorenz system.
                         Must be an array of floats of shape (3,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 3).
    """
    
    integrate_length = transient_length + return_length
    if isinstance(sigma, float): sigma = sigma * np.ones(integrate_length)
    if isinstance(beta, float): beta = beta * np.ones(integrate_length)
    if isinstance(rho, float): rho = rho * np.ones(integrate_length)
    if isinstance(omega, float): omega = omega * np.ones(integrate_length)
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)

    # Create a random intiail condition, if applicable.
    if isinstance(x0, str) and x0 == 'random':
        
        # Locations and scales are according to observed means and standard
        # deviations of the Lorenz attractor with default parameters.
        x0_0 = rng.normal(loc=-0.036, scale=8.236)
        x0_1 = rng.normal(loc=-0.036, scale=9.162)
        x0_2 = rng.normal(loc=25.104, scale=7.663)
        x0 = np.array([x0_0, x0_1, x0_2])

    # Integrate the Lorenz system and return.    
    x = _lorenz(sigma, beta, rho, omega, x0, integrate_length, h)
    
    return x[transient_length:, return_dims].copy()

#%% Logistic Map

@numba.jit(nopython=True, fastmath=True)
def _logistic_map(r, x0, integrate_length, r_power, shift):
    
    """
    Applies the Logistic Map x_{n+1} = r ** r_power * x_n * (1 - x_n) + e.
    e usually represents stochastic dynamical noise, but the function call 
    allows for other uses, such as constant, or variable drift.
    """
    
    x = np.zeros(integrate_length)
    x[0] = x0

    for t in range(integrate_length - 1):
        x[t + 1] = r[t] ** r_power * x[t] * (1. - x[t]) + shift[t]

    return x

def get_logistic_map(
    r:                          Union[float, np.ndarray],
    r_power:                    Union[int, float]                    = 1,
    x0:                         Union[Literal['random'], np.ndarray] = 'random',
    transient_length:           int                                  = 5000,
    return_length:              int                                  = 100000,
    dynamical_noise:            float                                = 0,
    observational_noise:        float                                = 0,
    IC_seed:                    Union[int, None, Generator]          = None,
    dynamical_noise_seed:       Union[int, None, Generator]          = None,
    observational_noise_seed:   Union[int, None, Generator]          = None
    ) -> np.ndarray:
    
    """The Logistic Map-getting function.
    
    This function iterates and returns a solution to the Logistic Map.
    
    Args:
        r (float, np.ndarray): The logistic map parameter.
        r_power (float, int): Power of r in the map.
        x0 (np.ndarray): The initial condition of the Logistic Map
                         Must be an array of floats of shape (1,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        dynamical_noise (float): Amplitude of dynamical noise to use in
                                 generating the trajectory. Given in terms of 
                                 the standard-deviation of the true signal. 
                                 Noise is added to the system state at each 
                                 integration step before calculating the next 
                                 iteration.
        observational_noise (float): Amplitude of observational noise to use in
                                     generating the trajectory. Given in terms  
                                     of the standard-deviation of the true 
                                     signal. The trajectory is generated 
                                     without noise, but has noise added to each 
                                     time-step after generation.
        IC_seed (int): An integer for determining the random seed of the random
                       initial state. Is only used if x0 is 'random'.
        dynamical_noise_seed (int): An integer for determining the random seed
                                    used to generate dynamical noise.
        observational_noise_seed (int): An integer for determining the random
                                        seed used to generate observational
                                        noise.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 1).
    """
    
    integrate_length = transient_length + return_length
    if isinstance(r, float):
        r = r * np.ones(integrate_length)

    # Create a random intiail condition, if applicable.
    if x0 == 'random':
        IC_rng = default_rng(IC_seed)
        x0 = np.array(IC_rng.uniform(low = 0, high = 1))
        del IC_rng
    
    dynamical_noise_rng = default_rng(dynamical_noise_seed)
    noise = dynamical_noise_rng.uniform(
        low = -dynamical_noise, high = dynamical_noise, size = integrate_length)
    if dynamical_noise > 0: print(np.ptp(noise))
    del dynamical_noise_rng

    # Integrate the Lorenz system and return.    
    x = _logistic_map(r, x0, integrate_length, r_power, noise)
    
    observational_noise_rng = default_rng(observational_noise_seed)
    noise = observational_noise_rng.uniform(
        low = -observational_noise, high = observational_noise, size = integrate_length)
    if observational_noise > 0: print(np.ptp(noise))
    x += noise
    del observational_noise_rng
    del noise
    
    return x[transient_length:, None].copy()

#%% Gauss/Mouse Map

@numba.jit(nopython=True, fastmath=True)
def _gauss_map(a, b, x0, integrate_length, shift):
    
    """
    Applies the Gauss Map x_{n+1} = exp(-a * x_n ** 2) + b + e.
    e usually represents stochastic dynamical noise, but the function call 
    allows for other uses, such as constant, or variable drift.
    """
            
    x = np.zeros(integrate_length)
    x[0] = x0
    for t in range(integrate_length - 1):
        x[t + 1] = np.exp(- a[t] * x[t] ** 2) + b[t] + shift[t]
            
    return x

def get_gauss_map(
    a:                          Union[float, np.ndarray]             = 5,
    b:                          Union[float, np.ndarray]             = -.5,
    x0:                         Union[Literal['random'], np.ndarray] = 'random',
    transient_length:           int                                  = 5000,
    return_length:              int                                  = 100000,
    dynamical_noise:            float                                = 0,
    observational_noise:        float                                = 0,
    IC_seed:                    Union[int, None, Generator]          = None,
    dynamical_noise_seed:       Union[int, None, Generator]          = None,
    observational_noise_seed:   Union[int, None, Generator]          = None
    ) -> np.ndarray:
    
    """The Logistic Map-getting function.
    
    This function iterates and returns a solution to the Ricker Map.
    
    Args:
        a (float, np.ndarray): The exponential parameter.
        b (float, np.ndarray): The drift parameter.
        x0 (np.ndarray): The initial condition of the Logistic Map
                         Must be an array of floats of shape (1,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        dynamical_noise (float): Amplitude of dynamical noise to use in
                                 generating the trajectory. Given in terms of 
                                 the standard-deviation of the true signal. 
                                 Noise is added to the system state at each 
                                 integration step before calculating the next 
                                 iteration.
        observational_noise (float): Amplitude of observational noise to use in
                                     generating the trajectory. Given in terms  
                                     of the standard-deviation of the true 
                                     signal. The trajectory is generated 
                                     without noise, but has noise added to each 
                                     time-step after generation.
        IC_seed (int): An integer for determining the random seed of the random
                       initial state. Is only used if x0 is 'random'.
        dynamical_noise_seed (int): An integer for determining the random seed
                                    used to generate dynamical noise.
        observational_noise_seed (int): An integer for determining the random
                                        seed used to generate observational
                                        noise.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 1).
    """
    
    integrate_length = transient_length + return_length
    if isinstance(a, float):
        a = a * np.ones(integrate_length)
    if isinstance(b, float):
        b = b * np.ones(integrate_length)

    # Create a random intiail condition, if applicable.
    if x0 == 'random':
        IC_rng = default_rng(IC_seed)
        x0 = np.array(IC_rng.uniform(low = 0, high = 1))
        del IC_rng
    
    dynamical_noise_rng = default_rng(dynamical_noise_seed)
    noise = dynamical_noise_rng.uniform(
        low = -dynamical_noise, high = dynamical_noise, size = integrate_length)
    if dynamical_noise > 0:
        print(np.ptp(noise))
    del dynamical_noise_rng

    # Integrate the Lorenz system and return.    
    x = _gauss_map(a, b, x0, integrate_length, noise)
    
    observational_noise_rng = default_rng(observational_noise_seed)
    noise = observational_noise_rng.uniform(
        low = -observational_noise, high = observational_noise, size = integrate_length)
    if observational_noise > 0: 
        print(np.ptp(noise))
    x += noise
    del observational_noise_rng
    del noise
    
    return x[transient_length:, None].copy()