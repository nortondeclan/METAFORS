"""utils.py

The Utilities submodule for rescompy.

This module contains utility functions that are internally used in rescompy and
its submodules.

Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


import logging
from typing import Union, Iterable, Tuple


def check_range(
    arg:         Union[float, int],
    arg_name:    str,
    value:       Union[float, int],
    operator:    str,
    raise_error: bool
    ):
    """The Range-Checking function.
    
    A helper function for checking whether an argument is in a required range
    and raising appropriate warnings and errors.

    Args:
        arg (float, int): The value of the argument.
        arg_name (str): The name of the argument.
        value (float, int): The edge value of the accepted range.
        operator (str): The operator used to compare arg against value.
                        Must be one of 'eq' (equal), 'neq' (not equal), 'l'
                        (less than), 'leq' (less than or equal), 'g' (greater
                        than), or 'geq' (greater than or equal).
        raise_error (bool): If True, will raise an error if the check fails.
                            Otherwise, will raise a warning instead.
    """    

    error = False
    if raise_error:
        str_raise = 'must'
    else:
        str_raise = 'is recommended to'
    if isinstance(arg, Iterable):
        for i in range(len(arg)):
            arg_i = arg[i]
            if operator == 'eq':
                str_operator = 'equal to'
                if not arg_i == value:
                    error = True
            elif operator == 'neq':
                str_operator = 'not equal to'
                if not arg_i != value:
                    error = True
            elif operator == 'leq':
                str_operator = 'less than or equal to'
                if not arg_i <= value:
                    error = True
            elif operator == 'geq':
                str_operator = 'greater than or equal to'
                if not arg_i >= value:
                    error = True
            elif operator == 'l':
                str_operator = 'less than'
                if not arg_i < value:
                    error = True
            elif operator == 'g':
                str_operator = 'greater than'
                if not arg_i > value:
                    error = True
            if error:
                msg = f"{arg_name}[{i}] has value {arg_i} but " \
                          f"{str_raise} be {str_operator} {value}."
                if raise_error:
                    logging.error(msg)
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
    else:            
        if operator == 'eq':
            str_operator = 'equal to'
            if not arg == value:
                error = True
        elif operator == 'neq':
            str_operator = 'not equal to'
            if not arg != value:
                error = True
        elif operator == 'leq':
            str_operator = 'less than or equal to'
            if not arg <= value:
                error = True
        elif operator == 'geq':
            str_operator = 'greater than or equal to'
            if not arg >= value:
                error = True
        elif operator == 'l':
            str_operator = 'less than'
            if not arg < value:
                error = True
        elif operator == 'g':
            str_operator = 'greater than'
            if not arg > value:
                error = True
        if error:
            msg = f"{arg_name} has value {arg} but {str_raise}" \
                    f" be {str_operator} {value}."
            if raise_error:
                logging.error(msg)
                raise ValueError(msg)
            else:
                logging.warning(msg)

def check_shape(
    shape:          Tuple[Union[None, int]],
    required_shape: Tuple[Union[None, int]],
    name:           str,
    ):
    
    for shape_ind in range(len(required_shape)):
        if required_shape[shape_ind] is not None:
            if shape[shape_ind] != required_shape[shape_ind]:
                msg = f"{name} must have shape {required_shape} but " \
                          f"has shape {shape} instead."
                logging.error(msg)
                raise ValueError(msg)