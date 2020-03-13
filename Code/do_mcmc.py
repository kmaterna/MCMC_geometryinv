# This is the computational guts of the MCMC algorithm. 

import numpy as np 
import mcmc_collections
from okada_wrapper import dc3d0wrapper, dc3dwrapper


# Here we will build a sampling system
# First we convert gps_inputs into the reference frame of the system
# We construct priors and initial vector of parameters from the config file. 
# We might construct a step-size vector. 
# We build an objective function with the right variables
# We sample it many times using MCMC from pymc3


def do_geometry_computation(params, gps_inputs):
	return; 