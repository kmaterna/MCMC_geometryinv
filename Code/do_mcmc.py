# This is the computational guts of the MCMC algorithm. 

import numpy as np 
from okada_wrapper import dc3d0wrapper, dc3dwrapper
import sys
import matplotlib
matplotlib.use('PS')  # forces a certain backend behavior of matplotlib on macosx for pymc3
import pymc3 as pm 
import matplotlib.pyplot as plt 
import mcmc_collections

# High level bit: 
# Here we will build a sampling system
# First we convert gps_inputs into the reference frame of the system
# We construct priors and initial vector of parameters from the config file. 
# We might construct a step-size vector. 
# We build an objective function with the right variables
# We establish a step size
# We sample it many times using MCMC from pymc3

# Notes: 
# The priors are VERY important. I get much better fits to data with uniform priors, obviously
# The definition of SIGMA on your model is very important too. 
# You should know what you're doing before playing with that. 
# I can get "well-converging" BAD model that doesn't match a prior, and doesn't match the data either. 
# It's not reproducible and sometimes depends on sample number, sometimes shows errors, sometimes not. 
# If I don't put sigma into the model at all, it works better. 
# 
# Next: 
# Corner plots with priors (optional)
# Step size
# Slice Sampling



def parse_prior(variable):
	# Variable could be 'uniform(0, 30)'. 
	# Variable could be 'normal(0,1)'.
	# Returns a pymc3 object. 
	return;



def do_geometry_computation(params, gps_inputs):
	if params.mode=="SIMPLE_TEST":
		dummy_bayesian_computation();
	return;


def dummy_bayesian_computation():
	# If the user would like a dummy calculation, we can do a simple y=f(x) model fitting. 

	# The example function we are giong to fit
	def f(intercept, slope, exponent, x):
		return intercept + slope * x + np.exp(x/exponent);	

	# Establishing the model parameters and making data. 
	true_slope=0.9;
	true_intercept=1.1;
	true_exponent=2.0;
	noise_strength=1.0; # with noise_strength = 1.0, everything works. with 0.2, it doesn't fit well. 
	xdata = np.linspace(0,5,100);
	ydata = [f(true_intercept,true_slope,true_exponent,i) +noise_strength*np.random.randn() for i in xdata];  # the actual function with noise

	# Setting up priors and doing MCMC sampling
	with pm.Model() as model:
		# Defining priors on the variables you want to estimate. 
		# Careful: you can try to estimate a data noise model, sigma
		# But you really should know what you're doing. 
		# This step changes the behavior of the system dramatically. 
		# sigma = pm.Normal('sigma',mu=0.3, sigma=0.1, testval=1.0);  # a data noise model. Somehow it breaks if testval is too small. 
		intercept=pm.Normal('Intercept', mu=0, sigma=20); # a wide-ranging uniform prior. 
		slope = pm.Normal('Slope', mu=0, sigma=10); # another wide-ranging prior
		exponent = pm.Normal('Exponent',mu=3, sigma=0.5); # another wide-ranging prior that doesnt work very well. 
		# Other possible priors:
		# intercept=pm.Uniform('Intercept', -2, 3); # a wide-ranging uniform prior. 
		# exponent = pm.Uniform('beta',1.85, 2.20); # another wide-ranging prior

		# define likelihood
		likelihood = pm.Normal('y', mu=f(intercept, slope, exponent, xdata), observed=ydata)
		
		# Sample the distribution
		trace=pm.sample(4000, tune=500);
		map_estimate = pm.find_MAP(model=model); # returns a dictionary


	# Organize outputs
	est_intercept=map_estimate['Intercept'];
	est_exponent =map_estimate['Exponent'];
	est_slope  =map_estimate['Slope'];
	intercept_std = np.std(trace['Intercept']);
	exponent_std = np.std(trace['Exponent']);
	slope_std = np.std(trace['Slope']);
	est_y = f(est_intercept, est_slope, est_exponent, xdata);
	true_y = f(true_intercept, true_slope, true_exponent, xdata);

	# Plot model vs. data
	plt.figure();
	plt.plot(xdata, ydata,'.',label='Observations');
	plt.plot(xdata, true_y, '-g', linewidth=2, label='Actual');
	plt.plot(xdata, est_y, '-r', linewidth=2, label='MCMC Model');
	plt.legend();
	plt.savefig('example_line.png');
	plt.close();

	# Printing the results: 
	print("----- RESULTS ------");
	print("Actual Intercept: %.2f " % true_intercept);
	print("MAP Intercept: %.2f +/- %.2f" % (est_intercept, intercept_std) );
	print("-----------");
	print("Actual Slope: %.2f " % true_slope);
	print("MAP Slope: %.2f +/- %.2f" % (est_slope, slope_std) );
	print("-----------");
	print("Actual Exponent: %.2f " % true_exponent);
	print("MAP Exponent: %.2f +/- %.2f" % (est_exponent, exponent_std) );
	print("-----------");

	# The trace plots
	fig = plt.figure();
	ax = fig.add_subplot(111);
	pm.traceplot(trace);
	plt.savefig('posterior.png');
	plt.close();

	# The corner plot
	fig = plt.figure();
	ax = fig.add_subplot(111);
	pm.pairplot(trace);
	plt.savefig('corner_plot.png');
	plt.close();

	return; 


