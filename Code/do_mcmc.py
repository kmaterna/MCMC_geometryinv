# This is the computational guts of the MCMC algorithm. 

import numpy as np 
import sys
import matplotlib
matplotlib.use('PS')  # forces a certain backend behavior of matplotlib on macosx for pymc3
import pymc3 as pm
import matplotlib.pyplot as plt 
import theano.tensor as tt
import mcmc_collections
import okada_class
import conversion_math
import plotting

# High level bit: 
# Here we will build a sampling system
# We construct priors and initial vector of parameters from the config file. 
# We sample the posterior distribution many times using MCMC from pymc3

# Next: 
# Fix the problem with matplotlib


# THE MAJOR DRIVER FOR THE CALCULATION
def sparse_okada_calculation(params, GPSObject):

	def sparse_loglike(theta, x, data, sigma):
		"""
		Define a Gaussian log-likelihood function for a model with parameters theta. 
		Some of these parameters might be floats, or tt.dvectors.
		"""
		# In SPARSE mode, 3 varaibles are inverted for; the other 6 are held fixed. 
		dip=theta[0]; rake=theta[1]; width=theta[2];

		# These model parameters are specific for the SPARSE case. 
		model = okada_class.calc_gps_disp_vector(params.strike.value, dip, rake, 
			params.dx.value, params.dy.value, params.dz.value, params.Mag.value, 
			params.length.value, width, params.mu, params.alpha, x);

		return -(0.5/sigma**2)*np.sum((data - model)**2)


	# create our Op using the loglike function we just defined. 
	logl = okada_class.LogLike(sparse_loglike, GPSObject.gps_obs_vector, 
		GPSObject.gps_xy_vector, params.data_sigma);

	# The actual Bayesian inference for model parameters. 
	with pm.Model() as model:

		# Getting variables ready. Constants are defined in sparse_loglike. 
		theta = tt.as_tensor_variable([params.dip.gen(), params.rake.gen(), params.width.gen()]);

		# Use a DensityDist (use a lamdba function to "call" the Op). Sample dist. 
		pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
		trace=pm.sample(params.num_iter, tune=params.burn_in);

	return trace;


def dummy_bayesian_computation():
	# If the user would like a dummy calculation, we can do a simple y=f(x) model fitting. 
	# NOTES:
	# Metropolis is slow to converge
	# Slice is fast
	# NUTS is fast

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
		# sigma = pm.Normal('sigma',mu=0.3, sigma=0.1, testval=1.0);  # a data noise model. 
		# Somehow it breaks if testval is too small. 
		# If you just put sigma=constant, you can change the behavior too. 
		# If it matches the noise strength, then you get great convergence. 
		
		intercept=pm.Normal('intercept', mu=0, sigma=20); # a wide-ranging uniform prior. 
		slope = pm.Normal('slope', mu=0, sigma=10); # another wide-ranging prior
		exponent = pm.Normal('exponent',mu=3, sigma=0.5); # another wide-ranging prior that doesnt work very well. 
		# Other possible priors:
		# intercept=pm.Uniform('Intercept', -2, 3); # a wide-ranging uniform prior. 
		# exponent = pm.Uniform('beta',1.85, 2.20); # another wide-ranging prior

		# define likelihood
		likelihood = pm.Normal('y', mu=f(intercept, slope, exponent, xdata), sigma=0.5, observed=ydata)
		
		# Sample the distribution
		method=pm.Slice(vars=[intercept,slope,exponent]);  # slice sampling for all variables
		trace=pm.sample(4000, tune=500, step=method);  # if you remove step=method, then you'll do NUTS

		map_estimate = pm.find_MAP(model=model); # returns a dictionary


	# Organize outputs
	est_intercept=map_estimate['intercept'];
	est_exponent =map_estimate['exponent'];
	est_slope  =map_estimate['slope'];
	intercept_std = np.std(trace['intercept']);
	exponent_std = np.std(trace['exponent']);
	slope_std = np.std(trace['slope']);
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
	# I would also like to write this into a file. 
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

	plotting.outputs_trace_plots(trace,'');

	return; 


