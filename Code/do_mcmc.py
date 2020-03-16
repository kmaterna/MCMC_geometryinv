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
import io_gps
import plotting

# High level bit: 
# Here we will build a sampling system
# We construct priors and initial vector of parameters from the config file. 
# We sample the posterior distribution many times using MCMC from pymc3

# Next: 
# 1. Make a nice markdown with figures. 
# Code review everything
# Fix the problem with matplotlib


# UTILITY FUNCTIONS

def parse_priors(params):
	Mag, Mag_fixed       = parse_prior(params.Mag, 'Mag');
	dx, dx_fixed         = parse_prior(params.dx, 'dx');
	dy, dy_fixed         = parse_prior(params.dy, 'dy');
	dz, dz_fixed         = parse_prior(params.dz, 'dz');
	length, length_fixed = parse_prior(params.length, 'length');
	width, width_fixed   = parse_prior(params.width, 'width');
	strike, strike_fixed = parse_prior(params.strike, 'strike');
	dip, dip_fixed       = parse_prior(params.dip, 'dip');
	rake, rake_fixed     = parse_prior(params.rake, 'rake');
	myPriors = mcmc_collections.Distributions_object(
		Mag=Mag, Mag_fixed=Mag_fixed, Mag_std=0, 
		dx=dx, dx_fixed=dx_fixed, dx_std=0,
		dy=dy, dy_fixed=dy_fixed, dy_std=0,
		dz=dz, dz_fixed=dz_fixed, dz_std=0,
		length=length, length_fixed=length_fixed, length_std=0,
		width=width, width_fixed=width_fixed, width_std=0,
		strike=strike, strike_fixed=strike_fixed, strike_std=0,
		dip=dip, dip_fixed=dip_fixed, dip_std=0,
		rake=rake, rake_fixed=rake_fixed, rake_std=0);
	print("My priors:");
	print(myPriors);
	return myPriors;

def parse_prior(variable, name):
	# Takes a string. 
	# Returns a pymc3 object or a float. 
	# Parse the parameters in params
	# Example Inputs: 325, uniform(0, 30), normal(0,1)
	# Example Outputs: 325, pm.Uniform('dip',0,30), pm.Normal('dx',0,1). 
	if "uniform" in variable:
		fixed_flag=False;  # this variable is being estimated by MCMC
		inside_parenths = variable.split('(')[1];
		inside_parenths = inside_parenths.split(')')[0];
		bounds = inside_parenths.split(',');
		print("ESTIMATING "+name);
		print("Uniform Prior for "+name+": ", bounds);
		value = pm.Uniform(name, float(bounds[0]), float(bounds[1]) );
	elif "normal" in variable:
		fixed_flag=False;
		inside_parenths = variable.split('(')[1];
		inside_parenths = inside_parenths.split(')')[0];
		bounds = inside_parenths.split(',');
		print("ESTIMATING "+name);
		print("Normal Prior for "+name+": ", bounds);
		value = pm.Normal(name, mu=float(bounds[0]), sigma=float(bounds[1]) );
	else:
		value=float(variable);
		fixed_flag=True;
	return value, fixed_flag;

def parse_posterior(name, prior_value, prior_fixed, map_estimate, trace, outfile):
	ofile=open(outfile,'a');
	if prior_fixed==True:
		est = prior_value;
		std = 0;
		ofile.write("FIX "+name+": %.2f\n" % (est) );
	else:
		# est = map_estimate[name];  # Thinking this might not work for discrete data anymore... 
		est = np.mean(trace[name]);  # For discrete data
		std = np.std(trace[name]);
		print("MAP "+name+": %.2f +/- %.2f" % (est, std) );
		print("-----------");
		ofile.write("EST "+name+": %.2f +/- %.2f\n" % (est, std) );
	ofile.close();
	return est, std;

def parse_posteriors(myPriors, map_estimate, trace, outfile):
	ofile=open(outfile,'w');
	ofile.write("# ------- RESULTS: ------- #\n");
	ofile.close();
	Mag, Mag_std       = parse_posterior('Mag',myPriors.Mag, myPriors.Mag_fixed, map_estimate, trace, outfile);
	dx, dx_std         = parse_posterior('dx',myPriors.dx, myPriors.dx_fixed, map_estimate, trace, outfile);
	dy, dy_std         = parse_posterior('dy',myPriors.dy, myPriors.dy_fixed, map_estimate, trace, outfile);
	dz, dz_std         = parse_posterior('dz',myPriors.dz, myPriors.dz_fixed, map_estimate, trace, outfile);
	length, length_std = parse_posterior('length',myPriors.length, myPriors.length_fixed, map_estimate, trace, outfile);
	width, width_std   = parse_posterior('width',myPriors.width, myPriors.width_fixed, map_estimate, trace, outfile);
	strike, strike_std = parse_posterior('strike',myPriors.strike, myPriors.strike_fixed, map_estimate, trace, outfile);
	dip, dip_std       = parse_posterior('dip',myPriors.dip, myPriors.dip_fixed, map_estimate, trace, outfile);
	rake, rake_std     = parse_posterior('rake',myPriors.rake, myPriors.rake_fixed, map_estimate, trace, outfile);

	Posterior = mcmc_collections.Distributions_object(
		Mag=Mag, Mag_fixed=myPriors.Mag_fixed, Mag_std=Mag_std,
		dx=dx, dx_fixed=myPriors.dx_fixed, dx_std=dx_std,
		dy=dy, dy_fixed=myPriors.dy_fixed, dy_std=dy_std,
		dz=dz, dz_fixed=myPriors.dz_fixed, dz_std=dz_std,
		length=length, length_fixed=myPriors.length_fixed, length_std=length_std,
		width=width, width_fixed=myPriors.width_fixed, width_std=width_std,
		strike=strike, strike_fixed=myPriors.strike_fixed, strike_std=strike_std,
		dip=dip, dip_fixed=myPriors.dip_fixed, dip_std=dip_std,
		rake=rake, rake_fixed=myPriors.rake_fixed, rake_std=rake_std);
	return Posterior; 

def output_manager(params, myPriors, map_estimate, trace, GPSObject):
	# OUTPUTS (THIS IS GENERAL TO ALL TYPES OF INVERSIONS)
	print("----- RESULTS ------");
	Posteriors = parse_posteriors(myPriors, map_estimate, trace, params.model_file);
	plotting.outputs_trace_plots(trace, params.output_dir);

	# Write out all config params into the result file too. 
	ofile=open(params.model_file,'a');
	ofile.write("\n\n#------ CONFIG ------- # \n");
	for fld in params._fields:
		ofile.write("%s: %s\n" % (fld, str(getattr(params,fld))) );
	ofile.close();

	# Residual vs observation
	gps_pred_vector = okada_class.calc_gps_disp_vector(
		Posteriors.strike, Posteriors.dip, Posteriors.rake, 
		Posteriors.dx, Posteriors.dy, Posteriors.dz, 
		Posteriors.Mag, Posteriors.length, Posteriors.width, 
		params.mu, params.alpha, GPSObject.gps_xy_vector); 

	PredObject = mcmc_collections.GPS_disp_object(gps_ll_vector=GPSObject.gps_ll_vector, 
		gps_xy_vector=GPSObject.gps_xy_vector, gps_obs_vector=gps_pred_vector);
	io_gps.gps_output_manager(PredObject, params.pred_file);
	plotting.gps_residual_plot(params.gps_input_file, params.pred_file, params.model_file);

	return;



# THE MAJOR DRIVER FOR THE CALCULATION
def do_geometry_computation(params, GPSObject):
	if params.mode=="SIMPLE_TEST":
		dummy_bayesian_computation();
	if params.mode=="SPARSE":
		myPriors, map_estimate, trace = sparse_okada_calculation(params, GPSObject);
		output_manager(params, myPriors, map_estimate, trace, GPSObject);
	return;


def sparse_okada_calculation(params, GPSObject):

	# define your likelihood function that uses external codes
	def sparse_loglike(theta, x, data, sigma):
		"""
		A Gaussian log-likelihood function for a model with parameters theta. 
		Some of these parameters might be floats, if given in the config file. 
		Some might be tt.dvectors, if specified as distributions in the config file. 
		"""
		# In SPARSE mode, 3 varaibles are inverted for; the other 6 are held fixed. 
		dip=theta[0]; rake=theta[1]; width=theta[2];

		# These model parameters are specific for the SPARSE case. 
		model = okada_class.calc_gps_disp_vector(float(params.strike), dip, rake, 
			float(params.dx), float(params.dy), float(params.dz), float(params.Mag), 
			float(params.length), width, params.mu, params.alpha, x);

		return -(0.5/sigma**2)*np.sum((data - model)**2)


	# create our Op using the loglike function we just defined. 
	logl = okada_class.LogLike(sparse_loglike, GPSObject.gps_obs_vector, 
		GPSObject.gps_xy_vector, params.data_sigma);


	# The actual Bayesian inference for model parameters. 
	with pm.Model() as model:

		myPriors = parse_priors(params);

		# Getting variables ready. Constants are defined in sparse_loglike. 
		theta = tt.as_tensor_variable([myPriors.dip, myPriors.rake, myPriors.width]);

		# use a DensityDist (use a lamdba function to "call" the Op)
		pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

		# Sample the distribution
		trace=pm.sample(params.num_iter, tune=params.burn_in);

		map_estimate = pm.find_MAP(model=model); # returns a dictionary. 
		# I have found the MAP estimate is not so useful in real cases. 

	return myPriors, map_estimate, trace;


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

	outputs_traces(trace,'');

	return; 


