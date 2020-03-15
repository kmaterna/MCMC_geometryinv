# Class to use external function in PYMC3

import numpy as np 
import pymc3 as pm 
import theano.tensor as tt
import sys
import okada_functions
import conversion_math


# Function to compute GPS displacement vectors from Okada
def calc_gps_disp_vector(strike, dip, rake, fault_x, fault_y, depth, Mw, L, W, mu, alpha, gps_x_vector):
	# This is generally disp = f(model, position)
	# Using Okada's formulation. 
	num_pts=int(len(gps_x_vector)/2);
	gps_x = gps_x_vector[0:num_pts];
	gps_y = gps_x_vector[num_pts:];
	gps_x = [i-fault_x for i in gps_x];
	gps_y = [i-fault_y for i in gps_y];
	slip = conversion_math.get_slip_from_mw_area(Mw, L, W, mu);
	strike_slip, dip_slip = conversion_math.get_lflat_dip_slip(slip, rake);  # the standard definition
	ux, uy, uz = okada_functions.gps_okada(strike, dip, rake, depth, L, W, alpha, strike_slip, dip_slip, gps_x, gps_y);
	gps_disp_vector=np.concatenate((ux, uy, uz));
	return gps_disp_vector;


# define your really-complicated likelihood function that uses loads of external codes
def my_loglike(theta, x, data, sigma):
	"""
	A Gaussian log-likelihood function for a model with parameters given in theta
	Some of these parameters might be values,
	Some might be tt.dvectors
	"""
	dip=theta[0]; rake=theta[1]; W=theta[2];
	# strike=theta[0]; 
	# dip=theta[1]; 
	# rake=theta[2];
	# fault_x=theta[3];
	# fault_y=theta[4];
	# depth=theta[5];
	# Mw=theta[6];
	# L=theta[7];
	# W=theta[8];
	# mu=theta[9];
	# alpha=theta[10];
	strike=325;
	fault_x=0; 
	fault_y=0;
	depth=15;
	Mw=6.5;
	L=23;
	mu=30e9;
	alpha=0.66667;

	model = calc_gps_disp_vector(strike, dip, rake, fault_x, fault_y, depth, Mw, L, W, mu, alpha, x);

	return -(0.5/sigma**2)*np.sum((data - model)**2)




class LogLike(tt.Op):

	"""
	Specify what type of object will be passed and returned to the Op when it is
	called. In our case we will be passing it a vector of values (the parameters
	that define our model) and returning a single "scalar" value (the
	log-likelihood)
	"""
	itypes = [tt.dvector] # expects a vector of parameter values when called
	otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

	def __init__(self, loglike, data, x, sigma):
		"""
		Initialise the Op with various things that our log-likelihood function
		requires. Below are the things that are needed in this particular
		example.

		Parameters
		----------
		loglike:
			The log-likelihood (or whatever) function we've defined
		data:
			The "observed" data that our log-likelihood function takes in
		x:
			The dependent variable (aka 'x') that our model requires
		sigma:
			The noise standard deviation that our function requires.
		"""

		# add inputs as class attributes
		self.likelihood = loglike
		self.data = data
		self.x = x
		self.sigma = sigma

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		# I'm not really sure what this is doing... When do we call it? What is node? 
		# It must be included, however. 
		theta, = inputs  # this will contain my variables

		# call the log-likelihood function
		logl = self.likelihood(theta, self.x, self.data, self.sigma)

		outputs[0][0] = np.array(logl) # output the log-likelihood




