# Class to use external function in PYMC3

import numpy as np 
import pymc3 as pm 
import theano.tensor as tt
import sys
from okada_wrapper import dc3d0wrapper, dc3dwrapper
import conversion_math

# PART 1: A CLASS DEFINITION. 
# This class is necessary to wrap an external function (like Okada)
# for performing pymc3 sampling. 
# It contains loglikelihood distributions, a parameter vector, and observed data. 
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



# PART 2: MECHANICAL FUNCTION FOR OKADA DISLOCATIONS
# The inputs to these functions tend to be just floats and vectors. 

# Function to compute elastic displacement vectors using the parameters estimated in the 
# general solution to this 9-parameter geometry inversion. 
def calc_gps_disp_vector(strike, dip, rake, fault_x, fault_y, depth, Mw, L, W, mu, alpha, gps_x_vector):
	# This is generally disp = f(model, position)
	# Model contains up to 9 free parameters. 
	# Elastic dislocation returned as a vector. 
	num_pts=int(len(gps_x_vector)/2);
	gps_x = gps_x_vector[0:num_pts];
	gps_y = gps_x_vector[num_pts:];
	gps_x = [i-fault_x for i in gps_x];
	gps_y = [i-fault_y for i in gps_y];  # implementing dx and dy
	slip = conversion_math.get_slip_from_mw_area(Mw, L, W, mu);
	strike_slip, dip_slip = conversion_math.get_lflat_dip_slip(slip, rake);  # the standard definition
	ux, uy, uz = okada_at_origin(strike, dip, rake, depth, L, W, alpha, strike_slip, dip_slip, gps_x, gps_y);
	gps_disp_vector=np.concatenate((ux, uy, uz));
	return gps_disp_vector;


# After implementing the dx and dy stage... we can compute dislocations from the origin. 
def okada_at_origin(strike, dip, rake, depth, L, W, alpha, strike_slip, dip_slip, x, y):
	# Mechanical part. 
	# Assumes top back corner of fault plane is located at 0,0
	# Given strike, dip, rake, depth, length, width, alpha, strike_slip, and dip_slip...
	# Given vectors of positions x and y...
	# Returns vectors of displacements u, v, w.
	theta=strike-90
	theta=np.deg2rad(theta)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	R2=np.array([[np.cos(-theta),-np.sin(-theta)],[np.sin(-theta),np.cos(-theta)]])

	ux=np.zeros(np.shape(x));
	uy=np.zeros(np.shape(x));
	uz=np.zeros(np.shape(x));

	for k in range(len(x)):

		#Calculate on rotated position
		xy=R.dot(np.array([[x[k]], [y[k]]]));
		success, u, grad_u = dc3dwrapper(alpha, [xy[0], xy[1], 0.0],
                                 depth, dip, [0, L], [0, W],
                                 [strike_slip, dip_slip, 0.0])
        
		urot=R2.dot(np.array([[u[0]], [u[1]]]))
		ux[k]=urot[0]
		uy[k]=urot[1]
		uz[k]=u[2]  # vertical doesn't rotate
	return ux, uy, uz;	
	
