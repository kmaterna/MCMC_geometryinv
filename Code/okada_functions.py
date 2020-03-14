# Okada math functions
import numpy as np
from okada_wrapper import dc3d0wrapper, dc3dwrapper


def gps_okada(strike, dip, rake, depth, L, W, alpha, strike_slip, dip_slip, x, y):
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