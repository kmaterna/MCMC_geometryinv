# The purpose of this script is to create GPS data
# That we can invert. 
# The geometry parameters are known. 

import numpy as np
import matplotlib.pyplot as plt 
import random
from okada_wrapper import dc3d0wrapper, dc3dwrapper
import io_gps
import conversion_math


def make_data_dc3d():

	# Quantities that we select. 
	# [strike, dip, rake] = [343, 44, -62];
	[strike, dip, rake] = [325, 80, -180];
	[L, W, depth] = [23, 14, 15]; # comes from Wells and Coppersmith, normal fault, M6.5.
	[lon0, lat0] = [-121.0, 34.6]; # the top back corner of the fault plane
	# and the natural reference point of the system. 
	Mw=6.5;  # magnitude of event
	mu=30e9; # 30 GPa for shear modulus
	alpha=2./3 # usually don't touch this. 

	# Derived quantities
	slip = conversion_math.get_slip_from_mw_area(Mw, L, W, mu);
	strike_slip, dip_slip = conversion_math.get_lflat_dip_slip(slip, rake);
	print("strike slip:",strike_slip,"m (positive is left-lateral)");
	print("dip slip:",dip_slip,"m (positive is reverse)");

	# How to make GPS displacements
	gps_xrange=[-123, -119];
	gps_yrange=[32.5, 36.0];
	number_gps_points = 100; 

	gps_lon=[]; 
	gps_lat=[];
	for i in range(number_gps_points):
		gps_lon.append(random.uniform(gps_xrange[0], gps_xrange[1]));
		gps_lat.append(random.uniform(gps_yrange[0], gps_yrange[1]));

	# Mechanical part. 
	theta=strike-90
	theta=np.deg2rad(theta)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	R2=np.array([[np.cos(-theta),-np.sin(-theta)],[np.sin(-theta),np.cos(-theta)]])

	ux=np.zeros(np.shape(gps_lon));
	uy=np.zeros(np.shape(gps_lon));
	uz=np.zeros(np.shape(gps_lon));

	for k in range(len(gps_lon)):

		# Coordinate transformation relative to top back corner of fault plane.
		[x, y] = conversion_math.latlon2xy(gps_lon[k],gps_lat[k],lon0,lat0);

		#Calculate on rotated position
		xy=R.dot(np.array([[x], [y]]));
		success, u, grad_u = dc3dwrapper(alpha, [xy[0], xy[1], 0.0],
                                 depth, dip, [0, L], [0, W],
                                 [strike_slip, dip_slip, 0.0])
        
		urot=R2.dot(np.array([[u[0]], [u[1]]]))
		ux[k]=urot[0]
		uy[k]=urot[1]
		uz[k]=u[2]  # vertical doesn't rotate

	plt.figure(figsize=(16,16))
	plt.scatter(gps_lon, gps_lat, marker='o', s=150, c=uz, vmin=-0.015, vmax=0.015);
	plt.colorbar();
	plt.quiver(gps_lon,gps_lat,ux,uy,linewidths=0.01, edgecolors=('k'),scale=0.05);
	plt.xlim([np.min(gps_lon),np.max(gps_lon)])
	plt.ylim([np.min(gps_lat),np.max(gps_lat)])
	plt.grid()
	plt.title('strike=%d' % (strike))
	plt.plot(lon0, lat0, '.r',marker='o');
	plt.savefig("Displacement_model_"+str(Mw)+'_'+str(strike)+".png");
	
	outname='output_'+str(Mw)+'_'+str(strike)+'.txt';
	io_gps.write_gps_file(gps_lon, gps_lat, ux, uy, uz, outname);
	return;

if __name__=="__main__":
	make_data_dc3d();
