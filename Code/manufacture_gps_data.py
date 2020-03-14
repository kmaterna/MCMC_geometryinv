# The purpose of this script is to create GPS data
# That we can invert. 
# The geometry parameters are known. 

import numpy as np
import matplotlib
matplotlib.use('PS')  # forces a certain backend behavior of matplotlib on macosx for pymc3
import matplotlib.pyplot as plt 
import random
import okada_functions
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

	gps_lon=[]; gps_lat=[];
	for i in range(number_gps_points):
		gps_lon.append(random.uniform(gps_xrange[0], gps_xrange[1]));
		gps_lat.append(random.uniform(gps_yrange[0], gps_yrange[1]));

	# Coordinate transformation
	gps_x=[]; gps_y=[];
	for i in range(number_gps_points):
		kx, ky = conversion_math.latlon2xy(gps_lon[i], gps_lat[i], lon0, lat0);
		gps_x.append(kx);
		gps_y.append(ky);

	# Mechanical part. 
	# Assumes top back corner of fault plane is located at 0,0
	# Given strike, dip, rake, depth, alpha, strike_slip, and dip_slip...
	# Given vectors of gps_x and gps_y...
	# Returns vectors of gps_u, gps_v, and gps_w.
	ux, uy, uz = okada_functions.gps_okada(strike, dip, rake, depth, L, W, alpha, strike_slip, dip_slip, gps_x, gps_y);

	# Add some random noise
	ux = np.add(ux, 0.001*np.random.randn(len(ux)));
	uy = np.add(uy, 0.001*np.random.randn(len(ux)));
	uz = np.add(uz, 0.002*np.random.randn(len(ux)));

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
