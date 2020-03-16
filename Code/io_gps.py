
import numpy as np
import mcmc_collections
import conversion_math

# Input Manager
def gps_input_manager(params):
	# Reads file, and then performs coodinate transformation. 
	# Returns an object with: 
	# a vector of ux, uy, uz (3*ngps x 1). 
	# a vector of lon, lat (2*ngps x 1).
	# a vector of x, y (2*ngps x 1).

	[gps_lon, gps_lat, ux, uy, uz] = read_gps_file(params.gps_input_file);

	# Doing the geographic conversion and wrapping up the inputs
	gps_x=[]; gps_y=[];
	for i in range(len(gps_lon)):
		kx, ky = conversion_math.latlon2xy(gps_lon[i], gps_lat[i], params.lon0, params.lat0);
		gps_x.append(kx);
		gps_y.append(ky);
	gps_obs_vector = np.concatenate((ux, uy, uz));
	gps_xy_vector   = np.concatenate((gps_x, gps_y));
	gps_ll_vector  = np.concatenate((gps_lon, gps_lat));
	GPSObject = mcmc_collections.GPS_disp_object(gps_ll_vector=gps_ll_vector, gps_xy_vector=gps_xy_vector, 
		gps_obs_vector=gps_obs_vector);
	return GPSObject;


# Output Manager
def gps_output_manager(GPSObject, filename):
	num_pts=int(len(GPSObject.gps_xy_vector)/2);
	gps_lon = GPSObject.gps_ll_vector[0:num_pts];
	gps_lat = GPSObject.gps_ll_vector[num_pts:];
	ux = GPSObject.gps_obs_vector[0:num_pts];
	uy = GPSObject.gps_obs_vector[num_pts:2*num_pts];
	uz = GPSObject.gps_obs_vector[2*num_pts:];
	write_gps_file(gps_lon, gps_lat, ux, uy, uz, filename);
	return;

# ------- PURE INPUT/OUTPUT FUNCTIONS -------- # 

def read_gps_file(input_file):
	print("\nReading GPS displacements from %s" % input_file);
	[gps_lon, gps_lat, ux, uy, uz] = np.loadtxt(input_file,unpack=True,skiprows=1);
	return [gps_lon, gps_lat, ux, uy, uz];

def write_gps_file(gps_lon, gps_lat, ux, uy, uz, outname):
	print("Writing gps displacements to file %s " % outname);
	out=np.c_[gps_lon,gps_lat,ux,uy,uz]
	np.savetxt(outname,out,fmt='%.6f',header='x(km),y(km),ux(m),uy(m),uz(m)')	
	return;

