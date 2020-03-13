
import numpy as np
import mcmc_collections


def read_gps_file(input_file):
	print("\nReading GPS displacements from %s" % input_file);
	[gps_lon, gps_lat, ux, uy, uz] = np.loadtxt(input_file,unpack=True,skiprows=1);
	GPS_obj = mcmc_collections.GPS_disp_object(gps_lon=gps_lon, gps_lat=gps_lat,
		gps_x=[], gps_y=[], ux=ux, uy=uy, uz=uz);
	return GPS_obj;

def write_gps_file(gps_lon, gps_lat, ux, uy, uz, outname):
	print("Writing gps displacements to file %s " % outname);
	out=np.c_[gps_lon,gps_lat,ux,uy,uz]
	np.savetxt(outname,out,fmt='%.6f',header='x(km),y(km),ux(m),uy(m),uz(m)')	
	return;