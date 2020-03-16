# Stress and strain functions
# Conversion functions
# Fault plane geometric functions


import numpy as np 
import math
import haversine


def get_slip_from_mw_area(Mw, L, W, mu):
	# Comes from log M0 = 1.5 * Mw + 16.1 (Hanks and Kanamori, 1979)
	# W and L in km. 
	# mu in Pa
	M0=10**(Mw*1.5+16.1) # in Dyne-cm
	M0=M0*1e-7;  # now in N-m
	area=L*W*1e3*1e3;
	slip=M0/(mu*area);
	return slip;

def get_strike_vector(strike):
	# Returns a unit vector in x-y-z coordinates 
	theta=np.deg2rad(90-strike);
	strike_vector=[np.cos(theta),np.sin(theta),0];
	return strike_vector;

def get_dip_vector(strike,dip):
	# Returns a unit vector in x-y-z coordinates
	downdip_direction_theta = np.deg2rad(-strike);  # theta(strike+90)
	dip_unit_vector_z=np.sin(np.deg2rad(dip))  # the vertical component of the downdip unit vector
	dip_unit_vector_xy = np.sqrt(1-dip_unit_vector_z*dip_unit_vector_z);  # the horizontal component of the downdip unit vector
	dip_vector = [ dip_unit_vector_xy*np.cos(downdip_direction_theta), dip_unit_vector_xy*np.sin(downdip_direction_theta), -dip_unit_vector_z];
	return dip_vector;

def get_vector_magnitude(vector):
	total=0;
	for i in range(len(vector)):
		total=total+vector[i]*vector[i];
		magnitude=np.sqrt(total);
	return magnitude;

def get_strike(deltax, deltay):
	# Returns the strike of a line (in cw degrees from north) given the deltax and deltay in km. 
	slope = math.atan2(deltay,deltax);
	strike= 90-np.rad2deg(slope);
	if strike<0:
		strike=strike+360;
	return strike;

def get_lflat_dip_slip(slip, rake):
	strike_slip = slip * np.cos(np.deg2rad(rake));  # positive sign for convention of left lateral slip
	dip_slip = slip * np.sin(np.deg2rad(rake));
	return strike_slip, dip_slip;

def get_rtlat_dip_slip(slip, rake):
	strike_slip = -slip * np.cos(np.deg2rad(rake));  # negative sign for convention of right lateral slip
	dip_slip = slip * np.sin(np.deg2rad(rake));
	return strike_slip, dip_slip;

def get_strike_length(x0,x1,y0,y1):
	# Just the pythagorean theorem
	length=np.sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) );
	return length;

def get_downdip_width(top,bottom,dip):
	W = abs(top-bottom)/np.sin(np.deg2rad(dip));  # guaranteed to be between 0 and 90
	return W;

def get_top_bottom(center_depth, width,dip):
	# Given a fault, where is the top and bottom? 
	# Width is total downdip width of the fault. 
	top=center_depth-(width/2.0*np.sin(np.deg2rad(dip)));
	bottom=center_depth+(width/2.0*np.sin(np.deg2rad(dip)));
	return top,bottom;

def get_top_bottom_from_top(top_depth, width,dip):
	bottom=top_depth+(width*np.sin(np.deg2rad(dip)));
	return top_depth,bottom;

def add_vector_to_point(x0,y0,vector_mag,vector_heading):
	# Vector heading defined as strike- CW from north.
	theta=np.deg2rad(90-vector_heading);
	x1 = x0 + vector_mag*np.cos(theta);
	y1 = y0 + vector_mag*np.sin(theta);
	return x1, y1;

def get_rake(strike_slip, dip_slip):
	# Positive slip is right lateral, and reverse. 
	# Range is -180 to 180.
	rake = np.rad2deg(math.atan2(dip_slip,strike_slip));
	return rake;

def xy2lonlat(xi,yi,reflon,reflat):
	lat=reflat+( yi*1/111.000 );
	lon=reflon+( xi*1/(111.000*abs(np.cos(np.deg2rad(reflat)))) );
	return lon, lat;

def latlon2xy(loni,lati,lon0,lat0):
	# returns the distance between a point and a reference in km. 
	radius = haversine.distance([lat0,lon0], [lati,loni]);
	bearing = haversine.calculate_initial_compass_bearing((lat0, lon0),(lati, loni))
	azimuth = 90 - bearing;
	x = radius * np.cos(np.deg2rad(azimuth));
	y = radius * np.sin(np.deg2rad(azimuth));
	return [x, y];
