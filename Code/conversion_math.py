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

def get_plane_normal(strike, dip):
	# Given a strike and dip, find the orthogonal unit vectors aligned with strike and dip directions that sit within the plane. 
	# The plane normal is their cross product. 
	# Returns in x, y, z coordinates. 
	strike_vector=get_strike_vector(strike); # unit vector
	dip_vector = get_dip_vector(strike, dip); # unit vector
	plane_normal = np.cross(dip_vector, strike_vector);  # dip x strike for outward facing normal, by right hand rule. 
	return plane_normal;

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

def get_fault_center(fault_object, index):
	# Compute the x-y-z coordinates of the center of a fault patch. 
	# Index is the i'th fault patch in this fault_object
	W = get_downdip_width(fault_object.top[index],fault_object.bottom[index],fault_object.dipangle[index]);
	center_z = (fault_object.top[index]+fault_object.bottom[index])/2.0;
	updip_center_x=(fault_object.xstart[index]+fault_object.xfinish[index])/2.0;
	updip_center_y=(fault_object.ystart[index]+fault_object.yfinish[index])/2.0;
	vector_mag = W*np.cos(np.deg2rad(fault_object.dipangle[index]))/2.0;  # how far the middle is displaced downdip from map-view
	center_point = add_vector_to_point(updip_center_x,updip_center_y,vector_mag, fault_object.strike[index]+90);  # strike+90 = downdip direction. 
	center = [center_point[0],center_point[1],center_z]; 
	return center; 

def get_fault_four_corners(fault_object, i):
	# Get the four corners of the object, including updip and downdip. 
	W = get_downdip_width(fault_object.top[i],fault_object.bottom[i],fault_object.dipangle[i]);
	depth       = fault_object.top[i];
	strike      = fault_object.strike[i];
	dip         = fault_object.dipangle[i];

	updip_point0 = [fault_object.xstart[i],fault_object.ystart[i]];
	updip_point1 = [fault_object.xfinish[i],fault_object.yfinish[i]];
	vector_mag = W*np.cos(np.deg2rad(fault_object.dipangle[i]));  # how far the bottom edge is displaced downdip from map-view
	downdip_point0 = add_vector_to_point(fault_object.xstart[i],fault_object.ystart[i],vector_mag, strike+90);  # strike+90 = downdip direction. 
	downdip_point1 = add_vector_to_point(fault_object.xfinish[i],fault_object.yfinish[i], vector_mag, strike+90);

	x_total = [updip_point0[0], updip_point1[0], downdip_point1[0], downdip_point0[0],updip_point0[0]];
	y_total = [updip_point0[1], updip_point1[1], downdip_point1[1], downdip_point0[1],updip_point0[1]];
	x_updip = [updip_point0[0], updip_point1[0]];
	y_updip = [updip_point0[1], updip_point1[1]];
	return [x_total, y_total, x_updip, y_updip];

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



