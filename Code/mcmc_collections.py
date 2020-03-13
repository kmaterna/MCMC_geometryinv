# Definitions of objects used in this project

import collections

Params = collections.namedtuple('Params',[
	'gps_input_file',
	'num_iter',
	'burn_in',
	'step_size',
	'mode',
	'mu','alpha',
	'lon0','lat0',
	'Mag','style',
	'dx','dy','dz',
	'length','width',
	'strike','dip','rake',
	'outdir','title']);

GPS_disp_object = collections.namedtuple('GPS_obj',[
	'gps_lon','gps_lat',
	'gps_x','gps_y',
	'ux','uy','uz']);

Faults_object = collections.namedtuple('Faults_object',[
	'xstart','xfinish',
	'ystart','yfinish',
	'rtlat','reverse',
	'strike','dipangle','rake',
	'top','bottom','comment']);

Out_object = collections.namedtuple('Out_object',[
	'x','y',
	'x2d','y2d',
	'u_disp','v_disp','w_disp',
	'u_ll','v_ll','w_ll',
	'source_object','receiver_object',
	'receiver_normal','receiver_shear','receiver_coulomb']);

