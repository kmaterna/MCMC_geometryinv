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

Priors_object = collections.namedtuple('Priors_object',[
	'Mag', 'Mag_fixed', 
	'dx', 'dx_fixed', 
	'dy', 'dy_fixed',
	'dz', 'dz_fixed',
	'length','length_fixed',
	'width','width_fixed',
	'strike','strike_fixed',
	'dip','dip_fixed',
	'rake','rake_fixed']);

Posterior_object = collections.namedtuple('Posterior_object',[
	'Mag', 'Mag_fixed', 'Mag_std',
	'dx', 'dx_fixed', 'dx_std',
	'dy', 'dy_fixed', 'dy_std',
	'dz', 'dz_fixed', 'dz_std',
	'length','length_fixed', 'length_std',
	'width','width_fixed', 'width_std',
	'strike','strike_fixed', 'strike_std',
	'dip','dip_fixed', 'dip_std',
	'rake','rake_fixed', 'rake_std']);


Faults_object = collections.namedtuple('Faults_object',[
	'xstart','xfinish',
	'ystart','yfinish',
	'rtlat','reverse',
	'strike','dipangle','rake',
	'top','bottom','comment']);