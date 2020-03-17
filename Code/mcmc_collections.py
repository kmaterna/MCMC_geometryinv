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
	'data_sigma',
	'output_dir',
	'model_file','pred_file',
	'title']);

Variable = collections.namedtuple('Variable',[
	'value','str_value','gen','est_flag']);

GPS_disp_object = collections.namedtuple('GPS_obj',[
	'gps_ll_vector',
	'gps_xy_vector',
	'gps_obs_vector']);

# Intended for holding either the prior or posterior states of the variables. 
Distributions_object = collections.namedtuple('Distributions_object',[
	'Mag', 'Mag_std',
	'dx', 'dx_std',
	'dy', 'dy_std',
	'dz', 'dz_std',
	'length', 'length_std',
	'width', 'width_std',
	'strike', 'strike_std',
	'dip', 'dip_std',
	'rake', 'rake_std']);

Faults_object = collections.namedtuple('Faults_object',[
	'xstart','xfinish',
	'ystart','yfinish',
	'rtlat','reverse',
	'strike','dipangle','rake',
	'top','bottom','comment']);
