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
	'output_dir',
	'model_file','pred_file',
	'title']);

GPS_disp_object = collections.namedtuple('GPS_obj',[
	'gps_ll_vector',
	'gps_xy_vector',
	'gps_obs_vector']);

# Intended for holding either the prior or posterior states of the variables. 
Distributions_object = collections.namedtuple('Distributions_object',[
	'Mag', 'Mag_fixed', 'Mag_std',
	'dx', 'dx_fixed', 'dx_std',
	'dy', 'dy_fixed', 'dy_std',
	'dz', 'dz_fixed', 'dz_std',
	'length','length_fixed', 'length_std',
	'width','width_fixed', 'width_std',
	'strike','strike_fixed', 'strike_std',
	'dip','dip_fixed', 'dip_std',
	'rake','rake_fixed', 'rake_std']);
