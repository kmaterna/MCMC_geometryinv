# Driver to run Nonlinear Geometry Inversions in Python

import sys
import argparse, configparser
import mcmc_collections
import io_gps
import do_mcmc
import output_functions
import matplotlib
matplotlib.use('PS')  # forces a certain backend behavior of matplotlib on macosx for pymc3
import pymc3 as pm

def do_calculation():
	args = welcome_and_parse();
	params = read_config(args.config);
	gps_inputs = io_gps.gps_input_manager(params);
	do_geometry_computation(params, gps_inputs);
	return;

def welcome_and_parse():
	print("Welcome to a simple nonlinear inversion for coseismic geometry and slip parameters. ");
	print("This code uses geodetic displacements such as GPS and InSAR, and returns coseismic geometry parameters. ");
	parser = argparse.ArgumentParser(description='Run nonlinear geometry inversion in Python', epilog='\U0001f600 \U0001f600 \U0001f600 ');
	parser.add_argument('config',type=str,help='name of config file for calculation. Required.')
	args = parser.parse_args()
	print("Config file:",args.config);
	return args;

def read_config(config_file):
	print("Reading %s " % config_file);

	configobj=configparser.ConfigParser();
	configobj.optionxform = str # make the config file case-sensitive
	configobj.read(config_file);

	# Basic parameters
	gps_input_file=configobj.get('io-config','gps_input_file');
	output_dir=configobj.get('io-config','output_dir');
	model_file=configobj.get('io-config','model_file');
	pred_file=configobj.get('io-config','pred_file');
	title=configobj.get('io-config','title');
	if output_dir != "":
		model_file=output_dir+"/"+model_file;
		pred_file=output_dir+"/"+pred_file;

	# MCMC parameters
	num_iter=configobj.getint('mcmc-config','num_iter');
	burn_in=configobj.getint('mcmc-config','burn_in');
	step_size=configobj.getfloat('mcmc-config','step_size');

	# Compute Parameters and Other Parameters
	mode=configobj.get('compute-config','mode');
	mu=configobj.getfloat('compute-config','mu');
	alpha=configobj.getfloat('compute-config','alpha');
	lon0=configobj.getfloat('compute-config','lon0');
	lat0=configobj.getfloat('compute-config','lat0');
	style=configobj.get('compute-config','style');	
	data_sigma=configobj.getfloat('compute-config','data_sigma');	
	
	# Parameters that might be fixed or might be inversion parameters 
	# Might be floats or might be distributions (ranges)
	# Returns a Variable object that contains all of this information. 
	def get_prior_generator(variable, name):
		if "uniform" in variable:
			inside_parenths = variable.split('(')[1];
			inside_parenths = inside_parenths.split(')')[0];
			bounds = inside_parenths.split(',');
			def func():
				return pm.Uniform(name, float(bounds[0]), float(bounds[1]))
			return mcmc_collections.Variable(value=variable, str_value=variable,gen=func, est_flag=1);
		elif "normal" in variable:
			inside_parenths = variable.split('(')[1];
			inside_parenths = inside_parenths.split(')')[0];
			bounds = inside_parenths.split(',');
			def func():
				return pm.Normal(name, mu=float(bounds[0]), sigma=float(bounds[1]) );
			return mcmc_collections.Variable(value=variable, str_value=variable,gen=func, est_flag=1);
		else:
			return mcmc_collections.Variable(value=float(variable), str_value=variable,gen=lambda: float(variable), est_flag=0);

	Mag = get_prior_generator(configobj.get('compute-config','Mag'),'Mag');
	dx = get_prior_generator(configobj.get('compute-config','dx'),'dx');
	dy = get_prior_generator(configobj.get('compute-config','dy'),'dy');
	dz = get_prior_generator(configobj.get('compute-config','dz'),'dz');
	length = get_prior_generator(configobj.get('compute-config','length'),'length');
	width = get_prior_generator(configobj.get('compute-config','width'),'width');
	strike = get_prior_generator(configobj.get('compute-config','strike'),'strike');
	dip = get_prior_generator(configobj.get('compute-config','dip'),'dip');
	rake = get_prior_generator(configobj.get('compute-config','rake'),'rake');	

	Params = mcmc_collections.Params(gps_input_file=gps_input_file, num_iter=num_iter, 
		burn_in=burn_in, step_size=step_size, mode=mode, mu=mu, alpha=alpha, 
		lon0=lon0, lat0=lat0, Mag=Mag, style=style, dx=dx, dy=dy, dz=dz, length=length, 
		width=width, strike=strike, dip=dip, rake=rake, data_sigma=data_sigma, 
		output_dir=output_dir, model_file=model_file, pred_file=pred_file, title=title);

	num_params = 0;
	paramdict = Params._asdict();
	if mode=="SPARSE":
		print("\nComputing in SPARSE Mode.  Should have either 3 or 4 inverted parameters. ");
	elif mode=="MEDIUM":
		print("\nComputing in MEDIUM Mode.  Should have either 6 or 7 inverted parameters. ");
	elif mode=="FULL":
		print("\nComputing in FULL Mode.  Should have either 8 or 9 inverted parameters. ");
	elif mode=="SIMPLE_TEST":
		print("\nComputing a simple example of Bayesian inference for y=f(x). ");
	else:
		print("\nERROR! Mode unrecognized - should be either SPARSE, MEDIUM, or FULL. ");
		print("Exiting");
		sys.exit(0);
	print("Computing parameters: ")
	for key, value in paramdict.items():
		if "(" in str(value) and ")" in str(value):
			print(key, "with prior", str(value.value));

	return Params;

def do_geometry_computation(params, GPSObject):
	if params.mode=="SIMPLE_TEST":
		do_mcmc.dummy_bayesian_computation();
	if params.mode=="SPARSE":
		trace = do_mcmc.sparse_okada_calculation(params, GPSObject);
		output_functions.output_manager(params, trace, GPSObject);
	if params.mode=="FULL":
		trace = do_mcmc.full_okada_calculation(params, GPSObject);
		output_functions.output_manager(params, trace, GPSObject);		
	return;



if __name__=="__main__":
	do_calculation();
