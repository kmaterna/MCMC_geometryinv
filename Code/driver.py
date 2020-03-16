# Driver to run Nonlinear Geometry Inversions in Python

import sys
import argparse, configparser
import mcmc_collections
import io_gps
import do_mcmc

def do_calculation():
	args = welcome_and_parse();
	params = read_config(args.config);
	gps_inputs = io_gps.gps_input_manager(params);
	do_mcmc.do_geometry_computation(params, gps_inputs);
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
	num_iter=configobj.getint('mcmc-config','num_iter');
	burn_in=configobj.getint('mcmc-config','burn_in');
	step_size=configobj.getfloat('mcmc-config','step_size');

	# Compute parameters 
	mode=configobj.get('compute-config','mode');
	mu=configobj.getfloat('compute-config','mu');
	alpha=configobj.getfloat('compute-config','alpha');
	lon0=configobj.getfloat('compute-config','lon0');
	lat0=configobj.getfloat('compute-config','lat0');
	
	# Parameters that might be fixed or might be inversion parameters 
	# Might be floats or might be distributions (ranges)
	Mag=configobj.get('compute-config','Mag');
	style=configobj.get('compute-config','style');
	dx=configobj.get('compute-config','dx');
	dy=configobj.get('compute-config','dy');
	dz=configobj.get('compute-config','dz');
	length=configobj.get('compute-config','length');
	width=configobj.get('compute-config','width');
	strike=configobj.get('compute-config','strike');
	dip=configobj.get('compute-config','dip');
	rake=configobj.get('compute-config','rake');

	Params = mcmc_collections.Params(gps_input_file=gps_input_file, num_iter=num_iter, 
		burn_in=burn_in, step_size=step_size, mode=mode, mu=mu, alpha=alpha, 
		lon0=lon0, lat0=lat0, Mag=Mag, style=style, dx=dx, dy=dy, dz=dz, length=length, 
		width=width, strike=strike, dip=dip, rake=rake, output_dir=output_dir, 
		model_file=model_file, pred_file=pred_file, title=title);

	num_params = 0;
	paramdict = Params._asdict();
	if mode=="SPARSE":
		print("\nComputing in SPARSE Mode.  Should have either 3 or 4 parameters. ");
	elif mode=="MEDIUM":
		print("\nComputing in MEDIUM Mode.  Should have either 6 or 7 parameters. ");
	elif mode=="FULL":
		print("\nComputing in FULL Mode.  Should have either 8 or 9 parameters. ");
	elif mode=="SIMPLE_TEST":
		print("\nComputing a simple example of Bayesian inference for y=f(x). ");
	else:
		print("\nERROR! Mode unrecognized - should be either SPARSE, MEDIUM, or FULL. ");
		print("Exiting");
		sys.exit(0);
	print("Computing parameters: ")
	for key, value in paramdict.items():
		if "(" in str(value) and ")" in str(value):
			print(key, "with prior", value);

	return Params;



if __name__=="__main__":
	do_calculation();
