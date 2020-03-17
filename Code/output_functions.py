import numpy as np 
import sys
import matplotlib
matplotlib.use('PS')  # forces a certain backend behavior of matplotlib on macosx for pymc3
import pymc3 as pm
import matplotlib.pyplot as plt 
import theano.tensor as tt
import mcmc_collections
import okada_class
import io_gps
import plotting


# UTILITY AND OUTPUT FUNCTIONS

def parse_posterior(name, Variable, trace, outfile):
	ofile=open(outfile,'a');
	if Variable.est_flag==False:
		est = Variable.value;
		std = 0;
		ofile.write(" "+name+": %.2f\n" % (est) );
	else:
		est = np.mean(trace[name]);  # For discrete data
		std = np.std(trace[name]);
		print("EST "+name+": %.2f +/- %.2f" % (est, std) );
		print("-----------");
		ofile.write(" "+name+": %.2f +/- %.2f\n" % (est, std) );
	ofile.close();
	return est, std;

def parse_posteriors(params, trace):
	outfile=params.model_file;
	ofile=open(outfile,'w');
	ofile.write("# ------- RESULTS: ------- #\n");
	ofile.close();
	Mag, Mag_std       = parse_posterior('Mag',params.Mag, trace, outfile);
	dx, dx_std         = parse_posterior('dx',params.dx, trace, outfile);
	dy, dy_std         = parse_posterior('dy',params.dy, trace, outfile);
	dz, dz_std         = parse_posterior('dz',params.dz, trace, outfile);
	length, length_std = parse_posterior('length',params.dz, trace, outfile);
	width, width_std   = parse_posterior('width',params.width, trace, outfile);
	strike, strike_std = parse_posterior('strike',params.strike, trace, outfile);
	dip, dip_std       = parse_posterior('dip',params.dip, trace, outfile);
	rake, rake_std     = parse_posterior('rake',params.rake, trace, outfile);

	Posterior = mcmc_collections.Distributions_object(
		Mag=Mag, Mag_std=Mag_std,
		dx=dx, dx_std=dx_std, dy=dy, dy_std=dy_std, dz=dz, dz_std=dz_std,
		length=length, length_std=length_std, width=width, width_std=width_std,
		strike=strike, strike_std=strike_std, dip=dip, dip_std=dip_std, rake=rake, rake_std=rake_std);
	return Posterior; 

def output_manager(params, trace, GPSObject):
	# OUTPUTS (THIS IS GENERAL TO ALL TYPES OF INVERSIONS)
	print("----- RESULTS ------");
	Posteriors = parse_posteriors(params, trace,);
	# plotting.outputs_trace_plots(trace, params.output_dir);  # This takes a little while

	# Write out all config params into the result file too. 
	print("Writing to %s " % params.model_file);
	ofile=open(params.model_file,'a');
	ofile.write("\n\n#------ CONFIG ------- # \n");
	for fld in params._fields:
		if isinstance(getattr(params,fld),mcmc_collections.Variable):
			ofile.write("%s: %s\n" % (fld, getattr(params,fld).str_value) );  # for the 9 fault parameters
		else:
			ofile.write("%s: %s\n" % (fld, str(getattr(params,fld))) );
	ofile.close();

	# Residual vs observation
	print("Making predicted vector");
	gps_pred_vector = okada_class.calc_gps_disp_vector(
		Posteriors.strike, Posteriors.dip, Posteriors.rake, 
		Posteriors.dx, Posteriors.dy, Posteriors.dz, 
		Posteriors.Mag, Posteriors.length, Posteriors.width, 
		params.mu, params.alpha, GPSObject.gps_xy_vector); 
	PredObject = mcmc_collections.GPS_disp_object(gps_ll_vector=GPSObject.gps_ll_vector, 
		gps_xy_vector=GPSObject.gps_xy_vector, gps_obs_vector=gps_pred_vector);
	io_gps.gps_output_manager(PredObject, params.pred_file);
	print("Plotting gps residuals");
	plotting.gps_residual_plot(params.gps_input_file, params.pred_file, params.model_file);

	return;

