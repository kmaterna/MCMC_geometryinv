import numpy as np 
import sys
import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as patches
matplotlib.use('PS')  # forces a certain backend behavior of matplotlib on macosx for pymc3
import pymc3 as pm
import matplotlib.pyplot as plt 
import theano.tensor as tt
import mcmc_collections
import conversion_math
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


def outputs_trace_plots(trace, output_dir):
	# The trace plots
	fig = plt.figure();
	ax = fig.add_subplot(111);
	pm.traceplot(trace);
	plt.savefig('posterior.png');
	plt.close();

	# The corner plot
	fig = plt.figure();
	ax = fig.add_subplot(111);
	pm.pairplot(trace);
	plt.savefig('corner_plot.png');
	plt.close();
	return



def gps_residual_plot(obsfile, predfile, modelfile):
	# NEXT: It will have the fault plane on those figures (conversion math!)
	# NEXT: Scale bar for horizontals and color bar for verticals

	[gps_lon, gps_lat, ux_obs, uy_obs, uz_obs]=io_gps.read_gps_file(obsfile);
	[gps_lon, gps_lat, ux_pred, uy_pred, uz_pred]=io_gps.read_gps_file(predfile);
	ux_obs=np.multiply(ux_obs, 1000);
	uy_obs=np.multiply(uy_obs, 1000);
	uz_obs=np.multiply(uz_obs, 1000);
	ux_pred=np.multiply(ux_pred, 1000);
	uy_pred=np.multiply(uy_pred, 1000);
	uz_pred=np.multiply(uz_pred, 1000);  # converting to mm
	ux_res = np.subtract(ux_obs, ux_pred);
	uy_res = np.subtract(uy_obs, uy_pred);
	uz_res = np.subtract(uz_obs, uz_pred);

	# Read the fault parameters. 
	# Length, width, dip, strike, dx, dy, lon0, lat0
	# Reading the final model parameters
	# But reading lon0 and lat0 from the param section. 
	ifile=open(modelfile,'r');
	for line in ifile:
		if " strike" in line:
			strike=float(line.split()[1]);
		if " length" in line:
			length=float(line.split()[1]);
		if " width" in line:
			width=float(line.split()[1]);
		if " dip" in line:
			dip=float(line.split()[1]);
		if " dx" in line:
			dx=float(line.split()[1]);
		if " dy" in line:
			dy=float(line.split()[1]);	
		if "lon0" in line:
			lon0=float(line.split()[1]);
		if "lat0" in line:
			lat0=float(line.split()[1]);

	# Calculate the box that gets drawn for the fault.
	x0, y0 = dx, dy;
	x1, y1 = conversion_math.add_vector_to_point(x0,y0,length,strike);
	vector_mag = width*np.cos(np.deg2rad(dip));  # how far the middle is displaced from the top downdip from map-view
	xbackbottom, ybackbottom = conversion_math.add_vector_to_point(x0,y0, vector_mag, strike+90);  # strike+90 = downdip direction. 
	xfrontbottom, yfrontbottom = conversion_math.add_vector_to_point(x1,y1, vector_mag, strike+90);  # strike+90 = downdip direction. 

	start_lon, start_lat = conversion_math.xy2lonlat(x0,y0,lon0,lat0);
	end_lon, end_lat = conversion_math.xy2lonlat(x1,y1,lon0,lat0);
	b1lon, b1lat = conversion_math.xy2lonlat(xbackbottom,ybackbottom, lon0, lat0);
	b2lon, b2lat = conversion_math.xy2lonlat(xfrontbottom,yfrontbottom, lon0, lat0);
	
	thin_line_x = [start_lon, end_lon, b2lon, b1lon, start_lon];
	thin_line_y = [start_lat, end_lat, b2lat, b1lat, start_lat];
	thick_line_x = [start_lon, end_lon];
	thick_line_y = [start_lat, end_lat];


	# Plot formatting
	vmin=-6;
	vmax=6;
	cmap = 'jet';
	scale=50;

	fig,axarr = plt.subplots(1,3,sharey=True, figsize=(20, 8), dpi=300);
	axarr[0].scatter(gps_lon, gps_lat, marker='o', s=150, c=uz_obs, vmin=vmin, vmax=vmax, cmap=cmap);
	axarr[0].quiver(gps_lon,gps_lat,ux_obs,uy_obs,linewidths=0.01, edgecolors=('k'),scale=scale);
	axarr[0].set_xlim([np.min(gps_lon),np.max(gps_lon)])
	axarr[0].set_ylim([np.min(gps_lat),np.max(gps_lat)])
	# Annotations
	axarr[0].plot(thick_line_x,thick_line_y,color='red',linewidth=2);
	axarr[0].plot(thin_line_x,thin_line_y,color='red',linewidth=1);
	rect = patches.Rectangle((0.05,0.92),0.3,0.06, facecolor='white', transform=axarr[0].transAxes, edgecolor='black')
	axarr[0].add_patch(rect);
	axarr[0].quiver(0.08, 0.945, 5, 0, transform=axarr[0].transAxes, color='red', scale=scale,zorder=10);
	axarr[0].text(0.2, 0.935, "5 mm", transform=axarr[0].transAxes, color='red',fontsize=16);
	axarr[0].grid(True)
	axarr[0].tick_params(labelsize=16);
	axarr[0].set_title('Observed',fontsize=20);

	axarr[1].scatter(gps_lon, gps_lat, marker='o', s=150, c=uz_pred, vmin=vmin, vmax=vmax,cmap=cmap);
	axarr[1].quiver(gps_lon,gps_lat,ux_pred,uy_pred,linewidths=0.01, edgecolors=('k'),scale=scale);
	axarr[1].set_xlim([np.min(gps_lon),np.max(gps_lon)])
	axarr[1].set_ylim([np.min(gps_lat),np.max(gps_lat)])
	axarr[1].plot(thick_line_x,thick_line_y,color='red',linewidth=2);
	axarr[1].plot(thin_line_x,thin_line_y,color='red',linewidth=1);
	rect = patches.Rectangle((0.05,0.92),0.3,0.06, facecolor='white', transform=axarr[1].transAxes, edgecolor='black')
	axarr[1].add_patch(rect);
	axarr[1].quiver(0.08, 0.945, 5, 0, transform=axarr[1].transAxes, color='red', scale=scale,zorder=10);
	axarr[1].text(0.2, 0.935, "5 mm", transform=axarr[1].transAxes, color='red',fontsize=16);	
	axarr[1].grid(True)
	axarr[1].tick_params(labelsize=16);
	axarr[1].set_title('Modeled',fontsize=20);

	axarr[2].scatter(gps_lon, gps_lat, marker='o', s=150, c=uz_res, vmin=vmin, vmax=vmax,cmap=cmap);
	axarr[2].quiver(gps_lon,gps_lat,ux_res,uy_res,linewidths=0.01, edgecolors=('k'),scale=scale);
	axarr[2].set_xlim([np.min(gps_lon),np.max(gps_lon)])
	axarr[2].set_ylim([np.min(gps_lat),np.max(gps_lat)])
	axarr[2].plot(thick_line_x,thick_line_y,color='red',linewidth=2);
	axarr[2].plot(thin_line_x,thin_line_y,color='red',linewidth=1);	
	rect = patches.Rectangle((0.05,0.92),0.3,0.06, facecolor='white', transform=axarr[2].transAxes, edgecolor='black')
	axarr[2].add_patch(rect);
	axarr[2].quiver(0.08, 0.945, 5, 0, transform=axarr[2].transAxes, color='red', scale=scale,zorder=10);
	axarr[2].text(0.2, 0.935, "5 mm", transform=axarr[2].transAxes, color='red',fontsize=16);
	axarr[2].grid(True)
	axarr[2].tick_params(labelsize=16);
	axarr[2].set_title('Residual',fontsize=20);

	cbarax = fig.add_axes([0.85, 0.08, 0.1, 0.9],visible=False);
	color_boundary_object = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax);
	custom_cmap = cm.ScalarMappable(norm=color_boundary_object, cmap=cmap);
	custom_cmap.set_array(np.arange(vmin, vmax, 1.5));
	cb = plt.colorbar(custom_cmap,aspect=12,fraction=0.2, orientation='vertical');
	cb.set_label('Vertical (m)', fontsize=18);
	cb.ax.tick_params(labelsize=16);	

	fig.savefig('residuals.png')

	return;



def output_manager(params, trace, GPSObject):
	# OUTPUTS (THIS IS GENERAL TO ALL TYPES OF INVERSIONS)
	print("----- RESULTS ------");
	Posteriors = parse_posteriors(params, trace,);
	outputs_trace_plots(trace, params.output_dir);  # This takes a little while

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
	gps_residual_plot(params.gps_input_file, params.pred_file, params.model_file);
	return;



if __name__=="__main__":
	obsfile = "example_gps_6.5_325.txt";
	predfile= "gps_predicted_model.txt";
	modelfile="model_results.txt";
	gps_residual_plot(obsfile, predfile, modelfile);



