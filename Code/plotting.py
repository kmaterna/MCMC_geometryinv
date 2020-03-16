# Plotting
import numpy as np 
import matplotlib
matplotlib.use('PS')  # forces a certain backend behavior of matplotlib on macosx for pymc3
import pymc3 as pm
import matplotlib.pyplot as plt 
import io_gps




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
	ux_res = np.subtract(ux_obs, ux_pred);
	uy_res = np.subtract(uy_obs, uy_pred);
	uz_res = np.subtract(uz_obs, uz_pred);

	# Getting the fault plane is going to be a trick. 
	# Involving the results file


	vmin=-0.006;
	vmax=0.006;

	fig,axarr = plt.subplots(1,3,sharey=True, figsize=(20, 8), dpi=300);
	axarr[0].scatter(gps_lon, gps_lat, marker='o', s=150, c=uz_obs, vmin=vmin, vmax=vmax);
	axarr[0].quiver(gps_lon,gps_lat,ux_obs,uy_obs,linewidths=0.01, edgecolors=('k'),scale=0.05);
	axarr[0].set_xlim([np.min(gps_lon),np.max(gps_lon)])
	axarr[0].set_ylim([np.min(gps_lat),np.max(gps_lat)])
	axarr[0].plot(-121.0, 34.5,'.r',marker='o');
	axarr[0].grid(True)
	axarr[0].set_title('Observed');

	axarr[1].scatter(gps_lon, gps_lat, marker='o', s=150, c=uz_pred, vmin=vmin, vmax=vmax);
	axarr[1].quiver(gps_lon,gps_lat,ux_pred,uy_pred,linewidths=0.01, edgecolors=('k'),scale=0.05);
	axarr[1].set_xlim([np.min(gps_lon),np.max(gps_lon)])
	axarr[1].set_ylim([np.min(gps_lat),np.max(gps_lat)])
	axarr[1].plot(-121.0, 34.5,'.r',marker='o');
	axarr[1].grid(True)
	axarr[1].set_title('Modeled');

	axarr[2].scatter(gps_lon, gps_lat, marker='o', s=150, c=uz_res, vmin=vmin, vmax=vmax);
	axarr[2].quiver(gps_lon,gps_lat,ux_res,uy_res,linewidths=0.01, edgecolors=('k'),scale=0.05);
	axarr[2].set_xlim([np.min(gps_lon),np.max(gps_lon)])
	axarr[2].set_ylim([np.min(gps_lat),np.max(gps_lat)])
	axarr[2].plot(-121.0, 34.5,'.r',marker='o');
	axarr[2].grid(True)
	axarr[2].set_title('Residual');

	fig.savefig('residuals.png')

	return;




if __name__=="__main__":
	obsfile = "example_gps_6.5_325.txt";
	predfile= "gps_predicted_model.txt";
	modelfile="model_results.txt";
	gps_residual_plot(obsfile, predfile, modelfile);

