# MCMC_geometryinv

This code performs a nonlinear inversion for geometry parameters (strike, dip, rake, depth, slip, etc.) of a rectangular slip patch. This problem is also known as a geometry inversion, or a uniform-slip inversion. Inputs are surface displacement data from GPS or InSAR. The code uses Okada's (1992) Green's Functions to compute the elastic displacements. The inversion is solved using a Slice Sampling Markov Chain Monte Carlo algorithm from the PyMC3 library (https://docs.pymc.io/). 


## Code Description: 

### Inputs:
* config file specifying computation constants, prior distributions, and directories/files.
*  GPS displacement data. 

### Outputs: 
* Text files with best-fit parameter values, summaries, and predicted displacements. 
* Plots of observed, predicted, and residual displacements in map view.
* Corner plots for tradeoffs between parameters. 

### Usage: 
* Program is called by command line: "python driver.py config.txt"
* All parameters and behaviors are controlled in the config file. 
* Mode must be [SPARSE, MEDIUM, FULL, SIMPLE_TEST]
	* SPARSE Mode: 3-to-4 parameter inversion. When you have a surface fault trace that you trust, and you just want to invert for dip, rake, and width.
	* MEDIUM Mode: 5-to-6-parameter inversion. In the future.  
	* FULL Mode: 8-to-9-parameter inversion. Full 9 parameters are Strike, Dip, Rake, dx, dy, dz (Depth), Length, Width, and Magnitude. 
	* SIMPLE_TEST Mode: a basic parameter-fitting exercise for some test data and a function y=f(x). 
* Unit conventions: 
	* Strike is defined from 0° to 360°, clockwise from north.
	* Dip is defined from 0° to 90° by the right hand rule.
	* Rake has Left lateral strike slip as positive, and reverse dip slip as positive. 
	* Depth is in km, positive down. Dx and Dy are in km. 
	* Length and Width are in km. 
	* data_noise is in meters
* In all cases, the slip will be computed from Mag by the standard Mw-slip relationship. 
* In MEDIUM MODE, the Length and Width will be computed by Wells and Coppersmith (1994) scaling relationships in order to reduce the number of parameters. If you care which relationship is used, you can set the "style" to be ss, reverse, normal, or None. 
* In the config file, you specify a fixed parameter by writing it out, such as: 
	* strike = 34
* In the config file, you specify priors for an inverted parameter by either: 
    * strike = uniform(0,90)
    * strike = normal(45,20)
* Normal and Uniform prior distributions currently supported

### Helpful Tips
* Your coordinate system's center point (lon0, lat0) should be your best guess for the top back corner of the fault plane (looking along the fault if you look in the direction of the strike). 
* You should give the rake <180° to vary, not 360°. Naturally, the inversion can get stuck in two minima if you let the rake vary through all 360° (ex: both -180° and 180°). 
* Based on my trial and error with simple functions: 
	* The definition of SIGMA (your noise model) is VERY important. You should know what you're doing before playing with that. 
	* The priors are important. I get much better fits to data with uniform priors, obviously.
	* If I put an inappropriate sigma, I can get "well-converging" bad models that don't match EITHER the prior OR the data. They can be unstable and not reproducible. Don't do this. If I put something nearly accurate for sigma, the model fits the data much better. 
* In real usage:
	* The 3-parameter inversion takes about 2 minutes for 2500 samples
	* The 9-parameter inversion should take about 15 minutes, but sometimes various traces get stuck. In one example run, I had to stop it before all 4 traces finished. The results for the one trace were good though. 


### Future work:
* MEDIUM MODE, with Wells and Coppersmith scaling relationships in various "styles"
* InSAR LOS modeling

### Contributing:
Pull requests are welcome! Contributing features, finding bugs, etc. are encouraged.  You could also open an issue to discuss adapting the code further. 

## Specs: 
This code uses Python3, numpy, matplotlib, and pymc3. It requires you to have Ben Thompson's Okada Python wrapper on your pythonpath (https://github.com/tbenthompson/okada_wrapper). The original Okada documentation can be found at http://www.bosai.go.jp/study/application/dc3d/DC3Dhtml_E.html. Installing pymc3 on your system can be tricky, but it's necessary for this code (https://docs.pymc.io/). 

## Examples: ###
### Example 1: Simple Test Function ###

As a start, let's use MCMC to fit a set of data using the function *y=e^(x/a)+mx+c*. The three parameters (*a*, *m*, *c*) can be estimated based on priors that we establish and the data that we provide. The strength of the noise on the data can also influence the convergence of the MCMC algorithm and the numerical values of the results. A simple example will help in developing intuition for the problem. 

I produced sample data with a real function plus some simulated noise: 
* true_slope=0.9;
* true_intercept=1.1;
* true_exponent=2.0;
* noise_strength=0.5.

Then, I defined priors for all three parameters (I could use pm.Normal or pm.Uniform): 
```python
* intercept=pm.Normal('intercept', mu=0, sigma=20); # a wide-ranging prior
* slope = pm.Normal('slope', mu=0, sigma=10); # another wide-ranging prior
* exponent = pm.Normal('exponent',mu=3, sigma=0.5); # another wide-ranging prior
```

I let the MCMC algorithm run with a data noise model of sigma=0.5 (appropriate for this dataset). In one run, the results were: 
* MAP Intercept: 1.05 +/- 0.13 (Actual Intercept: 1.1)
* MAP Slope: 0.95 +/- 0.09 (Actual Slope: 0.90)
* MAP Exponent: 2.03 +/- 0.03 (Actual Exponent: 2.00)

The plots below show that this example reaches convergence and finds a very good match to the true parameters of the model. 

![Data_and_Model_Fit](https://github.com/kmaterna/MCMC_geometryinv/blob/master/Examples/simple_line/example_line.png)
![Parameter_Convergence](https://github.com/kmaterna/MCMC_geometryinv/blob/master/Examples/simple_line/posterior.png)


The tradeoff plots show that all three parameters have strong tradeoffs with one another. 

![Tradeoffs](https://github.com/kmaterna/MCMC_geometryinv/blob/master/Examples/simple_line/corner_plot.png)


### Example 2: Sparse Geometry Inversion with GPS ###

In this more complicated example, I am solving for width, rake, and dip of a rectangular elastic dislocation (SPARSE MODE). I am assuming that strike, depth, dx, dy, dz, and Magnitude of the dislocation are well-constrained and held fixed. I produced displacements at 100 randomly located GPS stations around a hypothetical right-lateral M6.5 earthquake, and I will invert them for the geometry parameters. 

The "true" parameters are:
* width = 14.0 km
* dip = 80.0 degrees
* rake = 160.0 degrees
* noise = 1 mm random

My choices of priors, model domain, numerical parameters, and data noise are specified in the configuration file. 

In this case, after running for a minute or two, the inversion produces a model that fits the data reasonably well: 
* EST width: 13.76 +/- 0.47
* EST dip: 83.11 +/- 1.76
* EST rake: 158.74 +/- 1.48

The residual plot shows the predicted model versus the observed data. For fault annotations on maps, the thicker line is the updip edge of the fault plane. 

![Parameter_Convergence](https://github.com/kmaterna/MCMC_geometryinv/blob/master/Examples/sparse_gps_inversion/Outputs/posterior.png)

![Data_and_Model_Fit](https://github.com/kmaterna/MCMC_geometryinv/blob/master/Examples/sparse_gps_inversion/Outputs/residuals.png)

![Tradeoffs](https://github.com/kmaterna/MCMC_geometryinv/blob/master/Examples/sparse_gps_inversion/Outputs/corner_plot.png)

This is a starting-off point for more involved research questions using a Bayesian framework. 

