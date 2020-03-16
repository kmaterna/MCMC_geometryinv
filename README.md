# MCMC_geometryinv

This code performs a nonlinear inversion for geometry parameters (strike, dip, rake, depth, slip, etc.) of a rectangular patch of slip given surface displacement data. It uses Okada's (1992) Green's Functions to compute the elastic displacements. The inversion is solved using a Markov Chain Monte Carlo algorithm. 


### Capabilities: ###
* Reads GPS displacement data. 
* Plots observed, predicted, and residual displacements.
* Corner plots for tradeoffs between parameters. 
* Produces output tables and plots. 
* Mode FULL: 8-to-9-parameter inversion. 
* Mode MEDIUM: 5-to-6-parameter inversion. 
* Mode SPARSE: 3-to-4 parameter inversion. 

### Notes: ###
* step_size might be kind of complicated in terms of units... 
* strike, rake, and dip are specified in the usual convention, in degrees. 
* In the config file, specify a fixed parameter by writing it out: 
    * strike = 34
* In the config file, specify an inverted parameter by either: 
    * strike = uniform(0,90)
    * strike = normal(45,20)
* In all cases, the slip will be computed from the magnitude using a Wells and Coppersmith (1994) relationship. In some cases, the Length and Width will also be computed in the same way. If you care which relationship is used, you can set the "style" to be ss, reverse, normal, or None. 
* Supports normal and uniform distributions as priors right now
* Your center point should be your best guess for the top back corner of the fault plane. 

### Future work: ###
* Other features. 

### Specs: ###
This code uses Python3, numpy, matplotlib, pymc3, and pygmt. It requires you to have Ben Thompson's Okada Python wrapper on your pythonpath (https://github.com/tbenthompson/okada_wrapper). The original Okada documentation can be found at http://www.bosai.go.jp/study/application/dc3d/DC3Dhtml_E.html. The elastic parameters mu and lamda are set in configure_calc.py. By convention, right lateral strike slip is positive, and reverse dip slip is positive. Strike is defined from 0 to 360 degrees, clockwise from north; dip is defined from 0 to 90 degrees by the right hand rule.

Example calculation: 

