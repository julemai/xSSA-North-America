# The Sensitivity of Simulated Streamflow to Individual Hydrologic Processes Across North America

*by Juliane Mai<sup> 1</sup>,  James R. Craig<sup> 1</sup>,  Bryan A. Tolson<sup> 1</sup>, and Richard Arsenault<sup> 2</sup>*<br><br>
*<sup> 1</sup> Dept. of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada.*<br>
*<sup> 2</sup> Dept. of Construction Engineering, École de technologie supérieure, Montreal, QC, Canada.*

## Abstract
Streamflow sensitivity to different hydrologic processes varies in both space and time. This sensitivity is traditionally evaluated for the parameters specific to a given hydrologic model simulating streamflow. In this study, we apply a novel analysis over more than 3000 basins across North America considering a blended hydrologic model structure, which includes not only parametric, but also structural uncertainties. This enables seamless quantification of model process sensitivities and parameter sensitivities across a continuous set of models. It also leads to high-level conclusions about the importance of water cycle components on streamflow predictions, such as quickflow being the most sensitive process for streamflow simulations across the North American continent. The results of the 3000 basins are used to derive an approximation of sensitivities based on physiographic and climatologic data without the need to perform expensive sensitivity analyses. Detailed spatio-temporal inputs and results are shared through an interactive website.

## Usage
The provided scripts enable the users to run their own extended Sobol' sensitivity analysis (xSSA). The forcing data for the basin need to be provided under `data_in/data_obs/<basin-id>`. See the provided examples for details on the format ([Raven](http://raven.uwaterloo.ca) standard forcing format). The physical characteristics such as forest cover and basin-average elevation need to be provided under `data_in/basin_metadata/basin_physical_characteristics.txt` which currently contains the information for the 3826 [HYSETS](https://osf.io/rpc3w/) basins used in this study. The user also needs to provide an executable of the [Raven](http://raven.uwaterloo.ca) modeling framework annd place it under `data_in/data_model/Raven.exe`. To run an analysis for one basin (e.g., USGS gauge station 03054500) use the following commands. The final results will be stored under `data_out/<basin-id>/`.

```
cd scripts

bb='03054500'           # an example basin
nsets='1000'            # number of iterations for analysis
                        # nsets='1000' takes about 22h
                        # number of model runs = (43+2) x nsets + (27+2) x nsets + (11+2) x nsets
tmpdir='/tmp/test-xssa' # a temporary directory; use ${SLURM_TMPDIR} for jobs on super-computers such as Graham

python raven_sa-usgs-canopex.py -i ${bb} -n ${nsets} -t ${tmpdir} -o nc
```

## Website
The results of this study are available on an interactive [webpage](http://www.hydrohub.org/sa_introduction.html#xssa-na). The website includes the download of figures and model setups as well as the visualization of sensitivities of parameters, process options and processes.  

<p align="center">
   <img src="https://github.com/julemai/xSSA-North-America/wiki/images/hydrohub_xssa.png" width="45%" />
</p>

## Creating Plots
This GitHub contains all scripts and data to reproduce the plots in the paper and Supplementary Material. Please see the main plotting script [here](https://github.com/julemai/xSSA-North-America/blob/master/scripts/figures/plot.sh) and select the figures you want to plot. The final figures will then be found in the folder `figures/`. All the data used to produce those figures can be found in the folder `scripts/data/`. 

## Citation

### Journal Publication
Mai, J., Craig, J. R., Tolson, B. A., and R. Arsenault (2021).<br>
The Sensitivity of Simulated Streamflow to Individual Hydrologic Processes Across North America. <br>
*Nature Communications*, 13, 455.<br>
https://doi.org/10.1038/s41467-022-28010-7

See the following publication for details on the introduction of the xSSA method and the Blended Model:<br>
Mai, J., Craig, J. R., and Tolson, B. A. (2020).<br>
Simultaneously determining global sensitivities of model parameters and model structure. <br>
*Hydrol. Earth Syst. Sci.*, 24, 5835–5858.<br>
https://doi.org/10.5194/hess-24-5835-2020

See the following publication for details on calibration of the Blended Model:<br>
Chlumsky, R., Mai, J., Craig, J. R., & Tolson, B. A. (2021). <br>
Simultaneous Calibration of Hydrologic Model Structure and Parameters Using a Blended Model. <br>
*Water Resources Research*, 57(5), e2020WR029229. <br>
http://doi.org/10.1029/2020WR029229

See the following publication for details about the HYSETS database:<br>
Arsenault, R., Brissette, F., Martel, J.-L., Troin, M., Lévesque, G., Davidson-Chaput, J., et al. (2020). <br>
A comprehensive, multisource database for hydrometeorological modeling of 14,425 North American watersheds. <br>
*Scientific Data*, 7, 1–12. <br>
http://doi.org/10.1038/s41597-020-00583-2


### Code and Data Publication
J. Mai, J. R. Craig, B. A. Tolson, and R. Arsenault (2021).<br>
The Sensitivity of Simulated Streamflow to Individual Hydrologic Processes Across North America. <br>
*Zenodo*<br>
[![DOI](https://zenodo.org/badge/224722499.svg)](https://zenodo.org/badge/latestdoi/224722499)
