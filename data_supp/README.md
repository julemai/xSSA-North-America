# Supplementary data

## Hype model performance

Calibration period:
	Jan 1981 - Dec 2012 (32 years) (1966-1980 warmup)
    calibrated against KGE
	reported also NSE

Number of stations:
    total:   5338
    overlap:  818   (applying proximity+area filter because station IDs are not given)

File name:
	hype_SMHI_stations.js

Data used in:
    Arheimer, B., Pimentel, R., Isberg, K., Crochemore, L., Andersson, J. C. M., Hasan, A., and Pineda, L.:
    Global catchment modelling using World-Wide HYPE (WWH), open data, and stepwise parameter estimation, 
    Hydrol. Earth Syst. Sci., 24, 535-559, 
    https://doi.org/10.5194/hess-24-535-2020, 2020.

Data downloaded from:
    wget https://wwhype.smhi.se/model-performance/generated/stations.js  --> hype_SMHI_stations.js

Data visualized here:
    https://hypeweb.smhi.se/explore-water/model-performances/model-performance-world/


## mHM model performance

Calibration period:
	Oct 1999 - Sep 2008 (9 years)
    calibrated against NSE
	derived also KGE because streamflow time series available

Number of stations:
    total:   492
    overlap: 162

File name:
	data_supp/rakovec_JGRA_2019/<station-id>/calib_001/output/daily_discharge.out 

Data used in:
    Rakovec, O., Mizukami, N., Kumar, R., Newman, A. J., Thober, S., Wood, A. W., et al. (2019).
	Diagnostic Evaluation of Large‐Domain Hydrologic Models Calibrated Across the Contiguous United States.
	Journal of Geophysical Research: Atmospheres, 124(24), 13991–14007.
	http://doi.org/10.1029/2019JD030767

Data downloaded from:
    https://zenodo.org/record/2630558#.Xsajzy0ZM0o


## VIC model performance

Calibration period:
	Oct 1999 - Sep 2008 (9 years)
    calibrated against NSE
	derived also KGE because streamflow time series available

Number of stations:
    total:   492
    overlap: 162

File name:
	data_supp/mizukami_WRR_2017/<station-id>/output/hcdn_calib_case04.txt
	headers are missing:
    - qsim is first column
	- qobs is second column

Data used in:
    Rakovec, O., Mizukami, N., Kumar, R., Newman, A. J., Thober, S., Wood, A. W., et al. (2019).
	Diagnostic Evaluation of Large‐Domain Hydrologic Models Calibrated Across the Contiguous United States.
	Journal of Geophysical Research: Atmospheres, 124(24), 13991–14007.
	http://doi.org/10.1029/2019JD030767

Data downloaded from:
    https://zenodo.org/record/2630558#.Xsajzy0ZM0o

