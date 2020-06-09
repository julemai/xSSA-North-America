Thu Nov 19 12:48:38 MST 2015
Steve Markstrom
markstro@usgs.gov


Files in this archive:

#############################################################################
hrusAllConus.zip

Shapefile of all HRUs. The attributes are as follows:

"region" - the NHDplus region (http://www.horizon-systems.com/nhdplus/)
where the HRU is located.

"hru_id_loc" - the local HRU ID within each NHDplus region

"hru_id" - the unique HRU ID within the CONUS domain

These HRU maps come from the "Geospatial Fabric"
(http://wwwbrr.cr.usgs.gov/projects/SW_MoWS/GeospatialFabric.html)

#############################################################################
directory sensScores/

This directory contains the sensitivity score summary files. There is one
subdirectory for each region.

The files in here have been tarred and gzipped.

After a region tar file (e.g. "r01.tar.gzip) has been uncompressed, there are
56 sensitivity score summary files contained within.  The files are named
according to the process and then the objective function. These are CSV files
that can be loaded directly into a spreadsheet or read into R with a one line
command. Each file has the same internal structure:

The header line "V1, ...., V35" each column is one of the 35 calibration
parameters. This is the order of the columns:

adjmix_rain, carea_max, cecn_coef, dday_intcp, dday_slope,
emis_noppt, fastcoef_lin, fastcoef_sq, freeh2o_cap, gwflow_coef,
jh_coef, jh_coef_hru, potet_sublim, ppt_rad_adj, pref_flow_den,
rad_trncf, radj_sppt, radj_wppt, radmax, sat_threshold, slowcoef_lin,
slowcoef_sq, smidx_coef, smidx_exp, soil2gw_max, soil_moist_max,
soil_rechr_max, srain_intcp, ssr2gw_exp, ssr2gw_rate, tmax_allrain,
tmax_allsnow, tmax_index, transp_tmax, wrain_intcp

Each line (after the header) is for an HRU, from HRU 1 to the total number
of HRUs for the region. The value in the row/column is the FAST parameter
sensitivity for the corresponding parameter, HRU, process (from the file
name), and objective function (from the file name).

Stats and lists from these values were made with R programs. Maps were made
with python.

#############################################################################


Snowmelt                --> snowmeltMeanSens.csv
Surface Runoff          --> sroffMeanSens.csv
Infiltration            --> infilMeanSens.csv
Soil moisture           --> soil_moistMeanSens.csv
evapotranspiration      --> hru_actetMeanSens.csv
Interflow               --> ssres_flowMeanSens.csv
Baseflow                --> gwres_flowMeanSens.csv
Runoff                  --> hru_outflowMeanSens.csv




