#!/usr/bin/env python

# Copyright 2019 Juliane Mai - juliane.mai(at)uwaterloo.ca
#
# License
# This file is part of Juliane Mai's personal code library.
#
# Juliane Mai's personal code library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Juliane Mai's personal code library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with Juliane Mai's personal code library.  If not, see <http://www.gnu.org/licenses/>.
#
# run with:
#     source ../env-3.5/bin/activate
#     run compare_performance_to_other_models.py -p compare-hype-vic-mhm-blended.pdf

from __future__ import print_function

# -------------------------------------------------------------------------
# General settings
#
dobw      = False # True: black & white
docomp    = True  # True: Print classification on top of modules
dosig     = False # True: add signature to plot
dolegend  = False # True: add legend to each subplot
doabc     = True  # True: add subpanel numbering
dotitle   = True  # True: add catchment titles to subpanels

"""

          Model       Cal-Period              Cal-criteria    Val-Period              Data reported
We        Blended     Jan 1991 - Dec 2010     NSE             Jan 1971 - Dec 1990     Calibration NSE; derived KGE
SMHI      Hype        Jan 1981 - Dec 2012     KGE             None                    Calibration KGE; derived NSE
NCAR      VIC         Oct 1999 - Sep 2008     NSE             Oct 1989 - Sep 1999     Calibration NSE
UFZ       mHM         Oct 1999 - Sep 2008     NSE             Oct 1989 - Sep 1999     Calibration NSE


CAMELS    SAC-SMA     Jan 1980 - Dec 2014     NSE             None                    Calibration NSE

Kratzert  VIC_basin   Oct 1999 - Sep 2008     KGE?            Oct 1989 - Sep 1999     Validation  NSE
Kratzert  mHM_basin   Oct 1999 - Sep 2008     KGE?            Oct 1989 - Sep 1999     Validation  NSE
Kratzert  SAC-SMA     Oct 1999 - Sep 2008     KGE?            Oct 1989 - Sep 1999     Validation  NSE
Kratzert  FUSE 902    Oct 1999 - Sep 2008     KGE?            Oct 1989 - Sep 1999     Validation  NSE
Kratzert  HBV_upper   Oct 1999 - Sep 2008     KGE?            Oct 1989 - Sep 1999     Validation  NSE
Kratzert  LSTM_NSE    Oct 1999 - Sep 2008     KGE?            Oct 1989 - Sep 1999     Validation  NSE

-----------------------
HYPE (SMHI)
-----------------------
Compares calibration results of blended model with calibration results presented in:

    Arheimer, B., Pimentel, R., Isberg, K., Crochemore, L., Andersson, J. C. M., Hasan, A., and Pineda, L.:
    Global catchment modelling using World-Wide HYPE (WWH), open data, and stepwise parameter estimation, 
    Hydrol. Earth Syst. Sci., 24, 535-559, 
    https://doi.org/10.5194/hess-24-535-2020, 2020.

and shown on:

    https://hypeweb.smhi.se/explore-water/model-performances/model-performance-world/

and downloaded (afterwards formated to JSON):

    wget https://wwhype.smhi.se/model-performance/generated/stations.js  --> hype_SMHI_stations.js

-----------------------
mHM (UFZ)
-----------------------

Used in:

    Rakovec, O., Mizukami, N., Kumar, R., Newman, A., Thober, S., Wood, A. W., et al. (2019). 
    Diagnostic evaluation of large‐domain hydrologic models calibrated across the contiguous United States. 
    Journal of Geophysical Research: Atmospheres, 2019; 124: 13991–14007. 
    https://doi.org/10.1029/2019JD030767

downloaded from:

    https://doi.org/10.5281/zenodo.2630558


-----------------------
VIC (NCAR)
-----------------------

Used in:

    Rakovec, O., Mizukami, N., Kumar, R., Newman, A., Thober, S., Wood, A. W., et al. (2019). 
    Diagnostic evaluation of large‐domain hydrologic models calibrated across the contiguous United States. 
    Journal of Geophysical Research: Atmospheres, 2019; 124: 13991–14007. 
    https://doi.org/10.1029/2019JD030767

downloaded from:

    https://doi.org/10.5281/zenodo.2630558




NOT USED HERE:

Used in:
    Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: 
    Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets, 
    Hydrol. Earth Syst. Sci., 23, 5089-5110, 
    https://doi.org/10.5194/hess-23-5089-2019, 2019.

Downloaded from:
   Kratzert, F. (2019). CAMELS benchmark models, HydroShare, https://doi.org/10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1

The following papers describe the calibration of the included models: 
- SAC-SMA & VIC (basin-wise calibrated): 
     Newman, A. J., Mizukami, N., Clark, M. P., Wood, A. W., Nijssen, B., and Nearing, G.: 
     Benchmarking of a physically based hydrologic model, Journal of Hydrometeorology, 18, 2215-2225, 2017. 
- mHM (basin-wise calibrated): 
     Mizukami, N., Rakovec, O., Newman, A. J., Clark, M. P., Wood, A. W., Gupta, H. V., and Kumar, R.: 
     On the choice of calibration metrics for "high-flow" estimation using hydrologic models, Hydrology and Earth System Sciences, 23, 2601-2614, 2019 

NOT USED HERE:
- FUSE: 
     These runs were generated by Nans Addor (n.addor@uea.ac.uk) and passed to me by personal communication. 
     The runs are part of on-going development on FUSE itself and might not reflect the final FUSE performance. 
- HBV: 
     Seibert, J., Vis, M. J. P., Lewis, E., and van Meerveld, H. J.: 
     Upper and lower benchmarks in hydrological modelling, Hydrological Processes, 32, 1120-1125, 2018. 
- VIC (regionally calibrated): 
     Mizukami, N., Clark, M. P., Newman, A. J., Wood, A. W., Gutmann, E. D., Nijssen, B., Rakovec, O., and Samaniego, L.: 
     Towards seamless large-domain parameter estimation for hydrologic models, Water Resources Research, 53, 8020-8040, 2017. 
- mHM (regionally calibrated): 
     Rakovec, O., Mizukami, N., Kumar, R., Newman, A. J., Thober, S., Wood, A. W., Clark, M. P., and Samaniego, L.: 
     Diagnostic Evaluation of Large-domain Hydrologic Models calibrated across the Contiguous United States, J. Geophysical Research - Atmospheres., in review, 2019.



History
-------
Written,  JM, May 2020
"""


# -------------------------------------------------------------------------
# Command line arguments - if script
#

# Comment|Uncomment - Begin
if __name__ == '__main__':

    import argparse
    import numpy as np

    pngbase              = ''
    pdffile              = ''
    usetex               = False
    
    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Compares calibration results of blended model with calibration results presented in Arnheimer et al. WRR (2020).''')
    parser.add_argument('-g', '--pngbase', action='store',
                    default=pngbase, dest='pngbase', metavar='pngbase',
                    help='Name basis for png output files (default: open screen window).')
    parser.add_argument('-p', '--pdffile', action='store',
                    default=pdffile, dest='pdffile', metavar='pdffile',
                    help='Name of pdf output file (default: open screen window).')
    parser.add_argument('-t', '--usetex', action='store_true', default=usetex, dest="usetex",
                    help="Use LaTeX to render text in pdf.")

    args                 = parser.parse_args()
    pngbase              = args.pngbase
    pdffile              = args.pdffile
    usetex               = args.usetex

    del parser, args
# Comment|Uncomment - End


    # -------------------------------------------------------------------------
    # Function definition - if function
    #    
    # -----------------------
    # add subolder scripts/lib to search path
    # -----------------------
    import sys
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path+'/lib')

    import numpy    as np
    import datetime as datetime
    import pandas   as pd
    import time
    import glob

    t1 = time.time()
    
    from   fread                import fread                   # in lib/
    from   sread                import sread                   # in lib/
    from   fsread               import fsread                  # in lib/
    from   autostring           import astr                    # in lib/
    from   pet_oudin            import pet_oudin               # in lib/
    from   climate_index_knoben import climate_index_knoben    # in lib/
    import color                                               # in lib/
    from   position             import position                # in lib/
    from   abc2plot             import abc2plot                # in lib/
    from   brewer               import get_brewer              # in lib/
    from   str2tex              import str2tex                 # in lib/
    from   errormeasures        import (kge,nse)               # in lib/

    # -------------------------------------------------------------------------
    # Read list of basins that got calibrated
    # -------------------------------------------------------------------------
    basin_ids_cal = np.transpose(sread("basins_calibration.dat",skip=0))[0]
    basin_ids_val = np.transpose(sread("basins_validation.dat",skip=0))[0]


    # -------------------------------------------------------------------------
    # Read basin metadata
    # -------------------------------------------------------------------------

    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac
    head = fread("../data_in/basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data_in/basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',cname=["lat","lon","area_km2","elevation_m","slope_deg","forest_frac","lat_gauge_deg","lon_gauge_deg"],sname=["basin_id","basin_name"])

    dict_metadata = {}
    for ibasin in range(len(meta_float)):
        dict_basin = {}
        dict_basin["name"]          = meta_string[ibasin][1]
        dict_basin["lat"]           = meta_float[ibasin][0]
        dict_basin["lon"]           = meta_float[ibasin][1]
        dict_basin["area_km2"]      = meta_float[ibasin][2]
        dict_basin["elevation_m"]   = meta_float[ibasin][3]
        dict_basin["slope_deg"]     = meta_float[ibasin][4]
        dict_basin["forest_frac"]   = meta_float[ibasin][5]
        dict_basin["lat_gauge_deg"] = meta_float[ibasin][6]
        dict_basin["lon_gauge_deg"] = meta_float[ibasin][7]

        if meta_string[ibasin][0] in basin_ids_cal:
            dict_metadata[meta_string[ibasin][0]] = dict_basin


    # -------------------------------------------------------------------------
    # Read basin calibration results
    # -------------------------------------------------------------------------

    # basin_id; basin_name; nse; rmse; kge
    head = fread("../data_in/basin_metadata/basin_calibration_results.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data_in/basin_metadata/basin_calibration_results.txt",skip=1,separator=';',cname=["nse","rmse","kge"],sname=["basin_id","basin_name"])

    meta_string = [ [ iitem.strip() for iitem in iline ] for iline in meta_string ]
    
    dict_calibration = {}
    for ibasin in range(len(meta_float)):
        dict_basin = {}
        dict_basin["name"]          = meta_string[ibasin][1]
        dict_basin["nse"]           = meta_float[ibasin][0]
        dict_basin["rmse"]          = meta_float[ibasin][1]
        dict_basin["kge"]           = meta_float[ibasin][2]

        if meta_string[ibasin][0] in basin_ids_cal:
            dict_calibration[meta_string[ibasin][0]] = dict_basin

    # -------------------------------------------------------------------------
    # Read basin validation results
    # -------------------------------------------------------------------------

    # basin_id; basin_name; nse; rmse; kge
    head = fread("../data_in/basin_metadata/basin_validation_results.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data_in/basin_metadata/basin_validation_results.txt",skip=1,separator=';',cname=["nse","rmse","kge"],sname=["basin_id","basin_name"])

    meta_string = [ [ iitem.strip() for iitem in iline ] for iline in meta_string ]
    
    dict_validation = {}
    for ibasin in range(len(meta_float)):
        dict_basin = {}
        dict_basin["name"]          = meta_string[ibasin][1]
        dict_basin["nse"]           = meta_float[ibasin][0]
        dict_basin["rmse"]          = meta_float[ibasin][1]
        dict_basin["kge"]           = meta_float[ibasin][2]

        if meta_string[ibasin][0] in basin_ids_val:
            dict_validation[meta_string[ibasin][0]] = dict_basin

    # -------------------------------------------------------------------------
    # Read HYPE data (calibration only)
    # -------------------------------------------------------------------------
    print("Reading HYPE results (calibration) ...")
    
    import json

    with open("../data_supp/hype_SMHI_stations.js") as json_file:
        hype_data = json.load(json_file)


    # -------------------------------------------------------------------------
    # Try to find station from our set in HYPE set (compare lat/lon and area)
    # -------------------------------------------------------------------------

    cc = 0
    merged_data_hype_cal = {}
    for basin_id in basin_ids_cal:

        our_lon = dict_metadata[basin_id]['lon_gauge_deg']
        our_lat = dict_metadata[basin_id]['lat_gauge_deg']

        # find closest station in HYPE dataset
        dist = [ ((ihype['x']-our_lon)**2+(ihype['y']-our_lat)**2)**0.5 for ihype in hype_data["data"] ]
        idx_mindist = np.argmin(dist)
        mindist = dist[idx_mindist]

        if mindist < 0.01:  # closer than 1km
            
            # area of closest station
            our_area  = dict_metadata[basin_id]['area_km2']
            hype_area = hype_data["data"][idx_mindist]['area'] / 10**6

            if np.abs(our_area-hype_area)/np.mean([our_area,hype_area]) < 0.01:

                cc += 1
                
                # print("")
                # print("lat  (HYPE/our): ", hype_data["data"][idx_mindist]['y'], " vs ", our_lat)
                # print("lon  (HYPE/our): ", hype_data["data"][idx_mindist]['x'], " vs ", our_lon)
                # print("area (HYPE/our): ", hype_data["data"][idx_mindist]['area'] / 10**6, " vs ", our_area)

                dict_merged = {}
                #
                # SMHI data for HYPE
                dict_merged['id_hype']   = hype_data["data"][idx_mindist]['subid']
                dict_merged['lon_hype']  = hype_data["data"][idx_mindist]['x']
                dict_merged['lat_hype']  = hype_data["data"][idx_mindist]['y']
                dict_merged['area_hype'] = hype_data["data"][idx_mindist]['area'] / 10**6
                dict_merged['kge_hype_cal']  = hype_data["data"][idx_mindist]['KGE']
                dict_merged['nse_hype_cal']  = hype_data["data"][idx_mindist]['NSE']
                #
                # our data for blended model
                dict_merged['name_our']  = dict_metadata[basin_id]['name']
                dict_merged['lon_our']   = dict_metadata[basin_id]['lon_gauge_deg']
                dict_merged['lat_our']   = dict_metadata[basin_id]['lat_gauge_deg']
                dict_merged['area_our']  = dict_metadata[basin_id]['area_km2']
                dict_merged['kge_our_cal']   = dict_calibration[basin_id]['kge']
                dict_merged['nse_our_cal']   = dict_calibration[basin_id]['nse']

                merged_data_hype_cal[basin_id] = dict_merged

    # our total number of stations:               3826
    # number of stations (no filter):             3826
    # number of stations (proximity filter):       996
    # number of stations (proximity+area filter):  818

    nses_hype_cal = np.array([ [merged_data_hype_cal[ii]["nse_hype_cal"], merged_data_hype_cal[ii]["nse_our_cal"]] for ii in merged_data_hype_cal ])
    kges_hype_cal = np.array([ [merged_data_hype_cal[ii]["kge_hype_cal"], merged_data_hype_cal[ii]["kge_our_cal"]] for ii in merged_data_hype_cal ])


    # -------------------------------------------------------------------------
    # Read VIC data (calibration)
    # -------------------------------------------------------------------------
    print("Reading VIC results (calibration) ...")
    
    basins_vic = glob.glob("../data_supp/mizukami_WRR_2017/*/output/hcdn_calib_case04.txt")
    basins_vic = [ ifile.split('/')[3] for ifile in basins_vic ]


    # -------------------------------------------------------------------------
    # Try to find station from our set in VIC set (calibration)
    # -------------------------------------------------------------------------

    cc = 0
    merged_data_vic_cal = {}
    for basin_id in basin_ids_cal:

        if basin_id in basins_vic:

            cc += 1
            vic_data = fread("../data_supp/mizukami_WRR_2017/"+basin_id+"/output/hcdn_calib_case04.txt",skip=0)

            dict_merged = {}
            #
            # vic simulated and observed streamflow

            # derive KGE and NSE
            dict_merged['kge_vic_cal']  = kge(vic_data[:,1],vic_data[:,0])
            dict_merged['nse_vic_cal']  = nse(vic_data[:,1],vic_data[:,0])
            #
            # our data for blended model
            dict_merged['name_our']  = dict_metadata[basin_id]['name']
            dict_merged['lon_our']   = dict_metadata[basin_id]['lon_gauge_deg']
            dict_merged['lat_our']   = dict_metadata[basin_id]['lat_gauge_deg']
            dict_merged['area_our']  = dict_metadata[basin_id]['area_km2']
            dict_merged['kge_our_cal']   = dict_calibration[basin_id]['kge']
            dict_merged['nse_our_cal']   = dict_calibration[basin_id]['nse']

            merged_data_vic_cal[basin_id] = dict_merged

    # our total number of stations:                533
    # number of calibrated stations in our DB:     169

    nses_vic_cal = np.array([ [merged_data_vic_cal[ii]["nse_vic_cal"], merged_data_vic_cal[ii]["nse_our_cal"]] for ii in merged_data_vic_cal ])
    kges_vic_cal = np.array([ [merged_data_vic_cal[ii]["kge_vic_cal"], merged_data_vic_cal[ii]["kge_our_cal"]] for ii in merged_data_vic_cal ])


    # -------------------------------------------------------------------------
    # Read VIC data (validation)
    # -------------------------------------------------------------------------
    print("Reading VIC results (validation) ...")
    
    basins_vic = glob.glob("../data_supp/mizukami_WRR_2017/*/output/hcdn_vali_case04.txt")
    basins_vic = [ ifile.split('/')[3] for ifile in basins_vic ]


    # -------------------------------------------------------------------------
    # Try to find station from our set in VIC set (validation)
    # -------------------------------------------------------------------------

    cc = 0
    merged_data_vic_val = {}
    for basin_id in basin_ids_val:

        if basin_id in basins_vic:

            cc += 1
            vic_data = fread("../data_supp/mizukami_WRR_2017/"+basin_id+"/output/hcdn_vali_case04.txt",skip=0)

            dict_merged = {}
            #
            # vic simulated and observed streamflow

            # derive KGE and NSE
            dict_merged['kge_vic_val']  = kge(vic_data[:,1],vic_data[:,0])
            dict_merged['nse_vic_val']  = nse(vic_data[:,1],vic_data[:,0])
            #
            # our data for blended model
            dict_merged['name_our']  = dict_metadata[basin_id]['name']
            dict_merged['lon_our']   = dict_metadata[basin_id]['lon_gauge_deg']
            dict_merged['lat_our']   = dict_metadata[basin_id]['lat_gauge_deg']
            dict_merged['area_our']  = dict_metadata[basin_id]['area_km2']
            dict_merged['kge_our_val']   = dict_validation[basin_id]['kge']
            dict_merged['nse_our_val']   = dict_validation[basin_id]['nse']

            merged_data_vic_val[basin_id] = dict_merged

    # our total number of stations:                533
    # number of calibrated stations in our DB:     169

    nses_vic_val = np.array([ [merged_data_vic_val[ii]["nse_vic_val"], merged_data_vic_val[ii]["nse_our_val"]] for ii in merged_data_vic_val ])
    kges_vic_val = np.array([ [merged_data_vic_val[ii]["kge_vic_val"], merged_data_vic_val[ii]["kge_our_val"]] for ii in merged_data_vic_val ])
    

    # -------------------------------------------------------------------------
    # Read mHM data (calibration)
    # -------------------------------------------------------------------------
    print("Reading mHM results (calibration) ...")
    
    basins_mhm = glob.glob("../data_supp/rakovec_JGRA_2019/*/calib_001/output/daily_discharge.out")
    basins_mhm = [ ifile.split('/')[3] for ifile in basins_mhm ]


    # -------------------------------------------------------------------------
    # Try to find station from our set in mHM set (calibration)
    # -------------------------------------------------------------------------

    cc = 0
    merged_data_mhm_cal = {}
    for basin_id in basin_ids_cal:

        if basin_id in basins_mhm:

            cc += 1
            mhm_data = fread("../data_supp/rakovec_JGRA_2019/"+basin_id+"/calib_001/output/daily_discharge.out",skip=1)

            dict_merged = {}
            #
            # mHM simulated and observed streamflow

            # derive KGE and NSE
            dict_merged['kge_mhm_cal']  = kge(mhm_data[:,4],mhm_data[:,5])
            dict_merged['nse_mhm_cal']  = nse(mhm_data[:,4],mhm_data[:,5])
            #
            # our data for blended model
            dict_merged['name_our']  = dict_metadata[basin_id]['name']
            dict_merged['lon_our']   = dict_metadata[basin_id]['lon_gauge_deg']
            dict_merged['lat_our']   = dict_metadata[basin_id]['lat_gauge_deg']
            dict_merged['area_our']  = dict_metadata[basin_id]['area_km2']
            dict_merged['kge_our_cal']   = dict_calibration[basin_id]['kge']
            dict_merged['nse_our_cal']   = dict_calibration[basin_id]['nse']

            merged_data_mhm_cal[basin_id] = dict_merged

    # our total number of stations:                492
    # number of calibrated stations in our DB:     162

    nses_mhm_cal = np.array([ [merged_data_mhm_cal[ii]["nse_mhm_cal"], merged_data_mhm_cal[ii]["nse_our_cal"]] for ii in merged_data_mhm_cal ])
    kges_mhm_cal = np.array([ [merged_data_mhm_cal[ii]["kge_mhm_cal"], merged_data_mhm_cal[ii]["kge_our_cal"]] for ii in merged_data_mhm_cal ])

    
    # -------------------------------------------------------------------------
    # Read mHM data (validation)
    # -------------------------------------------------------------------------
    print("Reading mHM results (validation) ...")
    
    basins_mhm = glob.glob("../data_supp/rakovec_JGRA_2019/*/calib_001/output/daily_discharge.out")
    basins_mhm = [ ifile.split('/')[3] for ifile in basins_mhm ]


    # -------------------------------------------------------------------------
    # Try to find station from our set in mHM set (validation)
    # -------------------------------------------------------------------------

    cc = 0
    merged_data_mhm_val = {}
    for basin_id in basin_ids_val:

        if basin_id in basins_mhm:

            cc += 1
            mhm_data = fread("../data_supp/rakovec_JGRA_2019/"+basin_id+"/calib_001/output/daily_discharge.out",skip=1)

            dict_merged = {}
            #
            # mHM simulated and observed streamflow

            # derive KGE and NSE
            dict_merged['kge_mhm_val']  = kge(mhm_data[:,4],mhm_data[:,5])
            dict_merged['nse_mhm_val']  = nse(mhm_data[:,4],mhm_data[:,5])
            #
            # our data for blended model
            dict_merged['name_our']  = dict_metadata[basin_id]['name']
            dict_merged['lon_our']   = dict_metadata[basin_id]['lon_gauge_deg']
            dict_merged['lat_our']   = dict_metadata[basin_id]['lat_gauge_deg']
            dict_merged['area_our']  = dict_metadata[basin_id]['area_km2']
            dict_merged['kge_our_val']   = dict_validation[basin_id]['kge']
            dict_merged['nse_our_val']   = dict_validation[basin_id]['nse']

            merged_data_mhm_val[basin_id] = dict_merged

    # our total number of stations:                492
    # number of calibrated stations in our DB:     162

    nses_mhm_val = np.array([ [merged_data_mhm_val[ii]["nse_mhm_val"], merged_data_mhm_val[ii]["nse_our_val"]] for ii in merged_data_mhm_val ])
    kges_mhm_val = np.array([ [merged_data_mhm_val[ii]["kge_mhm_val"], merged_data_mhm_val[ii]["kge_our_val"]] for ii in merged_data_mhm_val ])
        
# -------------------------------------------------------------------------
# Customize plots
#

if (pdffile == ''):
    if (pngbase == ''):
        outtype = 'x'
    else:
        outtype = 'png'
else:
    outtype = 'pdf'
    
# Main plot
nrow        = 6           # # of rows of subplots per figure
ncol        = 4           # # of columns of subplots per figure
hspace      = 0.07         # x-space between subplots
vspace      = 0.06        # y-space between subplots
right       = 0.9         # right space on page
textsize    = 8           # standard text size
dxabc       = 1.0         # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
# dyabc       = -13       # % of (max-min) shift up from lower x-axis for a,b,c,... labels
dyabc       = 0.0         # % of (max-min) shift up from lower x-axis for a,b,c,... labels
dxsig       = 1.23        # % of (max-min) shift to the right from left y-axis for signature
dysig       = -0.05       # % of (max-min) shift up from lower x-axis for signature
dxtit       = 0           # % of (max-min) shift to the right from left y-axis for title
dytit       = 1.3         # % of (max-min) shift up from lower x-axis for title

lwidth      = 0.5         # linewidth
elwidth     = 1.0         # errorbar line width
alwidth     = 0.5         # axis line width
glwidth     = 0.5         # grid line width
msize       = 1.0         # marker size
mwidth      = 0.5         # marker edge width
mcol1       = color.colours('blue')      # primary marker colour
mcol2       = color.colours('red')       # secondary
mcol3       = color.colours('red')       # third
mcols       = ['0.0', '0.4', '0.4', '0.7', '0.7', '1.0']
lcol1       = color.colours('blue')   # primary line colour
lcol2       = '0.0'
lcol3       = '0.0'
lcols       = ['None', 'None', 'None', 'None', 'None', '0.0']
hatches     = [None, None, None, None, None, '//']

# Legend
llxbbox     = 0.5         # x-anchor legend bounding box
llybbox     = 0.93         # y-anchor legend bounding box
llrspace    = 0.          # spacing between rows in legend
llcspace    = 0.2         # spacing between columns in legend
llhtextpad  = 0.4         # the pad between the legend handle and text
llhlength   = 1.5         # the length of the legend handles
frameon     = False       # if True, draw a frame around the legend. If None, use rc

# PNG
dpi         = 800         # 150 for testing
transparent = False
bbox_inches = 'tight'
pad_inches  = 0.035

# Clock options
ymax = 0.6
ntextsize   = 'medium'       # normal textsize
# modules
bmod        = 0.5            # fraction of ymax from center to start module colours
alphamod    = 0.7            # alpha channel for modules
fwm         = 0.05           # module width to remove at sides
ylabel1     = 1.15           # position of module names
ylabel2     = 1.35           # position of class names
mtextsize   = 'large'        # 1.3*textsize # textsize of module labels
# bars
bpar        = 0.4            # fraction of ymax from center to start with parameter bars
fwb         = [0.7,0.4,0.3]  # width of bars
plwidth     = 0.5
# parameters in centre
bplabel     = 0.1            # fractional distance of ymax of param numbers in centre from 0-line
ptextsize   = 'medium'       # 'small' # 0.8*textsize # textsize of param numbers in centre
# yaxis
space4yaxis = 2              # space for y-axis (integer)
ytextsize   = 'medium'       # 'small' # 0.8*textsize # textsize of y-axis
sig         = 'J Mai' # sign the plot

import matplotlib as mpl
if (outtype == 'pdf'):
    mpl.use('PDF') # set directly after import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    # Customize: http://matplotlib.sourceforge.net/users/customizing.html
    mpl.rc('ps', papersize='a4', usedistiller='xpdf') # ps2pdf
    mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
    if usetex:
        mpl.rc('text', usetex=True)
    else:
        mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        #mpl.rc('font',**{'family':'serif','serif':['times']})
    mpl.rc('text.latex', unicode=True)
elif (outtype == 'png'):
    mpl.use('Agg') # set directly after import matplotlib
    import matplotlib.pyplot as plt
    mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
    if usetex:
        mpl.rc('text', usetex=True)
    else:
        mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        #mpl.rc('font',**{'family':'serif','serif':['times']})
    mpl.rc('text.latex', unicode=True)
    mpl.rc('savefig', dpi=dpi, format='png')
else:
    import matplotlib.pyplot as plt
    mpl.rc('figure', figsize=(4./5.*8.27,4./5.*11.69)) # a4 portrait
mpl.rc('font', size=textsize)
mpl.rc('lines', linewidth=lwidth, color='black')
mpl.rc('axes', linewidth=alwidth, labelcolor='black')
mpl.rc('path', simplify=False) # do not remove

from matplotlib.patches import Rectangle, Circle, Polygon
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import shapely.geometry as sgeom
from copy import copy
# from mpl_toolkits.basemap import Basemap


# colors
if dobw:
    c = np.linspace(0.2, 0.85, nmod)
    c = np.ones(nmod)*0.7
    c = [ str(i) for i in c ]
    ocean_color = '0.1'
else:
    # c = [(165./255.,  0./255., 38./255.), # interception
    #      (215./255., 48./255., 39./255.), # snow
    #      (244./255.,109./255., 67./255.), # soil moisture
    #      (244./255.,109./255., 67./255.), # soil moisture
    #      (253./255.,174./255., 97./255.), # direct runoff
    #      (254./255.,224./255.,144./255.), # Evapotranspiration
    #      (171./255.,217./255.,233./255.), # interflow
    #      (116./255.,173./255.,209./255.), # percolation
    #      ( 69./255.,117./255.,180./255.), # routing
    #      ( 49./255., 54./255.,149./255.)] # geology
    c  = get_brewer('rdylbu11', rgb=True)
    tmp = c.pop(5)   # rm yellow
    np.random.shuffle(c)
    
    #c.insert(2,c[2]) # same colour for both soil moistures
    ocean_color = (151/256., 183/256., 224/256.)
    # ocean_color = color.get_brewer('accent5', rgb=True)[-1]

    # rainbow colors
    cc = color.get_brewer('dark_rainbow_256', rgb=True)
    cc = cc[::-1] # reverse colors
    cmap = mpl.colors.ListedColormap(cc)

    # green-pink colors
    cc = color.get_brewer('piyg10', rgb=True)
    cmap = mpl.colors.ListedColormap(cc)

    # # colors for each sub-basin from uwyellow4 (228,180,42) to gray
    # graylevel = 0.2
    # uwyellow = [251,213,79]
    # cc = [ (    (uwyellow[0]+ii/(22.-1)*(256*graylevel-uwyellow[0]))/256.,
    #             (uwyellow[1]+ii/(22.-1)*(256*graylevel-uwyellow[1]))/256.,
    #             (uwyellow[2]+ii/(22.-1)*(256*graylevel-uwyellow[2]))/256.) for ii in range(22) ]
    # cmap = mpl.colors.ListedColormap(cc)

    # # paired colors
    # cc = color.get_brewer('Paired6', rgb=True)
    # cmap = mpl.colors.ListedColormap(cc)

def find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.
    
    """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])


def lambert_xticks(ax, ticks):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    

def lambert_yticks(ax, ticks):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a Lambert Conformal projection."""
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels

    
# -------------------------------------------------------------------------
# Plot
#

if (outtype == 'pdf'):
    print('Plot PDF ', pdffile)
    pdf_pages = PdfPages(pdffile)
elif (outtype == 'png'):
    print('Plot PNG ', pngbase)
else:
    print('Plot X')
# figsize = mpl.rcParams['figure.figsize']



# sort basins starting with largest
areas = []
for ibasin_id,basin_id in enumerate(basin_ids_cal):
    areas.append(dict_metadata[basin_id]["area_km2"])
areas = np.array(areas)
idx_areas = np.argsort(areas)[::-1]
basin_ids_cal = np.array(basin_ids_cal)[idx_areas]




ifig = 0

# -------------------------------------------------------------------------
# Fig 1
#
ifig += 1
iplot = 0
print('Plot - Fig ', ifig)

fig = plt.figure(ifig)

# -----------------------------------------
# Scatterplot HYPE vs blended (calibration)
# -----------------------------------------
iplot += 1

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

sub.plot(nses_hype_cal[:,1],nses_hype_cal[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[0],  markerfacecolor='w',  label=str2tex('NSE$_\mathrm{cal}$'))
sub.plot(kges_hype_cal[:,1],kges_hype_cal[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[-1], markerfacecolor='w', label=str2tex('KGE$_\mathrm{cal}$'))
sub.plot([-1.0,1.0],[-1.0,1.0],linewidth=0.5*lwidth, linestyle='--',color='k')

# axis lables
# sub.set_xlabel(str2tex('Blended Raven\n(Mai et al., 2020)',usetex=usetex), color='black')
sub.set_ylabel(str2tex('HYPE\n(Arheimer et al., 2020)',usetex=usetex), color='black')


# limit plot range
sub.set_xlim([-0.0,1.0])
sub.set_ylim([-1.0,1.0])

# add label with current number of basins
sub.text(0.02,0.98,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_hype_cal))+"$",usetex=usetex),rotation=0,horizontalalignment="left", verticalalignment="top", transform=sub.transAxes)

# add label with calibration/validation tag
# sub.text(0.5,0.95,str2tex("Calibration",usetex=usetex),verticalalignment="top",horizontalalignment="center",transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')


# -----------------------------------------
# KDE HYPE vs blended
# -----------------------------------------
iplot += 2

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

from   statsmodels.distributions.empirical_distribution import ECDF

ecdf_nse_hype  = ECDF(nses_hype_cal[:,0])
min_z          = max(-10.0,np.min(nses_hype_cal[:,0]))
max_z          = np.max(nses_hype_cal[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_hype_cdf   = ecdf_nse_hype(z_grid)
linez1         = sub.plot(z_grid, nse_hype_cdf,  color=cc[0], linewidth=lwidth, linestyle="--", label=str2tex("NSE$_\mathrm{cal}^\mathrm{HYPE}$"))

ecdf_nse_raven = ECDF(nses_hype_cal[:,1])
min_z          = max(-10.0,np.min(nses_hype_cal[:,1]))
max_z          = np.max(nses_hype_cal[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_raven_cdf  = ecdf_nse_raven(z_grid)
linez2         = sub.plot(z_grid, nse_raven_cdf, color=cc[0], linewidth=lwidth, linestyle="-",  label=str2tex("NSE$_\mathrm{cal}^\mathrm{Raven}$"))

ecdf_kge_hype  = ECDF(kges_hype_cal[:,0])
min_z          = max(-10.0,np.min(kges_hype_cal[:,0]))
max_z          = np.max(kges_hype_cal[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_hype_cdf   = ecdf_kge_hype(z_grid)
linez3         = sub.plot(z_grid, kge_hype_cdf,  color=cc[-1], linewidth=lwidth, linestyle="--", label=str2tex("KGE$_\mathrm{cal}^\mathrm{HYPE}$"))

ecdf_kge_raven = ECDF(kges_hype_cal[:,1])
min_z          = max(-10.0,np.min(kges_hype_cal[:,1]))
max_z          = np.max(kges_hype_cal[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_raven_cdf  = ecdf_kge_raven(z_grid)
linez4         = sub.plot(z_grid, kge_raven_cdf, color=cc[-1], linewidth=lwidth, linestyle="-",  label=str2tex("KGE$_\mathrm{cal}^\mathrm{Raven}$"))

# limit plot range
sub.set_xlim([-0.2,1.0])
# sub.set_ylim([-0.0,1.0])

# axis lables
# sub.set_xlabel(str2tex('Performance metric [-]',usetex=usetex), color='black')
sub.set_ylabel(str2tex('CDF [-]',usetex=usetex), color='black')

# add label with current number of basins
# sub.text(1.1,0.5,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_hype_cal))+"$",usetex=usetex),rotation=90,horizontalalignment="left", verticalalignment="center", transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')

# -----------------------------------------
# Scatterplot VIC vs blended (calibration)
# -----------------------------------------
iplot += 2

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

sub.plot(nses_vic_cal[:,1],nses_vic_cal[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[0],  markerfacecolor='w',  label=str2tex('NSE$_\mathrm{cal}$'))
sub.plot(kges_vic_cal[:,1],kges_vic_cal[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[-1], markerfacecolor='w', label=str2tex('KGE$_\mathrm{cal}$'))
sub.plot([-1.0,1.0],[-1.0,1.0],linewidth=0.5*lwidth, linestyle='--',color='k')

# axis lables
# sub.set_xlabel(str2tex('Blended Raven\n(Mai et al., 2020)',usetex=usetex), color='black')
sub.set_ylabel(str2tex('VIC\n(Rakovec et al., 2019)',usetex=usetex), color='black')


# limit plot range
sub.set_xlim([-0.0,1.0])
sub.set_ylim([-1.0,1.0])

# add label with current number of basins
sub.text(0.02,0.02,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_vic_cal))+"$",usetex=usetex),rotation=0,horizontalalignment="left", verticalalignment="bottom", transform=sub.transAxes)

# add label with calibration/validation tag
# sub.text(0.5,0.95,str2tex("Calibration",usetex=usetex),verticalalignment="top",horizontalalignment="center",transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')


# -----------------------------------------
# Scatterplot VIC vs blended (validation)
# -----------------------------------------
iplot += 1

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

sub.plot(nses_vic_val[:,1],nses_vic_val[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[2],  markerfacecolor='w',  label=str2tex('NSE$_\mathrm{val}$'))
sub.plot(kges_vic_val[:,1],kges_vic_val[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[-3], markerfacecolor='w', label=str2tex('KGE$_\mathrm{val}$'))
sub.plot([-1.0,1.0],[-1.0,1.0],linewidth=0.5*lwidth, linestyle='--',color='k')

# axis lables
# sub.set_xlabel(str2tex('Blended Raven\n(Mai et al., 2020)',usetex=usetex), color='black')
# sub.set_ylabel(str2tex('VIC\n(Rakovec et al., 2019)',usetex=usetex), color='black')


# limit plot range
sub.set_xlim([-0.0,1.0])
sub.set_ylim([-1.0,1.0])

# add label with current number of basins
sub.text(0.02,0.02,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_vic_val))+"$",usetex=usetex),rotation=0,horizontalalignment="left", verticalalignment="bottom", transform=sub.transAxes)

# add label with calibration/validation tag
# sub.text(0.5,0.95,str2tex("Validation",usetex=usetex),verticalalignment="top",horizontalalignment="center",transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')


# -----------------------------------------
# KDE VIC vs blended (calibration)
# -----------------------------------------
iplot += 1

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

from   statsmodels.distributions.empirical_distribution import ECDF

ecdf_nse_vic  = ECDF(nses_vic_cal[:,0])
min_z          = max(-10.0,np.min(nses_vic_cal[:,0]))
max_z          = np.max(nses_vic_cal[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_vic_cdf   = ecdf_nse_vic(z_grid)
linez1         = sub.plot(z_grid, nse_vic_cdf,  color=cc[0], linewidth=lwidth, linestyle="--", label=str2tex("NSE$_\mathrm{cal}^\mathrm{VIC}$"))

ecdf_nse_raven = ECDF(nses_vic_cal[:,1])
min_z          = max(-10.0,np.min(nses_vic_cal[:,1]))
max_z          = np.max(nses_vic_cal[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_raven_cdf  = ecdf_nse_raven(z_grid)
linez2         = sub.plot(z_grid, nse_raven_cdf, color=cc[0], linewidth=lwidth, linestyle="-",  label=str2tex("NSE$_\mathrm{cal}^\mathrm{Raven}$"))

ecdf_kge_vic  = ECDF(kges_vic_cal[:,0])
min_z          = max(-10.0,np.min(kges_vic_cal[:,0]))
max_z          = np.max(kges_vic_cal[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_vic_cdf   = ecdf_kge_vic(z_grid)
linez3         = sub.plot(z_grid, kge_vic_cdf,  color=cc[-1], linewidth=lwidth, linestyle="--", label=str2tex("KGE$_\mathrm{cal}^\mathrm{VIC}$"))

ecdf_kge_raven = ECDF(kges_vic_cal[:,1])
min_z          = max(-10.0,np.min(kges_vic_cal[:,1]))
max_z          = np.max(kges_vic_cal[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_raven_cdf  = ecdf_kge_raven(z_grid)
linez4         = sub.plot(z_grid, kge_raven_cdf, color=cc[-1], linewidth=lwidth, linestyle="-",  label=str2tex("KGE$_\mathrm{cal}^\mathrm{Raven}$"))

# limit plot range
sub.set_xlim([-0.2,1.0])
# sub.set_ylim([-0.0,1.0])

# axis lables
#sub.set_xlabel(str2tex('Performance metric [-]',usetex=usetex), color='black')
sub.set_ylabel(str2tex('CDF [-]',usetex=usetex), color='black')

# add label with current number of basins
# sub.text(1.1,0.5,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_vic_cal))+"$",usetex=usetex),rotation=90,horizontalalignment="left", verticalalignment="center", transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')

# -----------------------------------------
# KDE VIC vs blended (validation)
# -----------------------------------------
iplot += 1

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

from   statsmodels.distributions.empirical_distribution import ECDF

ecdf_nse_vic  = ECDF(nses_vic_val[:,0])
min_z          = max(-10.0,np.min(nses_vic_val[:,0]))
max_z          = np.max(nses_vic_val[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_vic_cdf   = ecdf_nse_vic(z_grid)
linez1         = sub.plot(z_grid, nse_vic_cdf,  color=cc[2], linewidth=lwidth, linestyle="--", label=str2tex("NSE$_\mathrm{val}^\mathrm{VIC}$"))

ecdf_nse_raven = ECDF(nses_vic_val[:,1])
min_z          = max(-10.0,np.min(nses_vic_val[:,1]))
max_z          = np.max(nses_vic_val[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_raven_cdf  = ecdf_nse_raven(z_grid)
linez2         = sub.plot(z_grid, nse_raven_cdf, color=cc[2], linewidth=lwidth, linestyle="-",  label=str2tex("NSE$_\mathrm{val}^\mathrm{Raven}$"))

ecdf_kge_vic  = ECDF(kges_vic_val[:,0])
min_z          = max(-10.0,np.min(kges_vic_val[:,0]))
max_z          = np.max(kges_vic_val[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_vic_cdf   = ecdf_kge_vic(z_grid)
linez3         = sub.plot(z_grid, kge_vic_cdf,  color=cc[-3], linewidth=lwidth, linestyle="--", label=str2tex("KGE$_\mathrm{val}^\mathrm{VIC}$"))

ecdf_kge_raven = ECDF(kges_vic_val[:,1])
min_z          = max(-10.0,np.min(kges_vic_val[:,1]))
max_z          = np.max(kges_vic_val[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_raven_cdf  = ecdf_kge_raven(z_grid)
linez4         = sub.plot(z_grid, kge_raven_cdf, color=cc[-3], linewidth=lwidth, linestyle="-",  label=str2tex("KGE$_\mathrm{val}^\mathrm{Raven}$"))

# limit plot range
sub.set_xlim([-0.2,1.0])
# sub.set_ylim([-0.0,1.0])

# axis lables
# sub.set_xlabel(str2tex('Performance metric [-]',usetex=usetex), color='black')
# sub.set_ylabel(str2tex('CDF [-]',usetex=usetex), color='black')

# add label with current number of basins
# sub.text(1.1,0.5,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_vic_cal))+"$",usetex=usetex),rotation=90,horizontalalignment="left", verticalalignment="center", transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')
                
# -----------------------------------------
# Scatterplot mHM vs blended (calibration)
# -----------------------------------------
iplot += 1

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

sub.plot(nses_mhm_cal[:,1],nses_mhm_cal[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[0],  markerfacecolor='w',  label=str2tex('NSE$_\mathrm{cal}$'))
sub.plot(kges_mhm_cal[:,1],kges_mhm_cal[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[-1], markerfacecolor='w', label=str2tex('KGE$_\mathrm{cal}$'))
sub.plot([-1.0,1.0],[-1.0,1.0],linewidth=0.5*lwidth, linestyle='--',color='k')

# axis lables
sub.set_xlabel(str2tex('Blended Raven\n(Mai et al., 2020)',usetex=usetex), color='black')
sub.set_ylabel(str2tex('mHM\n(Rakovec et al., 2019)',usetex=usetex), color='black')


# limit plot range
sub.set_xlim([-0.0,1.0])
sub.set_ylim([-1.0,1.0])

# add label with current number of basins
sub.text(0.02,0.02,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_mhm_cal))+"$",usetex=usetex),rotation=0,horizontalalignment="left", verticalalignment="bottom", transform=sub.transAxes)

# add label with calibration/validation tag
# sub.text(0.5,0.95,str2tex("Calibration",usetex=usetex),verticalalignment="top",horizontalalignment="center",transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')

# -----------------------------------------
# Scatterplot mHM vs blended (validation)
# -----------------------------------------
iplot += 1

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

sub.plot(nses_mhm_val[:,1],nses_mhm_val[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[2],  markerfacecolor='w',  label=str2tex('NSE$_\mathrm{val}$'))
sub.plot(kges_mhm_val[:,1],kges_mhm_val[:,0],linewidth=0.0*lwidth,marker='o', markersize=3*msize, markeredgewidth=mwidth, markeredgecolor=cc[-3], markerfacecolor='w', label=str2tex('KGE$_\mathrm{val}$'))
sub.plot([-1.0,1.0],[-1.0,1.0],linewidth=0.5*lwidth, linestyle='--',color='k')

# axis lables
sub.set_xlabel(str2tex('Blended Raven\n(Mai et al., 2020)',usetex=usetex), color='black')
# sub.set_ylabel(str2tex('mHM\n(Rakovec et al., 2019)',usetex=usetex), color='black')


# limit plot range
sub.set_xlim([-0.0,1.0])
sub.set_ylim([-1.0,1.0])

# add label with current number of basins
sub.text(0.02,0.02,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_mhm_val))+"$",usetex=usetex),rotation=0,horizontalalignment="left", verticalalignment="bottom", transform=sub.transAxes)

# add label with calibration/validation tag
# sub.text(0.5,0.95,str2tex("Validation",usetex=usetex),verticalalignment="top",horizontalalignment="center",transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')


# -----------------------------------------
# KDE mHM vs blended (calibration)
# -----------------------------------------
iplot += 1

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

from   statsmodels.distributions.empirical_distribution import ECDF

ecdf_nse_mhm  = ECDF(nses_mhm_cal[:,0])
min_z          = max(-10.0,np.min(nses_mhm_cal[:,0]))
max_z          = np.max(nses_mhm_cal[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_mhm_cdf   = ecdf_nse_mhm(z_grid)
linez1         = sub.plot(z_grid, nse_mhm_cdf,  color=cc[0], linewidth=lwidth, linestyle="--", label=str2tex("NSE$_\mathrm{cal}^\mathrm{mHM}$"))

ecdf_nse_raven = ECDF(nses_mhm_cal[:,1])
min_z          = max(-10.0,np.min(nses_mhm_cal[:,1]))
max_z          = np.max(nses_mhm_cal[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_raven_cdf  = ecdf_nse_raven(z_grid)
linez2         = sub.plot(z_grid, nse_raven_cdf, color=cc[0], linewidth=lwidth, linestyle="-",  label=str2tex("NSE$_\mathrm{cal}^\mathrm{Raven}$"))

ecdf_kge_mhm  = ECDF(kges_mhm_cal[:,0])
min_z          = max(-10.0,np.min(kges_mhm_cal[:,0]))
max_z          = np.max(kges_mhm_cal[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_mhm_cdf   = ecdf_kge_mhm(z_grid)
linez3         = sub.plot(z_grid, kge_mhm_cdf,  color=cc[-1], linewidth=lwidth, linestyle="--", label=str2tex("KGE$_\mathrm{cal}^\mathrm{mHM}$"))

ecdf_kge_raven = ECDF(kges_mhm_cal[:,1])
min_z          = max(-10.0,np.min(kges_mhm_cal[:,1]))
max_z          = np.max(kges_mhm_cal[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_raven_cdf  = ecdf_kge_raven(z_grid)
linez4         = sub.plot(z_grid, kge_raven_cdf, color=cc[-1], linewidth=lwidth, linestyle="-",  label=str2tex("KGE$_\mathrm{cal}^\mathrm{Raven}$"))

# limit plot range
sub.set_xlim([-0.2,1.0])
# sub.set_ylim([-0.0,1.0])

# axis lables
sub.set_xlabel(str2tex('Performance metric [-]',usetex=usetex), color='black')
sub.set_ylabel(str2tex('CDF [-]',usetex=usetex), color='black')

# add label with current number of basins
# sub.text(1.1,0.5,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_mhm_cal))+"$",usetex=usetex),rotation=90,horizontalalignment="left", verticalalignment="center", transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')

# -----------------------------------------
# KDE mHM vs blended (validation)
# -----------------------------------------
iplot += 1

sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

from   statsmodels.distributions.empirical_distribution import ECDF

ecdf_nse_mhm  = ECDF(nses_mhm_val[:,0])
min_z          = max(-10.0,np.min(nses_mhm_val[:,0]))
max_z          = np.max(nses_mhm_val[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_mhm_cdf   = ecdf_nse_mhm(z_grid)
linez1         = sub.plot(z_grid, nse_mhm_cdf,  color=cc[2], linewidth=lwidth, linestyle="--", label=str2tex("NSE$_\mathrm{val}^\mathrm{mHM}$"))

ecdf_nse_raven = ECDF(nses_mhm_val[:,1])
min_z          = max(-10.0,np.min(nses_mhm_val[:,1]))
max_z          = np.max(nses_mhm_val[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
nse_raven_cdf  = ecdf_nse_raven(z_grid)
linez2         = sub.plot(z_grid, nse_raven_cdf, color=cc[2], linewidth=lwidth, linestyle="-",  label=str2tex("NSE$_\mathrm{val}^\mathrm{Raven}$"))

ecdf_kge_mhm  = ECDF(kges_mhm_val[:,0])
min_z          = max(-10.0,np.min(kges_mhm_val[:,0]))
max_z          = np.max(kges_mhm_val[:,0])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_mhm_cdf   = ecdf_kge_mhm(z_grid)
linez3         = sub.plot(z_grid, kge_mhm_cdf,  color=cc[-3], linewidth=lwidth, linestyle="--", label=str2tex("KGE$_\mathrm{val}^\mathrm{mHM}$"))

ecdf_kge_raven = ECDF(kges_mhm_val[:,1])
min_z          = max(-10.0,np.min(kges_mhm_val[:,1]))
max_z          = np.max(kges_mhm_val[:,1])
z_grid         = np.arange(min_z-0.05*(max_z-min_z), max_z+0.05*(max_z-min_z), 1.1*(max_z-min_z)/10000)
kge_raven_cdf  = ecdf_kge_raven(z_grid)
linez4         = sub.plot(z_grid, kge_raven_cdf, color=cc[-3], linewidth=lwidth, linestyle="-",  label=str2tex("KGE$_\mathrm{val}^\mathrm{Raven}$"))

# limit plot range
sub.set_xlim([-0.2,1.0])
# sub.set_ylim([-0.0,1.0])

# axis lables
sub.set_xlabel(str2tex('Performance metric [-]',usetex=usetex), color='black')
# sub.set_ylabel(str2tex('CDF [-]',usetex=usetex), color='black')

# add label with current number of basins
# sub.text(1.1,0.5,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(nses_mhm_cal))+"$",usetex=usetex),rotation=90,horizontalalignment="left", verticalalignment="center", transform=sub.transAxes)

# legend
ll = sub.legend(frameon=frameon, ncol=2,
                columnspacing=llcspace,
                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1)
                #fontsize = 'x-small')



if (outtype == 'pdf'):
    pdf_pages.savefig(fig)
    plt.close(fig)
elif (outtype == 'png'):
    pngfile = pngbase+"{0:04d}".format(ifig)+".png"
    fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close(fig)






    

# -------------------------------------------------------------------------
# Finished
#

if (outtype == 'pdf'):
    pdf_pages.close()
elif (outtype == 'png'):
    pass
else:
    plt.show()
        

t2    = time.time()
strin = '[m]: '+astr((t2-t1)/60.,1) if (t2-t1)>60. else '[s]: '+astr(t2-t1,0)
print('Time ', strin)

