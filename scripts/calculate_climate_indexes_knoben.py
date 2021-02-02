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
#
#     print on terminal (snow fraction as in raven)
#     run calculate_climate_indexes_knoben.py -i 08012000 -s 'raven'
#
#     print on terminal (snow fraction as in Knoben)
#     run calculate_climate_indexes_knoben.py -i 08012000 -s 'knoben'
#
#     save to file
#     run calculate_climate_indexes_knoben.py -i 02479000 -s 'raven' -o test.dat
#
#     save to final file
#     run calculate_climate_indexes_knoben.py -i 02479000 -s 'raven' -o ../data_in/basin_metadata/basin_climate_index_knoben_snow-raven.txt
#
#     save to final file
#     run calculate_climate_indexes_knoben.py -i 02479000 -s 'knoben' -o ../data_in/basin_metadata/basin_climate_index_knoben_snow-knoben.txt


from __future__ import print_function

"""
Derives the climate indexes following Knoben et al. WRR (2018).
Climate index is a RGB color where
   Red   = Aridity I_m
   Green = Seasonality I_m,r
   Blue  = Fraction of precipitation as snow f_S
The estimator needs mean daily temperature, daily precipitation, and 
daily potential evapotranspiration as inputs.

History
-------
Written,  JM, Feb 2020
"""

# -------------------------------------------------------------------------
# Command line arguments - if script
#

# Comment|Uncomment - Begin
if __name__ == '__main__':

    import argparse
    import numpy as np

    basin_ids = None
    outfile   = None # '../data_in/basin_metadata/basin_climate_index_knoben_snow=raven.txt'
    snow_calc = "raven"
    
    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Derives the climate indexes following Knoben et al. WRR (2018).''')
    parser.add_argument('-i', '--basin_ids', action='store',
                        default=basin_ids, dest='basin_ids', metavar='basin_ids',
                        help='Basin ID of basins to analyze. Mandatory. (default: None).')
    parser.add_argument('-o', '--outfile', action='store',
                        default=outfile, dest='outfile', metavar='outfile',
                        help='Filename to dump outputs. If not given printed to terminal. (default: None.).')
    parser.add_argument('-s', '--snow_calc', action='store',
                        default=snow_calc, dest='snow_calc', metavar='snow_calc',
                        help='Method of calculating fraction of precip as snow f_S. "knoben" uses sum of average monthly precipitation of months with average temperature below 0 while "raven" determines the fraction of snow for each day based on Dingman (delta=2.0, temp=0.15) and uses ratio of total sum of snow and precipitation over whole time period to derive f_S (default: "raven".).')

    args      = parser.parse_args()
    basin_ids = args.basin_ids
    outfile   = args.outfile
    snow_calc = args.snow_calc

    # convert basin_ids to list 
    basin_ids = basin_ids.strip()
    basin_ids = " ".join(basin_ids.split())
    basin_ids = basin_ids.split()
    # print('basin_ids: ',basin_ids)

    del parser, args
# Comment|Uncomment - End


    # -------------------------------------------------------------------------
    # Function definition - if function
    #    

    # Basin ID need to be specified
    if basin_ids is None:
        raise ValueError('Basin ID (option -i) needs to given. Basin ID needs to correspond to ID of a CAMELS basin.')

    # -----------------------
    # add subolder scripts/lib to search path
    # -----------------------
    import sys
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path+'/lib')

    import numpy    as np
    import pandas   as pd
    import datetime as datetime
    
    from   autostring           import astr                    # in lib/
    from   pet_oudin            import pet_oudin               # in lib/
    from   climate_index_knoben import climate_index_knoben    # in lib/

    # ----------------------------------
    # calculate climate indicators
    # ----------------------------------
    if not( outfile is None):
        file_climate_indexes = outfile
        ff_clim = open(file_climate_indexes, "w")
        ff_clim.write("basin_id; aridity I_m; seasonality I_m,r; fraction precipitation as snow f_S; red; green; blue \n")

        file_annual_forcing_stats = '.'.join(outfile.split('.')[:-1])+'_forcings.'+outfile.split('.')[-1]
        ff_forc = open(file_annual_forcing_stats, "w")
        ff_forc.write("basin_id; annual_sum_prec_mm; annual_ave_temp_degC; annual_sum_pet_mm \n")
    
    climate_indexes = {}
    for basin_id in basin_ids:

        # ----------------------------------
        # basin properties
        # ----------------------------------
        basin_prop = {}

        file_gauge_info = '../data_in/basin_metadata/basin_physical_characteristics.txt'
        ff = open(file_gauge_info, "r")
        lines = ff.readlines()
        ff.close()

        found = False
        for ill,ll in enumerate(lines):
            if ill > 0:
                tmp = ll.strip().split(';')
                if (tmp[0] == basin_id):
                    found = True
                    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac 
                  
                    basin_prop['id']           = str(basin_id)
                    basin_prop['name']         = str(tmp[1].strip())
                    basin_prop['lat_deg']      = np.float(tmp[2].strip())
                    basin_prop['lon_deg']      = np.float(tmp[3].strip())
                    basin_prop['area_km2']     = np.float(tmp[4].strip())
                    basin_prop['elevation_m']  = np.float(tmp[5].strip())
                    basin_prop['slope_deg']    = np.float(tmp[6].strip()) 
                    basin_prop['forest_frac']  = np.float(tmp[7].strip())

        if not(found):
            raise ValueError('Basin ID not found in '+file_gauge_info)

        # ----------------------------------------
        # read forcings
        # ----------------------------------------
        raven_forcings_file = "../data_in/data_obs/"+basin_id+"/model_basin_mean_forcing_Santa-Clara.rvt"
              
        ff = open(raven_forcings_file, "r")
        content = ff.readlines()
        ff.close()
      
        # get headers
        # :Parameters     TEMP_DAILY_MIN   TEMP_DAILY_MAX   PRECIP
        head = content[2].strip().split()[1:]
        idx_precip = [ head.index(v) for v in head if 'PRECIP' in v ][0]
        head_precip = np.array(head)[idx_precip]
        idx_tmin = [ head.index(v) for v in head if 'TEMP_DAILY_MIN' in v ][0]
        head_tmin = np.array(head)[idx_tmin]
        idx_tmax = [ head.index(v) for v in head if 'TEMP_DAILY_MAX' in v ][0]
        head_tmax = np.array(head)[idx_tmax]

        # get data
        data      = np.array( [ [ np.float(icc) for icc in cc.strip().split() ] for cc in content[4:-1] ])
        precip    = data[:,idx_precip]
        tmin      = data[:,idx_tmin]
        tmax      = data[:,idx_tmax]
        tave      = (tmax+tmin)/2.

        # get time array
        timeline = content[1].strip().split()
        refdate = timeline[0]
        reftime = timeline[1]
        deltat = np.float(timeline[2]) # in [days]
        ntime  = np.int(timeline[3])
        reference_date = datetime.datetime(np.int(refdate.split('-')[0]),np.int(refdate.split('-')[1]),np.int(refdate.split('-')[2]),
                          np.int(reftime.split(':')[0]),np.int(reftime.split(':')[1]),np.int(reftime.split(':')[2]))
        ttime = np.array( [ reference_date + datetime.timedelta(days=itime*deltat) for itime in range(ntime) ] )

        # derive PET
        doy = np.array([ ittime.timetuple().tm_yday   for ittime in ttime ])   # day of the year
        lat = np.ones(ntime) * basin_prop['lat_deg']                             # make latitudes same shape as doy and temp
        pet = pet_oudin(tave, lat, doy)

        # ---------------------------------------
        # derive annual forcings
        # put all in dataframe to make averaging easier
        data = np.transpose( np.array([ precip, tave, pet]) )
        df = pd.DataFrame(data, columns = ['precip', 'tave', 'pet'], index=pd.DatetimeIndex(ttime), dtype=float)

        # annual sum of precip
        precip_total_annual     = df.precip.resample("Y").agg(['sum'])                                  # 61 values
        precip_total_annual_ave = precip_total_annual.mean()                                            #  1 value

        # average annual temperature
        tave_ave_annual     = df.tave.resample("Y").agg(['mean'])                                       # 61 values
        tave_ave_annual_ave = tave_ave_annual.mean()                                                    #  1 value

        # average sum of pet
        pet_total_annual     = df.pet.resample("Y").agg(['sum'])                                        # 61 values
        pet_total_annual_ave = pet_total_annual.mean()                                                  #  1 value

        # ---------------------------------------

        if snow_calc == "raven":
            # derive snow amount as RAVEN does in ":RainSnowFraction RAINSNOW_DINGMAN"
            #    uses global parameters: G.rainsnow_temp=0.15;
            #                            G.rainsnow_delta=2.0;
            #    ---> temp < 0.15 - 1.0  --> snow = 1.0 * precip
            #         temp > 0.15 + 1.0  --> snow = 0.0 * precip
            #         0.15 - 1.0 < temp < 0.15 + 1.0  --> snow = (temp + delta/2.)*(1./delta) - 1./delta * precip
            snow_delta = 2.0
            snow_temp  = 0.15
            snow = np.ones(np.shape(precip))*-9999.0
            # all precip is snow
            idx = np.where(tave<snow_temp-snow_delta/2.)
            if len(idx) > 0:
                snow[idx[0]] = 1.0 * precip[idx[0]]
            # all precip is rain
            idx = np.where(tave>snow_temp+snow_delta/2.)
            if len(idx) > 0:
                snow[idx[0]] = 0.0 * precip[idx[0]]
            # fraction of precip is snow
            idx = np.where((tave<=snow_temp+snow_delta/2.) & (tave>=snow_temp-snow_delta/2.))
            if len(idx) > 0:
                snow[idx[0]] = ((snow_temp + snow_delta/2.)*(1./snow_delta) - 1./snow_delta * tave[idx[0]]) * precip[idx[0]]

            # snow as in RAVEN used
            climate_index = climate_index_knoben(ttime, precip, tave, pet, snow=snow, color=True, indicators=True)
        elif snow_calc == "knoben":
            # snow as in Knoben et al. (2018) used
            climate_index = climate_index_knoben(ttime, precip, tave, pet, snow=None, color=True, indicators=True)
        else:
            raise ValueError("This snow calculation routine (-s) is not implemented yet!")
        
        #
        climate_indexes[basin_id] = tuple([climate_index['color']['red'],climate_index['color']['green'],climate_index['color']['blue']])

        if not( outfile is None):
            print(basin_id,": ",
                      " aridity        = ",astr(climate_index['indicators']['aridity'],prec=4),
                      " seasonality    = ",astr(climate_index['indicators']['seasonality'],prec=4),
                      " precip_as_snow = ",astr(climate_index['indicators']['precip_as_snow'],prec=4))
            ff_clim.write(    basin_id+"; "+
                          astr(climate_index['indicators']['aridity'],prec=4)+"; "+
                          astr(climate_index['indicators']['seasonality'],prec=4)+"; "+
                          astr(climate_index['indicators']['precip_as_snow'],prec=4)+"; "+
                          astr(climate_index['color']['red'],prec=4)+"; "+
                          astr(climate_index['color']['green'],prec=4)+"; "+
                          astr(climate_index['color']['blue'],prec=4)+"\n")

            ff_forc.write(    basin_id+"; "+
                                  astr(precip_total_annual_ave[0],prec=4)+"; "+
                                  astr(tave_ave_annual_ave[0],prec=4)+"; "+
                                  astr(pet_total_annual_ave[0],prec=4)+"; "+"\n")
        else:
            print(climate_index)

    if not( outfile is None):
        ff_clim.close()
        print("Wrote: "+file_climate_indexes)

        ff_forc.close()
        print("Wrote: "+file_annual_forcing_stats)
