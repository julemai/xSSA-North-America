#!/usr/bin/env python
from __future__ import print_function

# Copyright 2016-2018 Juliane Mai - juliane.mai(at)uwaterloo.ca
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
# run:
#    run extract_data_from_mat.py

# -----------------------
# add subolder scripts/lib to search path
# -----------------------
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/../lib')


import argparse
import tables
import numpy as np
import datetime as dt
import jams
from scipy.io import loadmat

doProps  = True       # should always be true
doShapes = False
doQobs   = False
doMeteo  = False


# solution found on:
# https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac


# large files are stored differently
matfile = tables.open_file('../data_in/USGS_CANOPEX_5797basins_withElevationSlope.mat')

# smaller files are different
matfile_forest = loadmat('../data_in/USGS_CANOPEX_5797basins_Forest.mat')

# ----------------------------
# Basin properties
# ----------------------------
if doProps:

    basin_id      = np.array([ ''.join([ chr(ii) for ii in jj[0]] ) for jj in matfile.root.USGS_CANOPEX_5797.ID[0] ])
    basin_name    = np.array([ ''.join([ chr(ii) for ii in jj[0]] ) for jj in matfile.root.USGS_CANOPEX_5797.Name[0] ])
    lat           = np.array([ ii[0][0,0] for ii in matfile.root.USGS_CANOPEX_5797.Centroid_Lat[0] ])
    lon           = np.array([ ii[0][0,0] for ii in matfile.root.USGS_CANOPEX_5797.Centroid_Lon[0] ])
    area_km2      = np.array([ ii[0][0,0] for ii in matfile.root.USGS_CANOPEX_5797.DrainageArea[0] ])
    elevation_m   = np.array([ ii[0][0,0] for ii in matfile.root.USGS_CANOPEX_5797.Elevation[0] ])
    slope_deg     = np.array([ ii[0][0,0] for ii in matfile.root.USGS_CANOPEX_5797.Slope[0] ])
    forest_frac   = np.array([ ii[0] for ii in np.array(matfile_forest['Forest'][:]) ])
    nbasins = len(basin_id)

    filename = '../data_in/basin_metadata/'+'/basin_physical_characteristics.txt'

    ff = open(filename,'w')
    ff.write("basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac \n")
    for ibasin in range(nbasins):
        ff.write(    basin_id[ibasin]+'; '+
                     basin_name[ibasin]+'; '+
                     jams.astr(lat[ibasin],prec=8)+'; '+
                     jams.astr(lon[ibasin],prec=8)+'; '+
                     jams.astr(area_km2[ibasin],prec=8)+'; '+
                     jams.astr(elevation_m[ibasin],prec=8)+'; '+
                     jams.astr(slope_deg[ibasin],prec=8)+'; '+
                     jams.astr(forest_frac[ibasin],prec=8)+' \n' )
    ff.close()
        

# ----------------------------
# Basin shapes
# ----------------------------
if doShapes:

    lat = matfile.root.USGS_CANOPEX_5797.Latitude[0]
    lon = matfile.root.USGS_CANOPEX_5797.Longitude[0]

    nbasins = len(lon)
    for ibasin in range(nbasins): # range(3): # range(3974,nbasins): #

        len_shape = len(lon[ibasin][0])

        path_basin = "../data_in/data_obs/"+basin_id[ibasin]
        if not(os.path.exists(path_basin)):
            os.makedirs(path_basin) 
        
        filename = path_basin+'/shape.dat'

        print("Write: "+filename)
        ff = open(filename,'w')
        ff.write("lon ; lat \n  ")
        ff.write("nan; nan \n ")
        for icoord in range(len_shape):
            # skip coordinates that contain NaN's
            if ( not( np.any([ np.isnan(lon[ibasin][0][icoord][0]), np.isnan(lat[ibasin][0][icoord][0]) ]) ) ):
                ff.write( jams.astr(lon[ibasin][0][icoord][0],prec=8)+'; '+
                          jams.astr(lat[ibasin][0][icoord][0],prec=8)+' \n ')
            else:
                ff.write("nan; nan \n ")
        # make sure polygon is closed
        if ( np.any([ (lon[ibasin][0][0][0] != lon[ibasin][0][-1][0]), (lat[ibasin][0][0][0] != lat[ibasin][0][-1][0]) ]) ):
            ff.write( jams.astr(lon[ibasin][0][0][0],prec=8)+'; '+
                      jams.astr(lat[ibasin][0][0][0],prec=8)+' \n ')
        ff.write("nan; nan \n ")
        ff.close()

    # print('Shapes not implemented yet!')

# ----------------------------
# Observed streamflow
# ----------------------------
if doQobs:
    
    dates = matfile.root.USGS_CANOPEX_5797.Dates[0]
    Qobs  = matfile.root.USGS_CANOPEX_5797.Qobs[0]
    nbasins = len(Qobs)
    for ibasin in range(nbasins): #range(3): #

        path_basin = "../data_in/data_obs/"+basin_id[ibasin]
        if not(os.path.exists(path_basin)):
            os.makedirs(path_basin) 
        
        filename = path_basin+'/model_basin_streamflow.rvt'
        ff = open(filename,'w')

        ntime = np.shape(Qobs[ibasin][0])[1]
        # print(ibasin,'  ntime = ',ntime)
        if ( ntime != np.shape(dates[ibasin][0])[1]):
            print('Basin: ',basin_id[ibasin])
            raise ValueError('Number of time steps in Qobs does not match number of dates!')

        dates_basin = np.array([ matlab2datetime(ii) for ii in dates[ibasin][0][0] ])
        first_date = dates_basin[0]
        last_date  = dates_basin[-1]
        if ( (last_date - first_date).days + 1) != ntime:
            print('Basin: ',basin_id[ibasin])
            print('First date: ',first_date)
            print('Last  date: ',last_date)
            raise ValueError("Not one data point per day available!")

        date_str = ' '.join(first_date.isoformat().split('T'))

        ff.write(':ObservationData  HYDROGRAPH  1  m3/s \n')
        ff.write(date_str+'  1  '+str(ntime)+' \n')
        for itime in range(ntime):
            if not(np.isnan(Qobs[ibasin][0][0,itime])):
                # Qobs is not NaN
                ff.write("{0:15.8f} \n".format(Qobs[ibasin][0][0,itime]))
            else:
                # print('itime = ',itime,'  --> NaN found')
                ff.write("{0:15.8f} \n".format(-1.2345))
        ff.write(':EndObservationData  \n')
        ff.close()

# ----------------------------
# Meteorology
# ----------------------------
if doMeteo:
    
    meteo = matfile.root.USGS_CANOPEX_5797.meteo[0]
    nbasins = len(meteo)
    fff = open('../data_in/data_obs/tmin_larger_than_tmax.dat','w')
    
    for ibasin in range(nbasins): #range(3): #

        ntime = np.shape(meteo[ibasin][0])[1] 

        path_basin = "../data_in/data_obs/"+basin_id[ibasin]
        if not(os.path.exists(path_basin)):
            os.makedirs(path_basin) 
        
        filename = path_basin+'/model_basin_mean_forcing_Santa-Clara.rvt'
        ff = open(filename,'w')

        ff.write(':MultiData \n')
        ff.write(' 1950-01-01  00:00:00  1  '+str(ntime)+' \n')
        ff.write(':Parameters     TEMP_DAILY_MIN   TEMP_DAILY_MAX   PRECIP   \n')
        ff.write(':Units          C                C                mm/d     \n')
        for itime in range(ntime):
            if meteo[ibasin][0][0,itime] <= meteo[ibasin][0][1,itime]:
                # this is how it should be: Tmin < Tmax
                ff.write("{0:15.8f}  {1:15.8f}  {2:15.8f} \n".format(meteo[ibasin][0][0,itime],meteo[ibasin][0][1,itime],meteo[ibasin][0][2,itime]))
            else:
                # some bug: Tmin > Tmax
                print('Basin: ',basin_id[ibasin],'  itime: ',itime, '   --> swapped Tmin and Tmax')
                ff.write("{0:15.8f}  {1:15.8f}  {2:15.8f} \n".format(meteo[ibasin][0][1,itime],meteo[ibasin][0][0,itime],meteo[ibasin][0][2,itime]))
        ff.write(':EndMultiData  \n')
        ff.close()

        for itime in range(ntime):
            if meteo[ibasin][0][0,itime] > meteo[ibasin][0][1,itime]:
                fff.write('Basin: '+basin_id[ibasin]+'  itime: '+str(itime)+'   --> Tmin ('+jams.astr(meteo[ibasin][0][0,itime],prec=4)+
                          ') larger than Tmax ('+jams.astr(meteo[ibasin][0][1,itime],prec=4)+') \n')

        # files contain -0.00000000   --> RAVEN has problems with that...
        #
        # files=$( \ls /home/julemai/projects/rpp-hwheater/julemai/sa-usgs-canopex/data_in/data_obs/*/model_basin_mean_forcing_Santa-Clara.rvt )
        # for ff in ${files} ; do sed 's/-0.00000000/ 0.00000000/g' ${ff} > ${ff}.tmp ; mv ${ff}.tmp ${ff} ; done
        # for ff in $files ; do nn=$(grep '\-0\.00000000' ${ff} | wc -l) ; echo $ff $nn ; done

    fff.close()
