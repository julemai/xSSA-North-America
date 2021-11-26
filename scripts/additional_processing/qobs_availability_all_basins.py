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
#    run qobs_availability_all_basins.py    

# -----------------------
# add subolder scripts/lib to search path
# -----------------------
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/../lib')

import argparse
import numpy as np
import jams
import glob
import datetime

def read_qobs_range_from_rvt(filename,start_day,end_day):

    ff = open(filename, 'r')
    line = ff.readline().strip()
    line = " ".join(line.split())
    # print('line: ',line)
    line = ff.readline().strip()
    line = " ".join(line.split())
    # print('line: ',line)
    yr = np.int(line.split()[0].split('-')[0])
    mo = np.int(line.split()[0].split('-')[1])
    dy = np.int(line.split()[0].split('-')[2])
    hr = np.int(line.split()[1].split(':')[0])
    mi = np.int(line.split()[1].split(':')[1])
    ss = np.int(line.split()[1].split(':')[2])
    startdate = datetime.datetime(yr,mo,dy,hr,mi,ss)
    tdelta = np.float(line.split()[2])
    ntime  = np.int(line.split()[3])
    line = ff.readline().strip()
    line = " ".join(line.split())
    # print('line: ',line)

    enddate = startdate+datetime.timedelta(days=tdelta*(ntime-1))

    # print('>>> tdelta    = ',tdelta)
    # print('>>> ntime     = ',ntime)
    # print('>>> startdate = ',str(startdate))
    # print('>>> enddate   = ',str(enddate))   
    
    ff.close()

    if start_day < startdate:
        idx_start = 0
    elif start_day > enddate:
        nvalid = 0
        return [startdate,enddate,nvalid]
    else:
        idx_start = np.int( (start_day - startdate).days * tdelta )

    if end_day > enddate:
        idx_end = np.int(tdelta*ntime)
    elif end_day < startdate:
        nvalid = 0
        return [startdate,enddate,nvalid]
    else:
        idx_end = np.int( ((end_day - startdate).days + 1) * tdelta )

    ntime = idx_end - idx_start

    # print("idx_start :",idx_start)
    # print("idx_end   :",idx_end)
    # print("ntime     :",ntime)

    # read data
    ff = open(filename, 'r')
    lines = ff.readlines()
    ff.close()
    lines = lines[idx_start+2:idx_end+2]
    lines = np.array([ np.float(ll.strip()) for ll in lines ])
    
    # look for number of data points that are not -1.2345
    n_nodata_vals = len(np.where(lines == -1.2345)[0])
    nvalid = ntime - n_nodata_vals

    return [startdate,enddate,nvalid]

files = np.sort(glob.glob('../data_in/data_obs/*/model_basin_streamflow.rvt'))

nvalids_calibration = []
nvalids_validation  = []
for ff in files:

    # calibration period
    r1_start = datetime.datetime(1991,1,1,0,0)
    r1_end   = datetime.datetime(2010,12,31,0,0)
    sdate, edate, nvalid_calib = read_qobs_range_from_rvt(ff,r1_start,r1_end)
    nvalids_calibration.append(nvalid_calib)

    # validation period 1
    r2_start = datetime.datetime(1971,1,1,0,0)
    r2_end   = datetime.datetime(1990,12,31,0,0)
    sdate, edate, nvalid_valid = read_qobs_range_from_rvt(ff,r2_start,r2_end)
    nvalids_validation.append(nvalid_valid)

    print(ff.split('/')[3]+"; "+str(nvalid_calib)+"; "+str(nvalid_valid))
        
nvalids_calibration   = np.array(nvalids_calibration)
nvalids_validation    = np.array(nvalids_validation)
ntime_all = (r1_end-r1_start).days + 1

cc_ok   = len(np.where(nvalids_calibration == ntime_all)[0])
cc_ko   = len(np.where(nvalids_calibration == 0        )[0])
cc_miss = len(np.where(nvalids_calibration != ntime_all)[0])

print("Basins with ALL QOBS DATA available: ",cc_ok,"  (",1.0*cc_ok/len(files)*100.,"%)")
print("Basins with NO  QOBS DATA AT ALL   : ",cc_ko,"  (",1.0*cc_ko/len(files)*100.,"%)")
print("")
print("Basins with some missing Qobs:      ",cc_miss)
print("Basins with at least 10 years data: ",len(np.where(nvalids_calibration >= 10*365)[0]),"  (",1.0*(len(np.where(nvalids_calibration > 10*365)[0]))/len(files)*100.,"%)") 
print("Basins with at least  5 years data: ",len(np.where(nvalids_calibration >=  5*365)[0]),"  (",1.0*(len(np.where(nvalids_calibration >  5*365)[0]))/len(files)*100.,"%)   <---- used for calibration") 
print("Basins with at least  2 years data: ",len(np.where(nvalids_calibration >=  2*365)[0]),"  (",1.0*(len(np.where(nvalids_calibration >  2*365)[0]))/len(files)*100.,"%)") 
print("Basins with at least  1 year  data: ",len(np.where(nvalids_calibration >=  1*365)[0]),"  (",1.0*(len(np.where(nvalids_calibration >  1*365)[0]))/len(files)*100.,"%)") 

print("Number of calibration basins with enough data in validation period: ", len(np.where( (nvalids_calibration >=  5*365) & ((nvalids_validation >=  5*365)) )[0]))


# write basin lists
file_calib = open("basins_calibration.dat","w") 
file_valid = open("basins_validation.dat","w") 
for iff,ff in enumerate(files):

    if nvalids_calibration[iff] >= 5*365:
        file_calib.write(ff.split('/')[3]+"\n")

        if nvalids_validation[iff] >= 5*365:
            file_valid.write(ff.split('/')[3]+"\n")

file_calib.close()
file_valid.close()
