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

def read_qobs_range_from_rvt(filename):

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

    return [startdate,enddate]

files = glob.glob('../data_in/data_obs/*/model_basin_streamflow.rvt')

cc_ok = 0
cc_miss = 0
overlaps = []
for ff in files:
    #print(ff)
    sdate, edate = read_qobs_range_from_rvt(ff)
    #print('>>> startdate = ',str(sdate))
    #print('>>> enddate   = ',str(edate))

    r1_start = datetime.datetime(1991,1,1,0,0)
    r1_end   = datetime.datetime(2010,12,31,0,0)

    covers_sim_period = (sdate <= r1_start) and (edate >= r1_end)

    if covers_sim_period:
        cc_ok += 1
    else:
        cc_miss += 1
        
        overlap = max(0,min((r1_end - sdate).days, (edate - r1_start).days) + 1)
        overlaps.append(overlap)
overlaps = np.array(overlaps)        

print("Basins with ALL QOBS DATA available: ",cc_ok,"  (",1.0*cc_ok/len(files)*100.,"%)")
print("Basins with NO  QOBS DATA AT ALL   : ",len(np.where(overlaps==    0)[0]),"  (",1.0*(      len(np.where(overlaps==    0)[0]))/len(files)*100.,"%)")
print("")
print("Basins with some missing Qobs:      ",cc_miss)
print("Basins with at least 10 years data: ",cc_ok+len(np.where(overlaps>10*365)[0]),"  (",1.0*(cc_ok+len(np.where(overlaps>10*365)[0]))/len(files)*100.,"%)") 
print("Basins with at least  5 years data: ",cc_ok+len(np.where(overlaps> 5*365)[0]),"  (",1.0*(cc_ok+len(np.where(overlaps> 5*365)[0]))/len(files)*100.,"%)") 
print("Basins with at least  2 years data: ",cc_ok+len(np.where(overlaps> 2*365)[0]),"  (",1.0*(cc_ok+len(np.where(overlaps> 2*365)[0]))/len(files)*100.,"%)") 
print("Basins with at least  1 year  data: ",cc_ok+len(np.where(overlaps> 1*365)[0]),"  (",1.0*(cc_ok+len(np.where(overlaps> 1*365)[0]))/len(files)*100.,"%)") 
