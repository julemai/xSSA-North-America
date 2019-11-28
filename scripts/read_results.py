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

# You should have received a copy of the GNU Lesser General Public Licensefstop
# along with Juliane Mai's personal code library.  If not, see <http://www.gnu.org/licenses/>.
#
# run with:
#     source env-3.5/bin/activate
#     run read_results.py 

from __future__ import print_function

"""
Benchmark example to test Sobol' sensitivity analysis for models with multiple process options. 
The hydrologic modeling framework RAVEN is employed here.

History
-------
Written,  JM, Jun 2019
"""

# -----------------------
# add subolder scripts/lib to search path
# -----------------------
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/lib')


import numpy as np
import glob
import pickle
import sobol_index
import autostring

files = glob.glob("../data_out/*/results_nsets10.pkl")
for ff in files:
    setup = pickle.load( open( ff, "rb" ) )
    print(ff)
    print(setup['ntime'])
    print(setup['sobol_indexes']['paras'].keys())
    print(setup['sobol_indexes']['process_options'].keys())
    print(setup['sobol_indexes']['processes'].keys())

    # this is the weird cases
    if not('msti' in setup['sobol_indexes']['paras'].keys()):
        print('-------------------')
        print('>>> NOT WORKING <<<')
        print('-------------------')
        f_a = setup['f_a']
        print('shape: nse: ',np.shape(f_a['nse']))
        print('shape: Q:   ',np.shape(f_a['Q']))

        ikey = 'Q'
        si, sti, msi, msti, wsi, wsti = sobol_index.sobol_index(ya=setup['f_a'][ikey],
                                                                yb=setup['f_b'][ikey],
                                                                yc=setup['f_c_processes'][ikey],
                                                                si=True,
                                                                sti=True,
                                                                mean=True,
                                                                wmean=True,
                                                                method='Mai1999')
        print('msti = ',msti)
        print('diff stored si and new si:   ',np.sum(np.abs(si-setup['sobol_indexes']['processes']['si'][ikey])))
        print('diff stored sti and new sti: ',np.sum(np.abs(sti-setup['sobol_indexes']['processes']['sti'][ikey])))

    else:
        print('-------------------')
        print('>>>   WORKING   <<<')
        print('-------------------')
        print('mSTi for processes [Q]:   ', setup['sobol_indexes']['processes']['msti']['Q'])
        print('mSTi for processes [NSE]: ',                 setup['sobol_indexes']['processes']['msti']['nse'])    # None
        print(' STi for processes [NSE]: ', setup['sobol_indexes']['processes']['sti']['nse'])
        print('overall best NSE: ',np.max([np.max(setup['f_a']['nse']),
                                           np.max(setup['f_b']['nse']),
                                           np.max(setup['f_c_paras']['nse']),
                                           np.max(setup['f_c_process_options']['nse']),
                                           np.max(setup['f_c_processes']['nse'])]))
    print()
