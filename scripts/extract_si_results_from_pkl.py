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
#     run extract_si_results_from_pkl.py -i ../data_out/01010000/results_nsets2.pkl -o ../data_out/01010000/sensitivity_nsets2.pkl

from __future__ import print_function

"""
Extract sensitivity results from Pickle file that contains also model outputs. 
Resulting file is much smaller and hence easier to handle.

History
-------
Written,  JM, Nov 2019
"""

# -------------------------------------------------------------------------
# Command line arguments
# -------------------------------------------------------------------------

import argparse

input_file   = None
output_file  = None

parser  = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description='''Plot basin shape.''')
parser.add_argument('-i', '--input_file', action='store',
                    default=input_file, dest='input_file', metavar='input_file',
                    help='Input file name of pickle file containing all model results including sensitivity indexes. Mandatory. (default: None).')
parser.add_argument('-o', '--output_file', action='store',
                    default=output_file, dest='output_file', metavar='output_file',
                    help='Output file name of pickle file that will only contain sensitivity indexes. Mandatory. (default: None).')

args         = parser.parse_args()
input_file   = args.input_file
output_file  = args.output_file

if input_file is None:
    raise ValueError("Input file name must be given! (Option -i)")
if output_file is None:
    raise ValueError("Output file name must be given! (Option -o)")

# -----------------------
# add subolder scripts/lib to search path
# -----------------------
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/lib')

import pickle

with open(input_file, "rb") as ff:

    # load data (input_file)
    setup = pickle.load(ff)

    # save sobol indexes in extra file (output_file)
    dict_si = {}
    dict_si['sobol_indexes'] = setup['sobol_indexes']
    pickle.dump( dict_si, open( output_file, "wb" ) )

