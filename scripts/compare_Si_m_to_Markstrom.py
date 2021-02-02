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
#     run compare_Si_m_to_Markstrom.py -g test_


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
Plots sum of Si_mean of all parameters for each basin on a map. To compare to Figure 2 (panel for Mean of Runoff) in:

Markstrom, S. L., Hay, L. E., & Clark, M. P. (2016). 
Towards simplification of hydrologic modeling: identification of dominant processes. 
Hydrology and Earth System Sciences, 20(11), 4655-4671. 
http://doi.org/10.5194/hess-20-4655-2016

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
                                      description='''Plot sum of Si_mean of all parameters on map.''')
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

    import numpy          as np
    import datetime       as datetime
    import pandas         as pd
    import scipy.optimize as opt
    import time
    import glob           as glob
    import zipfile
    from scipy.stats import spearmanr

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
    import functions                                           # in lib/
    import netCDF4 as nc4
    #from   .general_functions   import logistic_offset_p      # in lib/
    #from   .fit_functions       import cost_square            # in lib/


    # Basin ID
    files = glob.glob("../data_out/*/results_nsets1000.nc")
    basin_ids = [ ff.split('/')[-2] for ff in files ]
    #raise ValueError('Basin ID (option -i) needs to given. Basin ID needs to correspond to ID of a CAMELS basin.')

    # -------------------------------------------------------------------------
    # Read basin metadata
    # -------------------------------------------------------------------------

    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac
    head = fread("../data_in/basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data_in/basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',cname=["lat","lon","area_km2","elevation_m","slope_deg","forest_frac","lat_gauge_deg","lon_gauge_deg"],sname=["basin_id","basin_name"])

    properties = ["lat","lon","area_km2","elevation_m","slope_deg","forest_frac","lat_gauge_deg","lon_gauge_deg"]
    
    dict_properties = {}
    for ibasin in range(len(meta_float)):
        dict_basin = {}
        dict_basin["name"]        = meta_string[ibasin][1]
        dict_basin[properties[0]] = meta_float[ibasin][0]
        dict_basin[properties[1]] = meta_float[ibasin][1]
        dict_basin[properties[2]] = meta_float[ibasin][2]
        dict_basin[properties[3]] = meta_float[ibasin][3]
        dict_basin[properties[4]] = meta_float[ibasin][4]
        dict_basin[properties[5]] = meta_float[ibasin][5]
        dict_basin[properties[6]] = meta_float[ibasin][6]
        dict_basin[properties[7]] = meta_float[ibasin][7]
    
        dict_properties[meta_string[ibasin][0]] = dict_basin

    # -------------------------------------------------------------------------
    # Read sensitivity results of (all) parameters
    # -------------------------------------------------------------------------
    print('')
    print('Reading xSSA sensitivity results ...')
    
    filename = "sa_for_markstrom_xSSA.dat"
    if os.path.exists(filename):

        sobol_indexes_dat = sread(filename, skip=1)
        sobol_indexes = {}
        for ss in sobol_indexes_dat:
            sobol_indexes[ss[0]] = np.float(ss[1])

    else:
        sobol_indexes = {}
        variable      = 'Q'
        for ibasin_id,basin_id in enumerate(basin_ids):

            print("Read sensitivities for: ",basin_id)

            nc4_file = "../data_out/"+basin_id+"/results_nsets1000.nc"
            nc4_in = nc4.Dataset(nc4_file, "r", format="NETCDF4")

            # ---------------------
            # sensitivity indexes: sobol_indexes['paras']['msi'][variable]
            # ---------------------
            analysis_type    = 'paras'       # 'paras', 'process_options', 'processes'
            sensi_index_type = 'msi'         # 'msi', 'msti', 'wsi', 'wsti' 

            ncvar_name = sensi_index_type+'_'+analysis_type.split('_')[-1]      # msi_paras, msi_options, msi_processes
            tmp_data   = nc4_in.groups[variable].variables[ncvar_name][:]
            sobol_indexes[basin_id] = np.sum(np.where(tmp_data<0.0,0.0,tmp_data))   # sum of all values (make negative ones 0.0)

            nc4_in.close()

        ff = open(filename, 'w')
        ff.write('basin_id,sum(msi_paras)\n')
        for ibasin in sobol_indexes:
            string = ibasin+','+astr(sobol_indexes[ibasin],prec=6)+'\n'
            ff.write(string)
        ff.close()

    min_sobol_indexes = np.min([ sobol_indexes[kk] for kk in sobol_indexes.keys() ] )
    max_sobol_indexes = np.max([ sobol_indexes[kk] for kk in sobol_indexes.keys() ] )

    print("min_sobol_indexes: ",min_sobol_indexes)
    print("max_sobol_indexes: ",max_sobol_indexes)

    # -------------------------------------------------------------------------
    # Read shape files from PMRS
    # -------------------------------------------------------------------------
    print('')
    print('Reading Markstrom HRU shapes ...')
    import shapefile
    from shapely.geometry import Polygon, MultiPolygon
    
    sf = shapefile.Reader("../data_supp/markstrom_HESS_2016/hrusAllConus/hrusAllConusDd")
    #myshp = open("../data_supp/markstrom_HESS_2016/hrusAllConus/hrusAllConusDd.shp", "rb")
    #mydbf = open("../data_supp/markstrom_HESS_2016/hrusAllConus/hrusAllConusDd.dbf", "rb")
    #r = shapefile.Reader(shp=myshp, dbf=mydbf)

    shapes = sf.shapes()
    nshapes = len(shapes)
    print("Number of Markstrom HRU shapes found: ",nshapes)

    # sf.fields
    #      [['hru_id_loc', 'N', 9, 0],     # hru_id in this region (unique per region; starts with 1)
    #       ['hru_id', 'N', 9, 0],         # hru_id overall (unique throughout whole dataset; starts with 1)
    #       ['region', 'C', 50, 0]]        # regions 1-18: 1-9, then 10 lower, 10 upper, then 11-18
    #                                      # The fact that region 10 (Missouri River Basin) is split is a real pain.
    #                                      # USGS didn't make this scheme up, but it follows the NHD plus region naming convention. 
    fields = ['hru_id_loc','hru_id','region']

    all_hru_shapes = {}
    regions = []
    for ishape in range(nshapes):

        rec = sf.record(ishape)      # e.g., [683, 17871, 'r04']
        hru_id_loc = rec[0]
        hru_id     = rec[1]
        region     = rec[2]
        regions.append(region)

        tmp_dict = {}
        tmp_dict['hru_id_loc'] = hru_id_loc
        tmp_dict['region']     = region
        # tmp_dict['geometry']   = shapes[ishape].points
        tmp_dict['geometry']   = sf.shape(ishape).__geo_interface__
        tmp_dict['bbox']       = shapes[ishape].bbox
        
        all_hru_shapes[hru_id] = tmp_dict

    # -------------------------------------------------------------------------
    # Read sensitivities for PMRS
    # -------------------------------------------------------------------------
    print('')
    print('Reading Markstrom sensitivities ...')
    regions = np.unique(regions)
    markstrom_sensi = {}
    for iregion in regions:
        markstrom_sensi[iregion] = fread('../data_supp/markstrom_HESS_2016/sensScores/julie/'+iregion+'_hru_outflowMeanSens.csv',skip=1)


    # -------------------------------------------------------------------------
    # Read PRMS results for all HRUs for each basin and average over all PRMS 35 variables over all HRUs for
    # file: ../data_supp/markstrom_HESS_2016/sensScores/<region>/hru_outflowMeanSens.csv
    #       lines: hru_id_loc for this <region>
    # -------------------------------------------------------------------------    
    filename = "sa_for_markstrom_FAST.dat"
    if os.path.exists(filename):

        print('')
        print('Reading FAST indexes from file ...')

        fast_indexes_dat = sread(filename, skip=1)
        fast_indexes = {}
        for ss in fast_indexes_dat:
            fast_indexes[ss[0]] = np.float(ss[1])

    else:

        print('')
        print('Deriving FAST indexes ...')
        
        fast_indexes = {}
        for ibasin_id,basin_id in enumerate(basin_ids):

            print("   ",basin_id)

            # read shape of our basin
            if os.path.exists("../data_in/data_obs/"+basin_id+"/shape_"+basin_id+"_coarse/shape_"+basin_id+"_coarse.shp"):
                sf = shapefile.Reader("../data_in/data_obs/"+basin_id+"/shape_"+basin_id+"_coarse/shape_"+basin_id+"_coarse")
            elif os.path.exists("../data_in/data_obs/"+basin_id+"/shape_"+basin_id+"_coarse.zip"):
                # unzip (puts files to current location)
                zip = zipfile.ZipFile("../data_in/data_obs/"+basin_id+"/shape_"+basin_id+"_coarse.zip")
                zip.extractall()
                # read
                sf = shapefile.Reader("shape_"+basin_id+"_coarse")
                # remove files 
                os.remove("shape_"+basin_id+"_coarse.dbf")
                os.remove("shape_"+basin_id+"_coarse.prj")
                os.remove("shape_"+basin_id+"_coarse.shp")
                os.remove("shape_"+basin_id+"_coarse.shx")
            else:
                print("../data_in/data_obs/"+basin_id+"/shape_"+basin_id+"_coarse")
                raise ValueError("Can not find shapefile.")

            # get points
            geometry_basin = sf.shape(0).__geo_interface__

            # get bbox
            bbox = sf.shape(0).bbox

            # find all HRUs that intersect with this shape
            p_xSSA = Polygon(geometry_basin['coordinates'][0])
            if not(p_xSSA.is_valid):
                # if for some reason this polygon is not valid, try buffer trick
                p_xSSA = p_xSSA.buffer(0.0)

            # check if an intersecting HRU will be found
            no_hru = True
            area_all_hru = 0.0
            sensi = 0.0

            for iHRU in all_hru_shapes: #range(100): #range(nshapes):

                area_this_hru = 0.0
                
                if all_hru_shapes[iHRU]['geometry']['type'] == 'MultiPolygon':
                    nmultis = len(all_hru_shapes[iHRU]['geometry']['coordinates'])
                    p_PRMS = []
                    for imulti in range(nmultis):
                        p_PRMS.append( Polygon(all_hru_shapes[iHRU]['geometry']['coordinates'][imulti][0]) )
                elif all_hru_shapes[iHRU]['geometry']['type'] == 'Polygon':
                    nmultis = 1
                    p_PRMS = [ Polygon(all_hru_shapes[iHRU]['geometry']['coordinates'][0]) ]
                else:
                    print('iHRU = ',iHRU,'    --> unknown geometry type')

                for imulti in range(nmultis):

                    if not(p_PRMS[imulti].is_valid):
                        # if for some reason this polygon is not valid, try buffer trick
                        p_PRMS[imulti] = p_PRMS[imulti].buffer(0.0)
                        
                    if (p_xSSA.intersects(p_PRMS[imulti])):

                        no_hru = False

                        # this fails:
                        # 06601200
                        # TopologyException: Input geom 0 is invalid: Self-intersection at or near point -109.62697453774797 49.520173297722323 at -109.62697453774797 49.520173297722323
                        p_intersect = p_xSSA.intersection(p_PRMS[imulti])
                        # print(p_intersect) 
                        # print('Basin: '+basin_id+' ---> ', p_intersect.area / p_PRMS[imulti].area * 100., '%   of HRU ',iHRU, '(part ',imulti+1,'/',nmultis,')')

                        # count areas that we account for in total
                        area_all_hru += p_intersect.area

                        # count areas we account for in this HRU (only different if HRU is split into multiple pieces)
                        area_this_hru += p_intersect.area

                # read in PMRS sensi results
                this_hru_region = all_hru_shapes[iHRU]['region']
                this_hru_id_loc = all_hru_shapes[iHRU]['hru_id_loc']
                sensi_from_file = np.sum(markstrom_sensi[this_hru_region][this_hru_id_loc-1])   # -1 because numberung starts with 1 not 0

                # scale them with size of this HRU compared to whole basin size
                sensi += sensi_from_file * area_this_hru    # should be devided by "p_xSSA.area"
                #                                           # but basins at border are not completely covered by Markstrom
                #                                           # so do at the end division by area_all_hru

            if no_hru:
                sensi = -9999.0
                area_all_hru = 1.0
            # else:
                # print("Area basin:        ", p_xSSA.area)
                # print("Area partial HRUs: ", area_all_hru)
                # print("Area Error:        ", np.abs(p_xSSA.area-area_all_hru)/p_xSSA.area * 100.,"%")

            # sensi
            fast_indexes[basin_id] = sensi / area_all_hru   # must be normalized by area
            #                                               # (should be  "p_xSSA.area"
            #                                               # but take "area_all_hru" for only partial basins)
            print('basin: ',basin_id,' FAST = ',fast_indexes[basin_id], '    xSSA = ',sobol_indexes[basin_id])

        # -------------------------------------------------------------------------
        # write those results to file
        # -------------------------------------------------------------------------
        ff = open(filename, 'w')
        ff.write('basin_id,sum(fast_paras)\n')
        for ibasin in fast_indexes:
            string = ibasin+','+astr(fast_indexes[ibasin],prec=6)+'\n'
            ff.write(string)
        ff.close()
        
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
dummy_rows  = 1
nrow        = 2    # # of rows of subplots per figure
ncol        = 5   # # of columns of subplots per figure
hspace      = 0.02         # x-space between subplots
vspace      = 0.02        # y-space between subplots
right       = 0.9         # right space on page
textsize    = 6           # standard text size
dxabc       = 0.0         # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
# dyabc       = -13       # % of (max-min) shift up from lower x-axis for a,b,c,... labels
dyabc       = 1.02         # % of (max-min) shift up from lower x-axis for a,b,c,... labels
dxsig       = 1.23        # % of (max-min) shift to the right from left y-axis for signature
dysig       = -0.05       # % of (max-min) shift up from lower x-axis for signature
dxtit       = 0           # % of (max-min) shift to the right from left y-axis for title
dytit       = 1.3         # % of (max-min) shift up from lower x-axis for title

lwidth      = 1.0         # linewidth
elwidth     = 1.0         # errorbar line width
alwidth     = 0.5         # axis line width
glwidth     = 0.5         # grid line width
msize       = 1.0         # marker size
mwidth      = 0.2         # marker edge width
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
llxbbox     = 0.0         # x-anchor legend bounding box
llybbox     = 1.0         # y-anchor legend bounding box
llrspace    = 0.          # spacing between rows in legend
llcspace    = 1.0         # spacing between columns in legend
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
    c  = color.get_brewer('rdylbu11', rgb=True)
    tmp = c.pop(5)   # rm yellow
    np.random.shuffle(c)
    
    #c.insert(2,c[2]) # same colour for both soil moistures
    ocean_color = (151/256., 183/256., 224/256.)
    # ocean_color = color.get_brewer('accent5', rgb=True)[-1]

    # rainbow colors
    cc = color.get_brewer('dark_rainbow_256', rgb=True)
    cc = cc[::-1] # reverse colors
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

    # colormap used by Markstrom (2016)
    cmap = mpl.cm.get_cmap('cubehelix_r')   #plt.get_cmap('cubehelix')

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


ifig = 0

# -------------------------------------------------------------------------
# Fig 1
#
ifig += 1
iplot = 0
print('Plot - Fig ', ifig)

fig = plt.figure(ifig)

iplot += 1

if True:

    # ----------------------------------------------------------------
    # (A) MARKSTROM MEAN RUNOFF (FIG 2)
    # ----------------------------------------------------------------
    
    #     [left, bottom, width, height]
    pos = [0.1,0.8,0.45,0.15]
    sub = fig.add_axes(pos, projection=ccrs.LambertConformal())
    #sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

    #sub.coastlines(resolution='10m')
    #sub.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # sub.add_feature(cfeature.LAND, linewidth=0.2)
    sub.add_feature(cfeature.COASTLINE, linewidth=0.2)
    sub.add_feature(cfeature.BORDERS,   linewidth=0.2)
    sub.add_feature(cfeature.LAKES,     linewidth=0.2, edgecolor='black')
    sub.add_feature(cfeature.OCEAN,     color=ocean_color)
    sub.set_extent([-140, -65, 22, 77], ccrs.PlateCarree())

    # *must* call draw in order to get the axis boundary used to add ticks:
    fig.canvas.draw()

    # Define gridline locations and draw the lines using cartopy's built-in gridliner:
    xticks = [-210, -190, -170, -150, -130, -110, -90, -70, -50, -30, -10, 10, 30, 50, 70, 90, 110, 130, 150, 170]
    yticks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    yticks = [10, 20, 30, 40, 50, 60, 70, 80]
    sub.gridlines(xlocs=xticks, ylocs=yticks,linewidth=0.1, color='gray', alpha=0.5, linestyle='-')

    # Label the end-points of the gridlines using the custom tick makers:
    sub.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    sub.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(sub, xticks)
    lambert_yticks(sub, yticks)

    # transform
    transform = ccrs.PlateCarree()._as_mpl_transform(sub)

    # add label with current number of basins
    sub.text(0.5,1.02,str2tex("PRMS (Markstrom et al., 2016)",usetex=usetex),horizontalalignment="center", verticalalignment="bottom",transform=sub.transAxes)

    # add label with current number of basins
    # sub.text(0.05,0.05,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(basin_ids))+"$",usetex=usetex),transform=sub.transAxes)

    # adjust frame linewidth 
    sub.outline_patch.set_linewidth(lwidth)

    coord_catch   = []
    for ibasin_id,basin_id in enumerate(basin_ids):

        shapefilename =  "wrong-directory"+basin_id+"/shape.dat"    # "observations/"+basin_id+"/shape.dat"

        # color based on sum of msi of paras
        icolor = 'red'
        max_sum_msi = 1.0
        min_sum_msi = 0.0
        isum_msi = fast_indexes[basin_id]       
        if isum_msi < min_sum_msi:
            icolor = cmap(0.0)
        elif isum_msi > max_sum_msi:
            icolor = cmap(1.0)
        else:
            icolor = cmap((isum_msi-min_sum_msi)/(max_sum_msi-min_sum_msi))        

        # if shapefile exists --> plot largest of the shapes
        if ( os.path.exists(shapefilename) ):
            # [lon lat]
            coords = fread(shapefilename, skip=1, separator=';')
            coord_catch.append(coords)
        
            # add catchment shape to plot
            nan_idx = np.where(np.isnan(coords[:,0]))[0]                                  # nan's, e.g., array([    0,     6, 16896, 16898])
            nshapes = len(nan_idx)-1                                                      #        e.g., 3
            longest_shape = np.where(np.diff(nan_idx)==np.max(np.diff(nan_idx)))[0][0]    #        e.g., #1
            for ishape in range(nshapes):

                start = nan_idx[ishape]+1
                end   = nan_idx[ishape+1]
                if ishape == longest_shape:
                    # plot only longest shape
                    #sub.add_patch(Polygon(np.transpose(map4(coords[start:end,0],coords[start:end,1])),
                    #                          facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))
                    sub.add_patch(Polygon(np.transpose([coords[start:end,0],coords[start:end,1]]),
                                              transform=ccrs.PlateCarree(), facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))

                    xxmin = np.nanmin(coords[start:end,0])
                    xxmax = np.nanmax(coords[start:end,0])
                    yymin = np.nanmin(coords[start:end,1])
                    yymax = np.nanmax(coords[start:end,1])

                    # if not(donolabel):
                    #     # annotate
                    #     #xpt, ypt  = map4(np.mean(coords[start:end,0]), np.mean(coords[start:end,1]))   # center of shape
                    #     xpt, ypt  = [ np.nanmean(coords[start:end,0]), np.nanmean(coords[start:end,1]) ]      # center of shape
                    #     x2,  y2   = (1.1,0.95-ibasin_id*0.1)                                            # position of text
                    #     transform = ccrs.PlateCarree()._as_mpl_transform(sub)
                    #     sub.annotate(basin_id,
                    #         xy=(xpt, ypt),   xycoords='data',
                    #         xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                    #         fontsize=8,
                    #         transform=transform,
                    #         verticalalignment='center',horizontalalignment='left',
                    #         arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                    #         zorder=400
                    #         )

            # print("Basin: ",basin_id)
            # print("   --> area      =  ",dict_properties[basin_id]["area_km2"])
            # print("   --> lon range = [",xxmin,",",xxmax,"]")
            # print("   --> lat range = [",yymin,",",yymax,"]")

        # shapefile doesnt exist only plot dot at location
        else:   
            # xpt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[0]
            # ypt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[1]
            xpt = dict_properties[basin_id]["lon"]
            ypt = dict_properties[basin_id]["lat"]
            x2,  y2   = (1.1,0.95-ibasin_id*0.1)
            
            sub.plot(xpt, ypt,
                     linestyle='None', marker='o', markeredgecolor=icolor, markerfacecolor=icolor,
                     transform=ccrs.PlateCarree(), 
                     markersize=0.7, markeredgewidth=0.0)

            # print("Basin: ",basin_id)
            # print("   --> lat/lon = [",dict_properties[basin_id]["lat"],",",dict_properties[basin_id]["lon"],"]")

    abc2plot(sub,dxabc/2,dyabc,iplot,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='bottom',horizontalalignment="left")

    # -------------------------------------------------------------------------
    # (1b) Colorbar
    # -------------------------------------------------------------------------
    iplot += 1
    csub    = fig.add_axes(position(1,1,1,hspace=hspace,vspace=vspace, left=0.2, right=0.45, top=0.778, bottom=0.772) )   # top=0.642, bottom=0.632

    cbar = mpl.colorbar.ColorbarBase(csub, norm=mpl.colors.Normalize(vmin=min_sum_msi, vmax=max_sum_msi), cmap=cmap, orientation='horizontal')
    cbar.set_label(str2tex("Sum of FAST main effects $\overline{S_i}$ [-]",usetex=usetex))

    msis= np.array([ sobol_indexes[basin_id] * 0.0 for basin_id in basin_ids ])              # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ADJUST WHEN I GET MARKSTROM DATA
    cticks = [ min_sum_msi+ii*(max_sum_msi-min_sum_msi)/10. for ii in range(10+1) ]  # cbar.get_ticks()      # tick labels
    percent_in_cat = np.diff([0]+ [ np.sum(msis < ctick) for ctick in cticks ])*100.0/np.shape(msis)[0]  
    
    # add percentages of basins with performance
    # for iitick,itick in enumerate(cticks):
    #     if iitick == 0:
    #         continue
    #     dist_from_middle = np.abs(iitick-((len(cticks))/2.))*1.0 / ((len(cticks)-1)/2.)
    #     if iitick<6:
    #         icolor = 'k' # (dist_from_middle,dist_from_middle,dist_from_middle)
    #     else:
    #         icolor = 'w' # (dist_from_middle,dist_from_middle,dist_from_middle)
    #     csub.text(1.0/(len(cticks)-1)*(iitick-0.5), 0.5,
    #                   str2tex(astr(percent_in_cat[iitick],prec=1)+'%',usetex=usetex),
    #                   color=icolor,
    #                   fontsize=textsize-3,
    #                   va='center',
    #                   ha='center',
    #                   rotation=0)





    # ----------------------------------------------------------------
    # (B) MAI sum of mSI of paras
    # ----------------------------------------------------------------
    
    #     [left, bottom, width, height]
    pos = [0.4,0.8,0.45,0.15]
    sub = fig.add_axes(pos, projection=ccrs.LambertConformal())
    #sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

    #sub.coastlines(resolution='10m')
    #sub.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # sub.add_feature(cfeature.LAND, linewidth=0.2)
    sub.add_feature(cfeature.COASTLINE, linewidth=0.2)
    sub.add_feature(cfeature.BORDERS,   linewidth=0.2)
    sub.add_feature(cfeature.LAKES,     linewidth=0.2, edgecolor='black')
    sub.add_feature(cfeature.OCEAN,     color=ocean_color)
    sub.set_extent([-140, -65, 22, 77], ccrs.PlateCarree())

    # *must* call draw in order to get the axis boundary used to add ticks:
    fig.canvas.draw()

    # Define gridline locations and draw the lines using cartopy's built-in gridliner:
    xticks = [-210, -190, -170, -150, -130, -110, -90, -70, -50, -30, -10, 10, 30, 50, 70, 90, 110, 130, 150, 170]
    yticks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    yticks = [10, 20, 30, 40, 50, 60, 70, 80]
    sub.gridlines(xlocs=xticks, ylocs=yticks,linewidth=0.1, color='gray', alpha=0.5, linestyle='-')

    # Label the end-points of the gridlines using the custom tick makers:
    sub.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    sub.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(sub, xticks)
    lambert_yticks(sub, yticks)

    # transform
    transform = ccrs.PlateCarree()._as_mpl_transform(sub)

    # add label with current number of basins
    sub.text(0.5,1.02,str2tex("Blended Raven (Mai et al., 2020)",usetex=usetex),horizontalalignment="center", verticalalignment="bottom",transform=sub.transAxes)

    # add label with current number of basins
    sub.text(0.05,0.05,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(basin_ids))+"$",usetex=usetex),transform=sub.transAxes)

    # adjust frame linewidth 
    sub.outline_patch.set_linewidth(lwidth)

    coord_catch   = []
    for ibasin_id,basin_id in enumerate(basin_ids):

        shapefilename =  "wrong-directory"+basin_id+"/shape.dat"    # "observations/"+basin_id+"/shape.dat"

        # color based on sum of msi of paras
        icolor = 'red'
        max_sum_msi = 1.0
        min_sum_msi = 0.0
        isum_msi = sobol_indexes[basin_id]
        if isum_msi < min_sum_msi:
            icolor = cmap(0.0)
        elif isum_msi > max_sum_msi:
            icolor = cmap(1.0)
        else:
            icolor = cmap((isum_msi-min_sum_msi)/(max_sum_msi-min_sum_msi))        

        # if shapefile exists --> plot largest of the shapes
        if ( os.path.exists(shapefilename) ):
            # [lon lat]
            coords = fread(shapefilename, skip=1, separator=';')
            coord_catch.append(coords)
        
            # add catchment shape to plot
            nan_idx = np.where(np.isnan(coords[:,0]))[0]                                  # nan's, e.g., array([    0,     6, 16896, 16898])
            nshapes = len(nan_idx)-1                                                      #        e.g., 3
            longest_shape = np.where(np.diff(nan_idx)==np.max(np.diff(nan_idx)))[0][0]    #        e.g., #1
            for ishape in range(nshapes):

                start = nan_idx[ishape]+1
                end   = nan_idx[ishape+1]
                if ishape == longest_shape:
                    # plot only longest shape
                    #sub.add_patch(Polygon(np.transpose(map4(coords[start:end,0],coords[start:end,1])),
                    #                          facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))
                    sub.add_patch(Polygon(np.transpose([coords[start:end,0],coords[start:end,1]]),
                                              transform=ccrs.PlateCarree(), facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))

                    xxmin = np.nanmin(coords[start:end,0])
                    xxmax = np.nanmax(coords[start:end,0])
                    yymin = np.nanmin(coords[start:end,1])
                    yymax = np.nanmax(coords[start:end,1])

                    # if not(donolabel):
                    #     # annotate
                    #     #xpt, ypt  = map4(np.mean(coords[start:end,0]), np.mean(coords[start:end,1]))   # center of shape
                    #     xpt, ypt  = [ np.nanmean(coords[start:end,0]), np.nanmean(coords[start:end,1]) ]      # center of shape
                    #     x2,  y2   = (1.1,0.95-ibasin_id*0.1)                                            # position of text
                    #     transform = ccrs.PlateCarree()._as_mpl_transform(sub)
                    #     sub.annotate(basin_id,
                    #         xy=(xpt, ypt),   xycoords='data',
                    #         xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                    #         fontsize=8,
                    #         transform=transform,
                    #         verticalalignment='center',horizontalalignment='left',
                    #         arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                    #         zorder=400
                    #         )

            # print("Basin: ",basin_id)
            # print("   --> area      =  ",dict_properties[basin_id]["area_km2"])
            # print("   --> lon range = [",xxmin,",",xxmax,"]")
            # print("   --> lat range = [",yymin,",",yymax,"]")

        # shapefile doesnt exist only plot dot at location
        else:   
            # xpt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[0]
            # ypt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[1]
            xpt = dict_properties[basin_id]["lon"]
            ypt = dict_properties[basin_id]["lat"]
            x2,  y2   = (1.1,0.95-ibasin_id*0.1)
            
            sub.plot(xpt, ypt,
                     linestyle='None', marker='o', markeredgecolor=icolor, markerfacecolor=icolor,
                     transform=ccrs.PlateCarree(), 
                     markersize=0.7, markeredgewidth=0.0)

            # print("Basin: ",basin_id)
            # print("   --> lat/lon = [",dict_properties[basin_id]["lat"],",",dict_properties[basin_id]["lon"],"]")

    abc2plot(sub,dxabc/2,dyabc,iplot,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='bottom',horizontalalignment="left")

    # -------------------------------------------------------------------------
    # (1b) Colorbar
    # -------------------------------------------------------------------------
    iplot += 1
    csub    = fig.add_axes(position(1,1,1,hspace=hspace,vspace=vspace, left=0.5, right=0.75, top=0.778, bottom=0.772) )   # top=0.642, bottom=0.632

    cbar = mpl.colorbar.ColorbarBase(csub, norm=mpl.colors.Normalize(vmin=min_sum_msi, vmax=max_sum_msi), cmap=cmap, orientation='horizontal')
    cbar.set_label(str2tex("Sum of Sobol' main effects $\overline{S_i}$ [-]",usetex=usetex))

    msis= np.array([ sobol_indexes[basin_id] for basin_id in basin_ids ])
    cticks = [ min_sum_msi+ii*(max_sum_msi-min_sum_msi)/10. for ii in range(10+1) ]  # cbar.get_ticks()      # tick labels
    percent_in_cat = np.diff([0]+ [ np.sum(msis < ctick) for ctick in cticks ])*100.0/np.shape(msis)[0]  
    
    # add percentages of basins with performance
    for iitick,itick in enumerate(cticks):
        if iitick == 0:
            continue
        dist_from_middle = np.abs(iitick-((len(cticks))/2.))*1.0 / ((len(cticks)-1)/2.)
        if iitick<6:
            icolor = 'k' # (dist_from_middle,dist_from_middle,dist_from_middle)
        else:
            icolor = 'w' # (dist_from_middle,dist_from_middle,dist_from_middle)
        csub.text(1.0/(len(cticks)-1)*(iitick-0.5), 0.5,
                      str2tex(astr(percent_in_cat[iitick],prec=1)+'%',usetex=usetex),
                      color=icolor,
                      fontsize=textsize-3,
                      va='center',
                      ha='center',
                      rotation=0)
        
    

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

