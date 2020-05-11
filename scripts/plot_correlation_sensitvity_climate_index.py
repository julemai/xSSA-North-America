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
#     run plot_correlation_sensitvity_climate_index.py -i "01010000 01010500 01013500 01014000 01015800 01018500 01019000 01020000 01021000 01021500 01030500 01031500 01033000 01033500 01034000 01037000 01041000 01042500 01043500 01045000 01049205 01049265 01049500 01053500 01053600"  -a "../data_in/basin_metadata/basin_climate_index_knoben_snow-raven.txt"

#    run plot_correlation_sensitvity_climate_index.py -g plot_correlation_sensitvity_climate_index_ -a "../data_in/basin_metadata/basin_climate_index_knoben_snow-raven.txt"


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
Plots climate indexes (aridity, seasonality, frac_precip_as_snow) against Sobol' total indexes per process.

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

    basin_ids            = None
    pngbase              = ''
    pdffile              = ''
    usetex               = False
    file_climate_indexes = None
    
    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Derives the climate indexes following Knoben et al. WRR (2018).''')
    parser.add_argument('-i', '--basin_ids', action='store',
                    default=basin_ids, dest='basin_ids', metavar='basin_ids',
                    help='Basin ID of basins to analyze. Mandatory. (default: None).')
    parser.add_argument('-g', '--pngbase', action='store',
                    default=pngbase, dest='pngbase', metavar='pngbase',
                    help='Name basis for png output files (default: open screen window).')
    parser.add_argument('-p', '--pdffile', action='store',
                    default=pdffile, dest='pdffile', metavar='pdffile',
                    help='Name of pdf output file (default: open screen window).')
    parser.add_argument('-t', '--usetex', action='store_true', default=usetex, dest="usetex",
                    help="Use LaTeX to render text in pdf.")
    parser.add_argument('-a', '--file_climate_indexes', action='store',
                    default=file_climate_indexes, dest='file_climate_indexes', metavar='file_climate_indexes',
                    help='File that contains climate indixes and colors. Mandatory. (default: None).')

    args                 = parser.parse_args()
    basin_ids            = args.basin_ids
    pngbase              = args.pngbase
    pdffile              = args.pdffile
    usetex               = args.usetex
    file_climate_indexes = args.file_climate_indexes

    if (file_climate_indexes is None):
        raise ValueError('File that contains climate indexes for each basin needs to be specified!\n E.g., "../data_in/basin_metadata/basin_climate_index_knoben_snow-raven.txt"')

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
    from scipy.stats import spearmanr

    t1 = time.time()
    
    from   fread                import fread                   # in lib/
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


    # Basin ID need to be specified
    if basin_ids is None:

        files = glob.glob("../data_out/*/results_nsets1000.nc")
        basin_ids = [ ff.split('/')[-2] for ff in files ]
        #raise ValueError('Basin ID (option -i) needs to given. Basin ID needs to correspond to ID of a CAMELS basin.')

    else:

        # convert basin_ids to list 
        basin_ids = basin_ids.strip()
        basin_ids = " ".join(basin_ids.split())
        basin_ids = basin_ids.split()
        # print('basin_ids: ',basin_ids)

    #stop

    # -------------------------------------------------------------------------
    # Read basin metadata
    # -------------------------------------------------------------------------

    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac
    head = fread("../data_in/basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data_in/basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',cname=["lat","lon","area_km2","elevation_m","slope_deg","forest_frac"],sname=["basin_id","basin_name"])

    properties = ["lat","lon","area_km2","elevation_m","slope_deg","forest_frac"]
    
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
    
        dict_properties[meta_string[ibasin][0]] = dict_basin

    # ----------------------------------
    # read climate indicators
    # ----------------------------------
    climate_indexes = {}
    indicators = ['aridity', 'seasonality', 'precip_as_snow']

    #file_climate_indexes = '../data_in/basin_metadata/basin_climate_index_knoben_snow-raven.txt'
    ff = open(file_climate_indexes, "r")
    lines = ff.readlines()
    ff.close()

    for basin_id in basin_ids:
        found = False
        for ill,ll in enumerate(lines):
            if ill > 0:
                tmp = ll.strip().split(';')
                if (tmp[0] == basin_id):
                    found = True
                    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac 

                    climate_index = {}
                    climate_index[indicators[0]]    = np.float(tmp[1].strip())
                    climate_index[indicators[1]]    = np.float(tmp[2].strip())
                    climate_index[indicators[2]]    = np.float(tmp[3].strip())
                    climate_index['red']            = np.float(tmp[4].strip())
                    climate_index['green']          = np.float(tmp[5].strip())
                    climate_index['blue']           = np.float(tmp[6].strip())

                    climate_indexes[str(basin_id)]  = climate_index

        if not(found):
            raise ValueError('Basin ID not found in '+file_climate_indexes)



    # -------------------------------------------------------------------------
    # Read sensitivity results of (all) processes
    # -------------------------------------------------------------------------
    # names of processes
    processes = ['Infiltration','Quickflow','Evaporation','Baseflow','Snow\n Balance', 'Convolution\n (Surface Runoff)', 'Convolution\n (Delayed Runoff)', 'Potential\n Melt', 'Percolation', 'Rain-Snow\n Partitioning', 'Precipitation\n Correction']
    sobol_indexes = {}
    variable      = 'Q'
    for ibasin_id,basin_id in enumerate(basin_ids):

        # import pickle
        # picklefile = "../data_out/"+basin_id+"/sensitivity_nsets1000.pkl"
        # setup         = pickle.load( open( picklefile, "rb" ) )
        # # sobol_indexes.append( setup['sobol_indexes']['processes']['wsti']['Q'] )
        # sobol_indexes[basin_id] = np.array(setup['sobol_indexes']['processes']['wsti']['Q'])

        nc4_file = "../data_out/"+basin_id+"/results_nsets1000.nc"
        nc4_in = nc4.Dataset(nc4_file, "r", format="NETCDF4")

        # ---------------------
        # sensitivity indexes: sobol_indexes['paras']['msi'][variable]
        # ---------------------
        analysis_type    = 'processes'   # 'paras', 'process_options', 
        sensi_index_type = 'wsti'        # 'msi', 'msti', 'wsi', 

        ncvar_name = sensi_index_type+'_'+analysis_type.split('_')[-1]      # msi_paras, msi_options, msi_processes
        sobol_indexes[basin_id] = nc4_in.groups[variable].variables[ncvar_name][:]

        nc4_in.close()

    min_sobol_indexes = np.min([ sobol_indexes[kk] for kk in sobol_indexes.keys() ] )
    max_sobol_indexes = np.max([ sobol_indexes[kk] for kk in sobol_indexes.keys() ] )
        
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
nrow        = len(processes)+dummy_rows    # # of rows of subplots per figure
ncol        = len(indicators)+len(properties)   # # of columns of subplots per figure
hspace      = 0.02         # x-space between subplots
vspace      = 0.02        # y-space between subplots
right       = 0.9         # right space on page
textsize    = 6           # standard text size
dxabc       = 1.0         # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
# dyabc       = -13       # % of (max-min) shift up from lower x-axis for a,b,c,... labels
dyabc       = 0.0         # % of (max-min) shift up from lower x-axis for a,b,c,... labels
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

# -----------------------------------------
# Correlation of process sensitivities with climate indicators
# -----------------------------------------

infil_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[20]   #  [20]
quick_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[43]   #  [55] 
evapo_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[66]   #  [80] 
basef_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[89]   #  [105]
snowb_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[112]  #  [130]
convs_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[135]  #  [155]
convd_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[157]  #  [180]
potme_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[179]  #  [205]
perco_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[201]  #  [230]
rspar_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[223]  #  [230]
rscor_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[255]  #  [230]
    
mcol = [infil_color, quick_color, evapo_color, basef_color, snowb_color, convs_color, convd_color, potme_color, perco_color, rspar_color, rscor_color ] 

for iprocess, process in enumerate(processes):
    for iindicator, indicator in enumerate(indicators):

        iplot += 1

        sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace))

        thresh_p_as_snow = [0.0,0.1]   # plot only basins with P as snow in this range; all = [0.0,1.0]
        
        wsti  = [ sobol_indexes[basin_id][iprocess]    for basin_id in basin_ids if (climate_indexes[basin_id]['precip_as_snow'] >= thresh_p_as_snow[0]) and (climate_indexes[basin_id]['precip_as_snow'] <= thresh_p_as_snow[1]) ]
        indic = [ climate_indexes[basin_id][indicator] for basin_id in basin_ids if (climate_indexes[basin_id]['precip_as_snow'] >= thresh_p_as_snow[0]) and (climate_indexes[basin_id]['precip_as_snow'] <= thresh_p_as_snow[1]) ]
        mark1 = sub.plot(indic,wsti)

        # spearman_rank_correlation
        coef, p = spearmanr(indic, wsti)
        if np.abs(coef) > 0.65:
            fontweight = 1000
            color='k'

            # --------------------------
            # fit logistic function
            # --------------------------
            xx = np.array(indic)
            yy = np.array(wsti)
            yy_norm = (np.array(wsti) - np.min(wsti)) / (np.max(wsti) - np.min(wsti))   # normalization leads to better fitting results
            idx = np.argsort(xx)
            xx = xx[idx]
            yy_norm = yy_norm[idx]
            inflection_point = np.mean(xx[ np.argmin( np.abs(yy_norm - np.median(yy_norm)) ) ])
            pini=np.array([np.mean(yy_norm[-100:]), (np.mean(yy_norm[-100:])-np.mean(yy_norm[:100]))/(np.mean(xx[-100:])-np.mean(xx[:100])), inflection_point, np.mean(yy_norm[:100])]) # [y-max, steepness, inflection point, offset]
            popti, f, d = opt.fmin_l_bfgs_b(functions.fit_functions.cost_square, pini,
                                        args=(functions.general_functions.logistic_offset_p,xx,yy_norm),
                                        approx_grad=1,
                                        bounds=[(None,None),(None,None),(None,None),(None,None)],#,(0.,0.2)],
                                        iprint=0, disp=0)
            print('')
            print('Indicator: ',indicator, '(rho = ',astr(coef,prec=2),')')
            print('Initial guess: ', astr(pini,prec=4))
            print('Params of logistic function: ', astr(popti,prec=4))

            if np.isnan(pini[2]):
                print("np.median(yy_norm) = ",np.median(yy_norm))
                print("np.argmin( np.abs(yy_norm - np.median(yy_norm)) ) = ",np.argmin( np.abs(yy_norm - np.median(yy_norm)) ))
                print("xx[ np.argmin( np.abs(yy_norm - np.median(yy_norm)) ) ] = ",xx[ np.argmin( np.abs(yy_norm - np.median(yy_norm)) ) ])

            # redo normalization of y values
            popti[0] = popti[0] * (np.max(wsti) - np.min(wsti))
            popti[3] = popti[3] * (np.max(wsti) - np.min(wsti)) + np.min(wsti)

            # plot fitted line
            fit_wsti = functions.general_functions.logistic_offset_p(xx,popti)
            line2    = plt.plot(xx, fit_wsti)
            plt.setp(line2, linestyle='-', linewidth=lwidth, color='k', marker='None', label=str2tex('$L$',usetex=usetex))

            
        else:
            fontweight = 'normal'
            color = (0.7,0.7,0.7)
        sub.text(0.5, 1.0, str2tex('$\\rho = '+astr(coef,prec=2)+'$',usetex=usetex), transform=sub.transAxes,
                 rotation=0, fontsize=textsize-1, weight=fontweight, color=color,
                 horizontalalignment='center', verticalalignment='bottom')

        # x-labels only last row
        if ((iplot-1) // ncol == nrow - dummy_rows - 1):
            xlabel=str2tex(indicator.replace('_', ' ').
                                            replace('aridity',         'aridity [-]').
                                            replace('seasonality',     'seasonality [-]').
                                            replace('precip as snow',  'P as snow [-]'),usetex=usetex)
        else:
            xlabel=''

        # y-labels only first column in every second row
        if ((iplot) % ncol == 1 and ((iplot-1) // ncol) % 2 == 1):
            ylabel=str2tex("Total Sobol' Index $ST_i$",usetex=usetex)
        else:
            ylabel=''

        # x-ticks only in last row
        if ((iplot-1) // ncol < nrow - dummy_rows - 1):
            # sub.axes.get_xaxis().set_visible(False)
            sub.axes.get_xaxis().set_ticks([])
        else:
            plt.xticks(rotation=90)

        # y-ticks only in first column
        if (iplot-1)%ncol != 0:
            # sub.axes.get_yaxis().set_visible(False)
            sub.axes.get_yaxis().set_ticks([])

        plt.setp(mark1, linestyle='None', color=mcol[iprocess], linewidth=0.0,
             marker='o', markeredgecolor=mcol[iprocess], markerfacecolor='None',
             markersize=msize, markeredgewidth=mwidth,
             label=str2tex(process,usetex=usetex))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    for ipropertie, propertie in enumerate(properties):

        iplot += 1

        sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace))

        wsti  = [ sobol_indexes[basin_id][iprocess]    for basin_id in basin_ids ]
        prop = [ dict_properties[basin_id][propertie]  for basin_id in basin_ids ]
        
        if propertie == 'area_km2':
            mark1 = sub.semilogx(prop,wsti) 
        else:
            mark1 = sub.plot(prop,wsti)

        # spearman_rank_correlation
        coef, p = spearmanr(prop, wsti)
        if np.abs(coef) > 0.65:
            fontweight = 1000
            color='k'

            # --------------------------
            # fit logistic function
            # --------------------------
            xx = np.array(prop)
            yy = np.array(wsti)
            yy_norm = (np.array(wsti) - np.min(wsti)) / (np.max(wsti) - np.min(wsti))   # normalization leads to better fitting results
            idx = np.argsort(xx)
            xx = xx[idx]
            yy = yy[idx]
            yy_norm = yy_norm[idx]
            inflection_point = np.mean(xx[ np.argmin( np.abs(yy_norm - np.median(yy_norm)) ) ])
            pini=np.array([np.mean(yy_norm[-100:]), (np.mean(yy_norm[-100:])-np.mean(yy_norm[:100]))/(np.mean(xx[-100:])-np.mean(xx[:100])), inflection_point, np.mean(yy_norm[:100])]) # [y-max, steepness, inflection point, offset]
            popti, f, d = opt.fmin_l_bfgs_b(functions.fit_functions.cost_square, pini,
                                        args=(functions.general_functions.logistic_offset_p,xx,yy_norm),
                                        approx_grad=1,
                                        bounds=[(None,None),(None,None),(None,None),(None,None)],#,(0.,0.2)],
                                        iprint=0, disp=0)
            print('')
            print('Property: ',propertie, '(rho = ',astr(coef,prec=2),')')
            print('Initial guess: ', astr(pini,prec=4))
            print('Params of logistic function: ', astr(popti,prec=4))

            if np.isnan(pini[2]):
                print("np.median(yy_norm) = ",np.median(yy_norm))
                print("np.argmin( np.abs(yy_norm - np.median(yy_norm)) ) = ",np.argmin( np.abs(yy_norm - np.median(yy_norm)) ))
                print("xx[ np.argmin( np.abs(yy_norm - np.median(yy_norm)) ) ] = ",xx[ np.argmin( np.abs(yy_norm - np.median(yy_norm)) ) ])

            # redo normalization of y values
            popti[0] = popti[0] * (np.max(wsti) - np.min(wsti))
            popti[3] = popti[3] * (np.max(wsti) - np.min(wsti)) + np.min(wsti)

            # plot fitted line
            fit_wsti = functions.general_functions.logistic_offset_p(xx,popti)
            line2    = plt.plot(xx, fit_wsti)
            plt.setp(line2, linestyle='-', linewidth=lwidth, color='k', marker='None', label=str2tex('$L$',usetex=usetex))
        else:
            fontweight = 'normal'
            color = (0.7,0.7,0.7)
        sub.text(0.5, 1.0, str2tex('$\\rho = '+astr(coef,prec=2)+'$',usetex=usetex), transform=sub.transAxes,
                 rotation=0, fontsize=textsize-1, fontweight=fontweight, color=color,
                 horizontalalignment='center', verticalalignment='bottom')

        # x-labels only last row
        if ((iplot-1) // ncol == nrow - dummy_rows - 1):
            xlabel=str2tex(propertie.replace('_', ' ').
                                            replace(' m',    ' [m]').
                                            replace(' km2',  ' [km$^2$]').
                                            replace(' frac', ' [-]').
                                            replace(' deg',  ' [$^\circ$]').
                                            replace('lat',   'lat [$^\circ$N]').
                                            replace('lon',   'lon [$^\circ$W]'),usetex=usetex)
        else:
            xlabel=''

        # y-labels only first column in every second row
        if ((iplot) % ncol == 1 and ((iplot-1) // ncol) % 2 == 1):
            ylabel=str2tex("Total Sobol' Index $ST_i$",usetex=usetex)
        else:
            ylabel=''

        # x-ticks only in last row
        if ((iplot-1) // ncol < nrow - dummy_rows - 1):
            # sub.axes.get_xaxis().set_visible(False)
            sub.axes.get_xaxis().set_ticks([])
        else:
            plt.xticks(rotation=90)

        # y-ticks only in first column
        if (iplot-1)%ncol != 0:
            # sub.axes.get_yaxis().set_visible(False)
            sub.axes.get_yaxis().set_ticks([])

        plt.setp(mark1, linestyle='None', color=mcol[iprocess], linewidth=0.0,
             marker='o', markeredgecolor=mcol[iprocess], markerfacecolor='None',
             markersize=msize, markeredgewidth=mwidth,
             label=str2tex(process,usetex=usetex))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if (iplot % ncol == 0):  # last column
        sub.text(1.2, 0.5, str2tex(process.replace('_', ' '),usetex=usetex), transform=sub.transAxes,
                 rotation=90, fontsize=textsize,
                 horizontalalignment='center', verticalalignment='center')
    

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

