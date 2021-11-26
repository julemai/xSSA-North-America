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
#     run figure_5_nse_gt_0.5_no-lat-lon_1-or-2pred.py

#    run figure_5_nse_gt_0.5_no-lat-lon_1-or-2pred.py -g figure_5_nse_gt_0.5_no-lat-lon_1-or-2pred


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
                                      description='''Plot derived and predicted sensitivities of processes based on 100 calibration/validation trials.''')
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
    sys.path.append(dir_path+'/../scripts/lib')

    import numpy          as np
    import datetime       as datetime
    import pandas         as pd
    import scipy.optimize as opt
    import time
    import glob           as glob
    import json
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
    from   scipy.stats          import pearsonr
    #from   .general_functions   import logistic_offset_p      # in lib/
    #from   .fit_functions       import cost_square            # in lib/


    # -------------------------------------------------------------------------
    # Read results from trials
    # -------------------------------------------------------------------------

    # regression fittings using 2 predictors are done using Mathematica
    # --> /Users/j6mai/Documents/GitHub/xSSA-North-America/scripts/fitting_models_to_wSTi.nb
    # and dumped to files:
    # --> /Users/j6mai/Documents/GitHub/Publications/mai_xSSA-North-America/data/mathematica/*.csv

    # ------------------------------
    # read results from regression using all results and trying to find best set of 1 or 2 predictors
    # (2 predictors are taken if R2 is at least 0.05 higher than with 1 predictor)
    # ------------------------------
    # meta_float, meta_string = fsread("../data/mathematica/nse_gt_0.5_no-lat-lon_1-or-2pred/results_1-or-2_predictors_nse_gt_0.5_no-lat-lon_postprocessed.csv",
    #                                      skip=0,separator=',',snc=4,nc=1, comment='#')

    ff = open("../data/mathematica/nse_gt_0.5_no-lat-lon_1-or-2pred_with_CV/results_1-or-2_predictors_nse_gt_0.5_no-lat-lon_with_CV_postprocessed.csv")
    content = ff.readlines()
    ff.close()

    meta_float = []
    meta_string = []
    for cc in content:

        tmp = [ ii.replace('"','') for ii in cc.strip().replace("'","").split(',') ]
        meta_float.append(float(tmp[-1]))
        meta_string.append(tmp[0:-1])

    base_rcorr    = np.array(meta_float) # .flatten()
    processes     = np.array([ ii[0].replace('"','').split('_')[1] for ii in meta_string ])                   # e.g. "wSTi_infiltration" --> "infiltration"
    predictors    = [ [ ii[iii].replace('"','') for iii in range(1,len(ii)-1) ] for ii in meta_string ]      # e.g. ["area_km2","frac_p_as_snow"]
    base_function = np.array([ [ ii[-1].replace('"','') ] for ii in meta_string ])                             # e.g. 0.34 - 0.11*x^2 - 0.91*y + 0.02*x^2*y

    # ------------------------------
    # derive Pearson correlation coefficients for table in paper
    # ------------------------------
    print("")
    print("-----------------")
    print("For table in publication")
    print("-----------------")
    header = fread('../data/derived/properties_vs_sensitivity_nse_gt_0.5.csv',skip=1,cskip=1,header=True)
    idx_wsti = [ iii for iii,ii in enumerate(header) if ( ('wSTi' in ii) and (ii.split('_')[1] in processes) ) ] # columns with wSTi and process that is analysed
    idx_prop = [ iii for iii,ii in enumerate(header) if not('wSTi' in ii) ] # idx of properties
    properti_3316 = fread('../data/derived/properties_vs_sensitivity_nse_gt_0.5.csv',skip=1,cskip=1)[:,idx_prop]
    obs_wSTi_3316 = fread('../data/derived/properties_vs_sensitivity_nse_gt_0.5.csv',skip=1,cskip=1)[:,idx_wsti]

    nbasins = np.shape(obs_wSTi_3316)[0]
    nprocesses = np.shape(obs_wSTi_3316)[1]
    sim_wSTi_3316 = np.ones([nbasins,nprocesses]) * -9999.

    for iprocess,process in enumerate(processes):

        npred = len(predictors[iprocess])

        if (npred == 2):
            idx_pred1 = np.where(np.array(header) == predictors[iprocess][0])[0][0]
            idx_pred2 = np.where(np.array(header) == predictors[iprocess][1])[0][0]

            x = properti_3316[:,idx_pred1]
            y = properti_3316[:,idx_pred2]

            if predictors[iprocess][0] == 'area_km2':
                x = np.log10(x)
            if predictors[iprocess][1] == 'area_km2':
                y = np.log10(y)
            if predictors[iprocess][0] == 'annual_ndays_wet_snow_1':
                x = x / 365.
            if predictors[iprocess][1] == 'annual_ndays_wet_snow_1':
                y = y / 365.
        if (npred == 1):
            idx_pred1 = np.where(np.array(header) == predictors[iprocess][0])[0][0]

            x = properti_3316[:,idx_pred1]

            if predictors[iprocess][0] == 'area_km2':
                x = np.log10(x)
            if predictors[iprocess][0] == 'annual_ndays_wet_snow_1':
                x = x / 365.

        sim_wSTi_3316[:,iprocess] = eval(base_function[iprocess][0].replace('*^','*10**').replace('^','**'))

        corrcoeff = pearsonr(sim_wSTi_3316[:,iprocess],obs_wSTi_3316[:,iprocess])[0]

        ysim = sim_wSTi_3316[:,iprocess]
        yobs = obs_wSTi_3316[:,iprocess]
        SS_Residual        = np.sum((yobs-ysim)**2)
        SS_Total           = np.sum((yobs-np.mean(yobs))**2)
        r_squared          = 1 - (float(SS_Residual))/SS_Total                        # RSquare = Coeff of Determination
        adjusted_r_squared = 1 - (1-r_squared)*(len(yobs)-1)/(len(yobs)-npred-1)      # adjusted RSquare = Coeff of Determination

        print("Pearson r for {:>20} = {:8.5f}   (based on {:4d} basins)".format(process,corrcoeff,nbasins))
        print("R2        for {:>20} = {:8.5f}   (based on {:4d} basins)".format(process,r_squared,nbasins))
        print("Adj. R2   for {:>20} = {:8.5f}   (based on {:4d} basins)".format(process,adjusted_r_squared,nbasins))
        print("")

    print("")

    # ------------------------------
    # read obs/mod from 100 trials for each process
    # ------------------------------
    modes   = ["obs","mod"]
    calvals = ["calib","valid"]

    trial_samples = [ [ [ [] for calval in calvals ] for mode in modes ] for process in processes ]

    for iprocess,process in enumerate(processes):
        for imode,mode in enumerate(modes):
            for icalval,calval in enumerate(calvals):

                # shape: trial_samples[nprocesses][2=obs/mod][2=cal/val][npoints,100]   # npoints = 66% in cal and 33% in val
                trial_samples[iprocess][imode][icalval] = fread("../data/mathematica/nse_gt_0.5_no-lat-lon_1-or-2pred_with_CV/trials_wSTi_"+process+"_wSti_"+mode+"_"+calval+".csv",skip=1)

    ntrials = np.shape(trial_samples[0][0][0])[1]
    nprocesses = len(trial_samples)

    # -------------------------------------------------------------------------
    # Read sensitivity results of (all) processes
    # -------------------------------------------------------------------------
    # names of processes
    processes_clear         = ['Infiltration','Quickflow','Evaporation','Baseflow','Snow\n Balance',
                             'Convolution\n (Surface Runoff)', 'Convolution\n (Delayed Runoff)',
                             'Potential\n Melt', 'Percolation',
                             'Rain-Snow\n Partitioning', 'Precipitation\n Correction']
    processes_clear         = ['Infiltration','Quickflow','Evaporation','Snow Balance',
                             'Convolution (srfc runoff)',
                             'Potential Melt', 'Percolation',
                             'Rain-Snow Partitioning', 'Precipitation Correction']

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
nrow        = 5           # # of rows of subplots per figure
ncol        = 3           # # of columns of subplots per figure
hspace      = 0.06        # x-space between subplots
vspace      = 0.04        # y-space between subplots
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
    #else:
    #      mpl.rc('font',**{'family':'serif','serif':['Palatino']})
    #     mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #     #mpl.rc('font',**{'family':'serif','serif':['times']})
    mpl.rc('text.latex') #, unicode=True)
elif (outtype == 'png'):
    mpl.use('Agg') # set directly after import matplotlib
    import matplotlib.pyplot as plt
    mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
    if usetex:
        mpl.rc('text', usetex=True)
    #else:
    #    mpl.rc('font',**{'family':'serif','serif':['Palatino']})
    #     mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #     #mpl.rc('font',**{'family':'serif','serif':['times']})
    mpl.rc('text.latex') #, unicode=True)
    mpl.rc('savefig', dpi=dpi, format='png')
else:
    import matplotlib.pyplot as plt
    mpl.rc('figure', figsize=(4./5.*8.27,4./5.*11.69)) # a4 portrait
mpl.rc('font', size=textsize)
mpl.rc('lines', linewidth=lwidth, color='black')
mpl.rc('axes', linewidth=alwidth, labelcolor='black')
mpl.rc('path', simplify=False) # do not remove

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Circle, Polygon
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

    # green-pink colors
    cc = color.get_brewer('piyg10', rgb=True)
    low_cc = tuple([1.0,1.0,1.0])
    del cc[0]  # drop darkest two pink color
    del cc[0]  # drop darkest two pink color
    cc = list([low_cc])+cc      # prepend "white"
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
mcol = [infil_color, quick_color, evapo_color,              snowb_color, convs_color,              potme_color, perco_color, rspar_color, rscor_color ]

min_wsti_to_plot = 0.01 # do not consider or plot datapoints with wSTI < min_wsti_to_plot

for iprocess, process in enumerate(processes[0:]):

    print('Process: ',process)

    iplot += 1

    print("plot ",iplot,"  --> pos = ",position(nrow,ncol,iplot,hspace=hspace,vspace=vspace))
    sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace))

    # set background color to the light gey used for low frequencies
    sub.set_facecolor(low_cc)

    icalval=0
    xx_calib = trial_samples[iprocess][0][icalval] # "observed" wSTi ==> shape(npoints,trials)
    yy_calib = trial_samples[iprocess][1][icalval] #  predicted wSTi ==> shape(npoints,trials)

    icalval=1
    xx_valid = trial_samples[iprocess][0][icalval] # "observed" wSTi ==> shape(npoints,trials)
    yy_valid = trial_samples[iprocess][1][icalval] #  predicted wSTi ==> shape(npoints,trials)

    maxval_xy = np.max([np.max(xx_calib),np.max(xx_valid),np.max(yy_calib),np.max(yy_valid)]) * 1.1

    # spearman_rank_correlation (of each trial)
    #coef_calib = np.array([ spearmanr(np.array(xx_calib)[:,itrial], np.array(yy_calib)[:,itrial]).correlation for itrial in range(ntrials) ])
    #coef_valid = np.array([ spearmanr(np.array(xx_valid)[:,itrial], np.array(yy_valid)[:,itrial]).correlation for itrial in range(ntrials) ])

    # pearson corr coeffi (of each trial)
    coef_calib = np.array([ pearsonr(np.array(xx_calib)[:,itrial], np.array(yy_calib)[:,itrial])[0] for itrial in range(ntrials) ])
    coef_valid = np.array([ pearsonr(np.array(xx_valid)[:,itrial], np.array(yy_valid)[:,itrial])[0] for itrial in range(ntrials) ])

    # mean absolute error (of each trial)
    mae_calib = np.mean(np.abs(xx_calib-yy_calib),axis=0)
    mae_valid = np.mean(np.abs(xx_valid-yy_valid),axis=0)

    # R**2 and adjusted R**2 (of each trial)
    r2_calib    = np.array([ -9999. for itrial in range(ntrials) ])
    r2adj_calib = np.array([ -9999. for itrial in range(ntrials) ])
    r2_valid    = np.array([ -9999. for itrial in range(ntrials) ])
    r2adj_valid = np.array([ -9999. for itrial in range(ntrials) ])
    for itrial in range(ntrials):

        npred = len(predictors[iprocess])
        ysim = np.array(yy_calib)[:,itrial]
        yobs = np.array(xx_calib)[:,itrial]
        SS_Residual        = np.sum((yobs-ysim)**2)
        SS_Total           = np.sum((yobs-np.mean(yobs))**2)
        r_squared          = 1 - (float(SS_Residual))/SS_Total                        # RSquare = Coeff of Determination
        adjusted_r_squared = 1 - (1-r_squared)*(len(yobs)-1)/(len(yobs)-npred-1)      # adjusted RSquare = Coeff of Determination

        r2_calib[itrial]    = r_squared
        r2adj_calib[itrial] = adjusted_r_squared

        npred = len(predictors[iprocess])
        ysim = np.array(yy_valid)[:,itrial]
        yobs = np.array(xx_valid)[:,itrial]
        SS_Residual        = np.sum((yobs-ysim)**2)
        SS_Total           = np.sum((yobs-np.mean(yobs))**2)
        r_squared          = 1 - (float(SS_Residual))/SS_Total                        # RSquare = Coeff of Determination
        adjusted_r_squared = 1 - (1-r_squared)*(len(yobs)-1)/(len(yobs)-npred-1)      # adjusted RSquare = Coeff of Determination

        r2_valid[itrial]    = r_squared
        r2adj_valid[itrial] = adjusted_r_squared


    # mark1 = sub.plot(xx_valid.flatten(),yy_valid.flatten())
    # color=mcol[iprocess]
    # color='0.7'
    # plt.setp(mark1, linestyle='None', color=color, linewidth=0.0, alpha=0.8,
    #      marker='o', markeredgecolor=color, markerfacecolor='None',
    #      markersize=msize, markeredgewidth=mwidth,
    #      label=str2tex(process,usetex=usetex))

    inorm = 'pow'

    if inorm == 'log':
        min_sti = 0.01
        max_sti = 1.0
        norm = mcolors.LogNorm(min_sti,max_sti)
    elif inorm == 'pow':
        pow_lambda = 0.2
        max_pow    = 3000.
        norm = mcolors.PowerNorm(pow_lambda,vmin=0,vmax=max_pow)
    else:
        raise ValueError('Norm for colormap not known.')

    hist1 = sub.hist2d(xx_valid.flatten(),yy_valid.flatten(), bins=30, norm=norm, cmap=cmap)

    # title = process
    sub.text(0.5, 1.0, str2tex(processes_clear[iprocess],usetex=usetex), transform=sub.transAxes,
                 rotation=0, fontsize=textsize+1,
                 horizontalalignment='center', verticalalignment='bottom')

    # sub.text(0.96, 0.12, str2tex('$r_{cal} = '+astr(np.round(np.mean(coef_calib),3),prec=3)+' \pm '+astr(np.round(np.std(coef_calib),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
    #              rotation=0, fontsize=textsize-2,
    #              horizontalalignment='right', verticalalignment='bottom')
    # sub.text(0.96, 0.04, str2tex('$r_{val} = '+astr(np.round(np.mean(coef_valid),3),prec=3)+' \pm '+astr(np.round(np.std(coef_valid),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
    #              rotation=0, fontsize=textsize-2,
    #              horizontalalignment='right', verticalalignment='bottom')
    sub.text(0.96, 0.10, str2tex('$R^2_{adj,cal} = '+astr(np.round(np.mean(r2adj_calib),3),prec=3)+' \pm '+astr(np.round(np.std(r2adj_calib),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
                 rotation=0, fontsize=textsize-2,
                 horizontalalignment='right', verticalalignment='bottom')
    sub.text(0.96, 0.02, str2tex('$R^2_{adj,val} = '+astr(np.round(np.mean(r2adj_valid),3),prec=3)+' \pm '+astr(np.round(np.std(r2adj_valid),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
                 rotation=0, fontsize=textsize-2,
                 horizontalalignment='right', verticalalignment='bottom')
    sub.text(0.04, 0.96, str2tex('$MAE_{cal} = '+astr(np.round(np.mean(mae_calib),3),prec=3)+' \pm '+astr(np.round(np.std(mae_calib),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
                 rotation=0, fontsize=textsize-2,
                 horizontalalignment='left', verticalalignment='top')
    sub.text(0.04, 0.88, str2tex('$MAE_{val} = '+astr(np.round(np.mean(mae_valid),3),prec=3)+' \pm '+astr(np.round(np.std(mae_valid),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
                 rotation=0, fontsize=textsize-2,
                 horizontalalignment='left', verticalalignment='top')

    # x-labels only last row
    if (iplot == 8):
        xlabel=str2tex('xSSA-derived $ST_i^w$ [-]',usetex=usetex)
    else:
        xlabel=''

    # y-labels only first column in every second row
    if ((iplot) % ncol == 1 and ((iplot-1) // ncol) % 2 == 1):
        ylabel=str2tex("predicted $ST_i^w$ [-]",usetex=usetex)
    else:
        ylabel=''

    # 1:1 line for reference
    sub.plot([0.0,maxval_xy],[0.0,maxval_xy],linewidth=lwidth,color='k')

    # # x-ticks only in last row
    # if ((iplot-1) // ncol < nrow - dummy_rows - 1):
    #     # sub.axes.get_xaxis().set_visible(False)
    #     sub.axes.get_xaxis().set_ticks([])
    # else:
    #     plt.xticks(rotation=90)

    # # y-ticks only in first column
    # if (iplot-1)%ncol != 0:
    #     # sub.axes.get_yaxis().set_visible(False)
    #     sub.axes.get_yaxis().set_ticks([])

    sub.set_xlim([0.0,maxval_xy])
    sub.set_ylim([0.0,maxval_xy])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # add ABC
    sub.text(1.00,1.0,str2tex(chr(96+iplot),usetex=usetex),
                         verticalalignment='bottom',horizontalalignment='right',
                         fontweight='bold',
                         fontsize=textsize+2,transform=sub.transAxes)

# colorbar
# [left, bottom, width, height]
pos_cbar = [0.2,0.38,0.6,0.01]
print("pos cbar: ",pos_cbar)
csub    = fig.add_axes( pos_cbar )

if inorm == 'log':
    ticks = [ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ]
    cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='horizontal', extend='min')
    cbar.set_ticklabels([ "{:.1e}".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
    cbar.set_label(str2tex("Density [-]",usetex=usetex))
elif inorm == 'pow':
    ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
    ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
    cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, ticks=ticks, cmap=cmap, orientation='horizontal', extend='max')  #
    cbar.set_ticklabels([ "{:.0e}".format(itick) for itick in ticks ])
    cbar.set_label(str2tex("Frequency [-]",usetex=usetex))
else:
    raise ValueError('Norm for colormap not known.')

# color bins
for ibin in range(cmap.N):
    print("Color bin #",ibin+1,"  :: [",((max_pow**pow_lambda)/cmap.N*(ibin))**(1./pow_lambda),',',((max_pow**pow_lambda)/cmap.N*(ibin+1))**(1./pow_lambda),']')


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
