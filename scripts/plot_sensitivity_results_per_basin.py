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
#     run figure_2.py -t pdf -p figure_2 -n 10
#     python figure_2.py -t pdf -p 03MD001 -i ../data_out/03MD001/results_nsets1000.pkl

# files=$( \ls ../data_out/*/results_nsets1000.pkl )
# for ff in $files ; do bb=$( echo $ff | cut -d '/' -f 3 ) ; echo ${bb} ; python figure_2.py -t pdf -p ../data_out/${bb}/${bb} -i ../data_out/${bb}/results_nsets1000.pkl ; pdfcrop ../data_out/${bb}/${bb}.pdf ; mv ../data_out/${bb}/${bb}-crop.pdf ../data_out/${bb}/${bb}.pdf ; echo "---------------" ; echo " " ; done

# scp julemai@graham.computecanada.ca:/home/julemai/projects/rpp-hwheater/julemai/sa-usgs-canopex/data_out/*/*.pdf .

#!/usr/bin/env python
from __future__ import print_function

"""

Plots results of RAVEN sensitivity analysis with multiple process options

History
-------
Written,  JM, Jun 2019
"""

# -------------------------------------------------------------------------
# Command line arguments - if script
#

# Comment|Uncomment - Begin
if __name__ == '__main__':

    import argparse
    import numpy as np

    plotname    = ''
    outtype     = ''
    usetex      = False
    serif       = False
    nsets       = 100            # number of Sobol sequences
    nboot       = 1             # Set to 1 for single run of SI and STI calculation
    variable    = 'Q'           # model output variable
    picklefile  = None          # default "results_nsets<nsets>_snow.pkl"
    
    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Benchmark example to test Sensitivity Analysis for models with multiple process options.''')
    parser.add_argument('-p', '--plotname', action='store',
                        default=plotname, dest='plotname', metavar='plotname',
                        help='Name of plot output file for types pdf, html or d3, '
                        'and name basis for type png (default: '+__file__[0:__file__.rfind(".")]+').')
    parser.add_argument('-s', '--serif', action='store_true', default=serif, dest="serif",
                    help="Use serif font; default sans serif.")
    parser.add_argument('-t', '--type', action='store',
                        default=outtype, dest='outtype', metavar='outtype',
                        help='Output type is pdf, png, html, or d3 (default: open screen windows).')
    parser.add_argument('-u', '--usetex', action='store_true', default=usetex, dest="usetex",
                        help="Use LaTeX to render text in pdf, png and html.")
    parser.add_argument('-b', '--nboot', action='store',
                        default=nboot, dest='nboot', metavar='nboot',
                        help='Number of bootstrap samples (default: nboot=10).')
    parser.add_argument('-n', '--nsets', action='store',
                        default=nsets, dest='nsets', metavar='nsets',
                        help='Number of sensitivity samples (default: nsets=10).')
    parser.add_argument('-v', '--variable', action='store',
                        default=variable, dest='variable', metavar='variable',
                        help='Model output variable name. Must match key in dictionaries in Pickle. (default: variable="Q").')
    parser.add_argument('-i', '--picklefile', action='store',
                        default=picklefile, dest='picklefile', metavar='picklefile',
                        help="Name of Pickle that contains all model runs and Sobol' indexes. (default: results_nsets<nsets>.pkl).")

    args     = parser.parse_args()
    plotname = args.plotname
    outtype  = args.outtype
    serif    = args.serif
    usetex   = args.usetex
    nboot    = np.int(args.nboot)
    nsets    = np.int(args.nsets)
    variable = args.variable
    picklefile = args.picklefile

    if picklefile is None:
        picklefile = "results_nsets"+str(nsets)+".pkl"

    del parser, args
# Comment|Uncomment - End


# -------------------------------------------------------------------------
# Function definition - if function
#

    # Check input
    outtype = outtype.lower()
    outtypes = ['', 'pdf', 'png', 'html', 'd3']
    if outtype not in outtypes:
        print('\nError: output type must be in ', outtypes)
        import sys
        sys.exit()

    import numpy as np
    import sa_model_multiple_processes
    import jams
    import copy
    import sobol
    import time
    import re
    import os
    import datetime
    from   raven_templates import RVI, RVT, RVP, RVH, RVC
    from   raven_common    import writeString, makeDirectories
    from   pathlib2        import Path
    import subprocess
    import shutil
    from   fread           import fread
    
    t1 = time.time()

    if (outtype == 'd3'):
        try:
            import mpld3
        except:
            print("No mpld3 found. Use output type html instead of d3.")
            outtype = 'html'


    # -------------------------------------------------------------------------
    # Setup
    #
    dowhite    = False  # True: black background, False: white background
    title      = False   # True: title on plots, False: no plot titles
    textbox    = False  # if true: additional information is set as text box within plot
    textbox_x  = 0.95
    textbox_y  = 0.85

    # -------------------------------------------------------------------------
    # Setup Calculations
    #
    if dowhite:
        fgcolor = 'white'
        bgcolor = 'black'
    else:
        fgcolor = 'black'
        bgcolor = 'white'

    # colors
    cols1 = jams.color.get_brewer('YlOrRd9', rgb=True)
    cols1 = jams.get_brewer( 'WhiteYellowOrangeRed',rgb=True)[30:]
    cols1 = jams.get_brewer( 'dark_rainbow_256',rgb=True)   # blue to red

    cols2 = jams.color.get_brewer('YlOrRd9', rgb=True)[::-1]
    cols2 = jams.get_brewer( 'WhiteYellowOrangeRed',rgb=True)[30:][::-1]
    cols2 = jams.get_brewer( 'dark_rainbow_256',rgb=True)[::-1]  # red to blue

    cols3 = [cols2[0],cols2[95],cols2[-1]]  # red, yellow, blue
    cols3 = [jams.color.colours('gray'),cols2[0],jams.color.colours('white')]  # gray red white

    # -------------------------------------------------------------------------
    # Set processes and options
    # -------------------------------------------------------------------------

    # list of parameters that go into each option (numbering starts with 0)
    options_paras_raven = [
        [[0,28],[1,28],[2,28]],                           # parameters of infiltration                 options     
        [[3,28],[4,5,28],[4,5,6,28]],                     # parameters of quickflow                    options
        [[7,28],[7,8,9,28]],                              # parameters of evaporation                  options
        [[10],[10,11]],                                   # parameters of baseflow                     options
        [[12,13,14,15,16,17],[],[17,18]],                 # parameters of snow balance                 options     # HMETS, SIMPLE_MELT, HBV
        [[19,20]],                                        # parameters for convolution (surf. runoff)  option
        [[21,22]],                                        # parameters for convolution (delay. runoff) option
        [[23,24,25,26]],                                  # parameters for potential melt              option
        [[27,28,29]],                                     # parameters for percolation                 option
        #[[29,30]],                                       # parameters for soil model                  option
        ]

    # -------------------------------------------------------------------------
    # Read results
    # -------------------------------------------------------------------------
    import pickle
    setup         = pickle.load( open( picklefile, "rb" ) )
    sobol_indexes = setup['sobol_indexes']

    # -------------------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------------------
    infil_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[20]
    quick_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[55]
    evapo_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[80]
    basef_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[105]
    snowb_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[130]
    convs_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[155]
    convd_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[180]
    potme_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[205]
    perco_color = jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[230]
    soilm_color = (0.7,0.7,0.7) #jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[255]
    
    # -------------------------------------------------------------------------
    # Plotting of results
    # -------------------------------------------------------------------------
    # Main plot
    ncol        = 2           # number columns
    nrow        = 4           # number of rows
    textsize    = 10          # standard text size
    dxabc       = 0.03          # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
    dyabc       = 0.92          # % of (max-min) shift up from lower x-axis for a,b,c,... labels
    dxsig       = 1.23        # % of (max-min) shift to the right from left y-axis for signature
    dysig       = -0.075      # % of (max-min) shift up from lower x-axis for signature
    dxtit       = 0           # % of (max-min) shift to the right from left y-axis for title
    dytit       = 1.2         # % of (max-min) shift up from lower x-axis for title
    hspace      = 0.16        # x-space between subplots
    vspace      = 0.06        # y-space between subplots

    lwidth      = 0.5         # linewidth
    elwidth     = 0.5         # errorbar line width
    alwidth     = 1.0         # axis line width
    glwidth     = 0.5         # grid line width
    msize       = 8.0         # marker size
    mwidth      = 0.0         # marker edge width
    mcol1       = '0.7'       # primary marker colour
    mcol2       = '0.0'       # secondary
    mcol3       = '0.0'       # third
    mcols       = jams.color.colours(['blue','green','yellow','orange','red','darkgray','darkblue','black','darkgreen','gray'])
    lcol0       = jams.color.colours('black')    # line colour
    lcol1       = jams.color.colours('blue')     # line colour
    lcol2       = jams.color.colours('green')    # line colour
    lcol3       = jams.color.colours('yellow')   # line colour
    lcols       = jams.color.colours(['black','blue','green','yellow'])
    markers     = ['o','v','s','^']

    # Legend
    llxbbox     = 0.98        # x-anchor legend bounding box
    llybbox     = 0.98        # y-anchor legend bounding box
    llrspace    = 0.          # spacing between rows in legend
    llcspace    = 1.0         # spacing between columns in legend
    llhtextpad  = 0.4         # the pad between the legend handle and text
    llhlength   = 1.5         # the length of the legend handles
    frameon     = False       # if True, draw a frame around the legend. If None, use rc
      
    import matplotlib as mpl
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    mpl.use('TkAgg')
    
    if (outtype == 'pdf'):
        mpl.use('PDF') # set directly after import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        # Customize: http://matplotlib.sourceforge.net/users/customizing.html
        mpl.rc('ps', papersize='a4', usedistiller='xpdf') # ps2pdf
        # mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
        if usetex:
            mpl.rc('text', usetex=True)
            if not serif:
                #   r'\usepackage{helvet}',                             # use Helvetica
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage[math,lf,mathtabular,footnotefigures]{MyriadPro}', # use MyriadPro font
                    r'\renewcommand{\familydefault}{\sfdefault}',       # normal text font is sans serif
                    r'\figureversion{lining,tabular}',
                    r'\usepackage{wasysym}',                            # for permil symbol (load after MyriadPro)
                    ]
            else:
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage{wasysym}'                     # for permil symbol
                    ]
        else:
            if serif:
                mpl.rcParams['font.family']     = 'serif'
                mpl.rcParams['font.sans-serif'] = 'Times'
            else:
                mpl.rcParams['font.family']     = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = 'Arial'       # Arial, Verdana
    elif (outtype == 'png') or (outtype == 'html') or (outtype == 'd3'):
        mpl.use('Agg') # set directly after import matplotlib
        import matplotlib.pyplot as plt
        # mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
        if usetex:
            mpl.rc('text', usetex=True)
            if not serif:
                #   r'\usepackage{helvet}',                             # use Helvetica
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage[math,lf,mathtabular,footnotefigures]{MyriadPro}', # use MyriadPro font
                    r'\renewcommand{\familydefault}{\sfdefault}',       # normal text font is sans serif
                    r'\figureversion{lining,tabular}',
                    r'\usepackage{wasysym}',                            # for permil symbol (load after MyriadPro)
                    ]
            else:
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage{wasysym}'                     # for permil symbol
                    ]
        else:
            if serif:
                mpl.rcParams['font.family']     = 'serif'
                mpl.rcParams['font.sans-serif'] = 'Times'
            else:
                mpl.rcParams['font.family']     = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = 'Arial'       # Arial, Verdana
        mpl.rc('savefig', dpi=dpi, format='png')
    else:
        import matplotlib.pyplot as plt
        # mpl.rc('figure', figsize=(4./5.*8.27,4./5.*11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
    mpl.rc('text.latex', unicode=True)
    mpl.rc('font', size=textsize)
    mpl.rc('path', simplify=False) # do not remove
    # print(mpl.rcParams)
    mpl.rc('axes', linewidth=alwidth, edgecolor=fgcolor, facecolor=bgcolor, labelcolor=fgcolor)
    mpl.rc('figure', edgecolor=bgcolor, facecolor='grey')
    mpl.rc('grid', color=fgcolor)
    mpl.rc('lines', linewidth=lwidth, color=fgcolor)
    mpl.rc('patch', edgecolor=fgcolor)
    mpl.rc('savefig', edgecolor=bgcolor, facecolor=bgcolor)
    mpl.rc('patch', edgecolor=fgcolor)
    mpl.rc('text', color=fgcolor)
    mpl.rc('xtick', color=fgcolor)
    mpl.rc('ytick', color=fgcolor)

    if (outtype == 'pdf'):
        pdffile = plotname+'.pdf'
        print('Plot PDF ', pdffile)
        pdf_pages = PdfPages(pdffile)
    elif (outtype == 'png'):
        print('Plot PNG ', plotname)
    else:
        print('Plot X')

    t1  = time.time()
    ifig = 0

    figsize = mpl.rcParams['figure.figsize']
    mpl.rcParams['axes.linewidth'] = lwidth

    
    ifig = 0

    # -------------------------------------------------
    # arithmetic mean Sobol' indexes
    # -------------------------------------------------
    
    ifig += 1
    iplot = 0
    print('Plot - Fig ', ifig)
    fig = plt.figure(ifig)

    # -----------------------
    # plot
    # -----------------------
    ylim = [-0.1, 0.8]

    # -------------
    # Parameter sensitivities
    # -------------
    iplot += 1
    sub = fig.add_axes(jams.position(nrow, 1, iplot, hspace=hspace/2, vspace=vspace))

    paras = [      '$x_{1}$',  '$x_{2}$',  '$x_{3}$',  '$x_{4}$',  '$x_{5}$',  '$x_{6}$',  '$x_{7}$',  '$x_{8}$',  '$x_{9}$',  '$x_{10}$',
                   '$x_{11}$', '$x_{12}$', '$x_{13}$', '$x_{14}$', '$x_{15}$', '$x_{16}$', '$x_{17}$', '$x_{18}$', '$x_{19}$', '$x_{20}$',
                   '$x_{21}$', '$x_{22}$', '$x_{23}$', '$x_{24}$', '$x_{25}$', '$x_{26}$', '$x_{27}$', '$x_{28}$', '$x_{29}$', '$x_{30}$',
                   '$r_{1}$', '$r_{2}$', '$r_{3}$', '$r_{4}$', '$r_{5}$', '$r_{6}$', '$r_{7}$', '$r_{8}$']
    paras = [ jams.str2tex(ii,usetex=usetex) for ii in paras ]

    keys = sobol_indexes['paras']['msi'].keys()
    ikey = variable
    if not( variable in keys):
        print("")
        print("Variables in pickle: ",keys)
        print("Variable given:      ",variable)
        raise ValueError("Variable given is not available in Pickle!")
    
    npara = np.shape(sobol_indexes['paras']['msi'][ikey])[0]
    mark1 = sub.bar(np.arange(npara), sobol_indexes['paras']['msti'][ikey], align='center', alpha=0.3,
                    color=(infil_color, infil_color, infil_color,
                           quick_color, quick_color, quick_color, quick_color,
                           evapo_color, evapo_color, evapo_color,
                           basef_color, basef_color,
                           snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, 
                           convs_color,convs_color,
                           convd_color,convd_color,
                           potme_color,potme_color,potme_color,potme_color,
                           perco_color,
                           soilm_color,soilm_color,
                           # weights generating random numbers
                           infil_color, infil_color,
                           quick_color, quick_color,
                           evapo_color,
                           basef_color,
                           snowb_color, snowb_color, snowb_color) )    # STi wmean
    mark2 = sub.bar(np.arange(npara), sobol_indexes['paras']['msi'][ikey], align='center', alpha=1.0,
                    color=(infil_color, infil_color, infil_color,
                           quick_color, quick_color, quick_color, quick_color,
                           evapo_color, evapo_color, evapo_color,
                           basef_color, basef_color,
                           snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, 
                           convs_color,convs_color,
                           convd_color,convd_color,
                           potme_color,potme_color,potme_color,potme_color,
                           perco_color,
                           soilm_color,soilm_color,
                           # weights generating random numbers
                           infil_color, infil_color,
                           quick_color, quick_color,
                           evapo_color,
                           basef_color,
                           snowb_color, snowb_color, snowb_color) )    # Si  wmean

    sub.set_ylim(ylim)

    npara = len(paras)
    plt.xticks(np.arange(npara), paras,rotation=90,fontsize='x-small')
    
    plt.title(jams.str2tex('Sensitivities of Model Variables',usetex=usetex))
    plt.ylabel(jams.str2tex("Sobol' Index",usetex=usetex))

    jams.abc2plot(sub,dxabc/2,dyabc,iplot,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='top')

    # -------------
    # Process option sensitivities
    # -------------
    iplot += 2
    sub = fig.add_axes(jams.position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    procopt = ['INF_HMETS', 'INF_VIC_ARNO', 'INF_HBV', 'BASE_LINEAR_ANALYTIC', 'BASE_VIC', 'BASE_TOPMODEL',
               'SOILEVAP_ALL', 'SOILEVAP_TOPMODEL', 'BASE_LINEAR_ANALYTIC', 'BASE_POWER_LAW',
               'SNOBAL_HMETS', 'SNOBAL_SIMPLE_MELT', 'SNOBAL_HBV', 'CONVOL_GAMMA', 'CONVOL_GAMMA_2', 'POTMELT_HMETS', 'PERC_LINEAR',
               'Infiltration weight gen. $r_{1}$', 'Infiltration weight gen. $r_{2}$', 'Quickflow weight gen. $r_{3}$', 'Quickflow weight gen. $r_{4}$',
               'Evaporation weight gen. $r_{5}$', 'Baseflow weight gen. $r_{6}$', 'Snow Balance weight gen. $r_{7}$', 'Snow Balance weight gen. $r_{8}$']  
    procopt = [ jams.str2tex(ii,usetex=usetex) for ii in procopt ]

    keys = sobol_indexes['process_options']['msi'].keys()
    ikey = variable
    if not( variable in keys):
        print("")
        print("Variables in pickle: ",keys)
        print("Variable given:      ",variable)
        raise ValueError("Variable given is not available in Pickle!")
    
    nopt = np.shape(sobol_indexes['process_options']['msi'][ikey])[0] 
    mark1 = sub.bar(np.arange(nopt), sobol_indexes['process_options']['msti'][ikey], align='center', alpha=0.6,
                    color=(infil_color, infil_color, infil_color,
                           quick_color, quick_color, quick_color,
                           evapo_color, evapo_color,
                           basef_color, basef_color,
                           snowb_color, snowb_color, snowb_color, 
                           convs_color, convd_color, potme_color, perco_color, #soilm_color,
                           infil_color, infil_color,
                           quick_color, quick_color,
                           evapo_color,
                           basef_color,
                           snowb_color, snowb_color, snowb_color))    # STi wmean
    mark2 = sub.bar(np.arange(nopt), sobol_indexes['process_options']['msi'][ikey], align='center', alpha=1.0,
                    color=(infil_color, infil_color, infil_color,
                           quick_color, quick_color, quick_color,
                           evapo_color, evapo_color,
                           basef_color, basef_color,
                           snowb_color, snowb_color, snowb_color, 
                           convs_color, convd_color, potme_color, perco_color, #soilm_color,
                           infil_color, infil_color,
                           quick_color, quick_color,
                           evapo_color,
                           basef_color,
                           snowb_color, snowb_color, snowb_color))        # Si  wmean  

    sub.set_ylim(ylim)

    nopt = len(procopt)
    plt.xticks(np.arange(nopt), procopt,rotation=90,fontsize='x-small')
    
    plt.title(jams.str2tex('Sensitivities of Process Options',usetex=usetex))
    plt.ylabel(jams.str2tex("Sobol' Index",usetex=usetex))

    jams.abc2plot(sub,dxabc,dyabc,iplot-1,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='top')

    # -------------
    # Process sensitivities
    # -------------
    iplot += 1
    sub = fig.add_axes(jams.position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    processes = ['Infiltration','Quickflow','Evaporation','Baseflow','Snow Balance', 'Convolution\n(surface runoff)', 'Convolution\n(delayed runoff)', 'Potential melt', 'Percolation']  # , 'Soil model'
    processes = [ jams.str2tex(ii,usetex=usetex) for ii in processes ]

    keys = sobol_indexes['processes']['msi'].keys()
    ikey = variable
    if not( variable in keys):
        print("")
        print("Variables in pickle: ",keys)
        print("Variable given:      ",variable)
        raise ValueError("Variable given is not available in Pickle!")
    
    nproc = np.shape(sobol_indexes['processes']['msi'][ikey])[0]
    mark1 = sub.bar(    np.arange(nproc), sobol_indexes['processes']['msti'][ikey], align='center', alpha=0.6,
                        color=(infil_color, quick_color, evapo_color, basef_color, snowb_color,
                               convs_color, convd_color, potme_color, perco_color, soilm_color))    # STi wmean
    mark2 = sub.bar(    np.arange(nproc), sobol_indexes['processes']['msi'][ikey], align='center', alpha=1.0,
                        color=(infil_color, quick_color, evapo_color, basef_color, snowb_color,
                               convs_color, convd_color, potme_color, perco_color, soilm_color))    # Si  wmean

    sub.set_ylim(ylim)
    
    plt.xticks(np.arange(nproc), processes,rotation=90,fontsize='x-small')
    plt.title(jams.str2tex('Sensitivities of Processes',usetex=usetex))
    #plt.ylabel(jams.str2tex("Sobol' Index",usetex=usetex))

    jams.abc2plot(sub,dxabc,dyabc,iplot-1,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='top')
    
    # Create custom artists
    #      (left, bottom), width, height
    boxSi_1   = patches.Rectangle( (0.00, -0.70), 0.03, 0.05, color = infil_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_2   = patches.Rectangle( (0.04, -0.70), 0.03, 0.05, color = quick_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_3   = patches.Rectangle( (0.08, -0.70), 0.03, 0.05, color = evapo_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_4   = patches.Rectangle( (0.12, -0.70), 0.03, 0.05, color = basef_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_5   = patches.Rectangle( (0.16, -0.70), 0.03, 0.05, color = snowb_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_6   = patches.Rectangle( (0.20, -0.70), 0.03, 0.05, color = convs_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_7   = patches.Rectangle( (0.24, -0.70), 0.03, 0.05, color = convd_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_8   = patches.Rectangle( (0.28, -0.70), 0.03, 0.05, color = potme_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_9   = patches.Rectangle( (0.32, -0.70), 0.03, 0.05, color = perco_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_10  = patches.Rectangle( (0.36, -0.70), 0.03, 0.05, color = soilm_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_1  = patches.Rectangle( (0.00, -0.83), 0.03, 0.05, color = infil_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_2  = patches.Rectangle( (0.04, -0.83), 0.03, 0.05, color = quick_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_3  = patches.Rectangle( (0.08, -0.83), 0.03, 0.05, color = evapo_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_4  = patches.Rectangle( (0.12, -0.83), 0.03, 0.05, color = basef_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_5  = patches.Rectangle( (0.16, -0.83), 0.03, 0.05, color = snowb_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_6  = patches.Rectangle( (0.20, -0.83), 0.03, 0.05, color = convs_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_7  = patches.Rectangle( (0.24, -0.83), 0.03, 0.05, color = convd_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_8  = patches.Rectangle( (0.28, -0.83), 0.03, 0.05, color = potme_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_9  = patches.Rectangle( (0.32, -0.83), 0.03, 0.05, color = perco_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_10 = patches.Rectangle( (0.36, -0.83), 0.03, 0.05, color = soilm_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    sub.add_patch(boxSi_1)  ;  sub.add_patch(boxSTi_1)
    sub.add_patch(boxSi_2)  ;  sub.add_patch(boxSTi_2)
    sub.add_patch(boxSi_3)  ;  sub.add_patch(boxSTi_3)
    sub.add_patch(boxSi_4)  ;  sub.add_patch(boxSTi_4)
    sub.add_patch(boxSi_5)  ;  sub.add_patch(boxSTi_5)
    sub.add_patch(boxSi_6)  ;  sub.add_patch(boxSTi_6)
    sub.add_patch(boxSi_7)  ;  sub.add_patch(boxSTi_7)
    sub.add_patch(boxSi_8)  ;  sub.add_patch(boxSTi_8)
    sub.add_patch(boxSi_9)  ;  sub.add_patch(boxSTi_9)
    sub.add_patch(boxSi_10) ;  sub.add_patch(boxSTi_10)
    
    sub.text(0.42, -0.67, jams.str2tex("Sobol' main effect $\overline{S_i}$",usetex=usetex), horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.42, -0.80, jams.str2tex("Sobol' total effect $\overline{ST_i}$",usetex=usetex), horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)


    if (outtype == 'pdf'):
        pdf_pages.savefig(fig)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------------------------------
    # variance-weighted mean Sobol' indexes
    # -------------------------------------------------
    
    ifig += 1
    iplot = 0
    print('Plot - Fig ', ifig)
    fig = plt.figure(ifig)

    # -----------------------
    # plot
    # -----------------------
    ylim = [-0.1, 0.8]

    # -------------
    # Parameter sensitivities
    # -------------
    iplot += 1
    sub = fig.add_axes(jams.position(nrow, 1, iplot, hspace=hspace/2, vspace=vspace))

    paras = [      '$x_{1}$',  '$x_{2}$',  '$x_{3}$',  '$x_{4}$',  '$x_{5}$',  '$x_{6}$',  '$x_{7}$',  '$x_{8}$',  '$x_{9}$',  '$x_{10}$',
                   '$x_{11}$', '$x_{12}$', '$x_{13}$', '$x_{14}$', '$x_{15}$', '$x_{16}$', '$x_{17}$', '$x_{18}$', '$x_{19}$', '$x_{20}$',
                   '$x_{21}$', '$x_{22}$', '$x_{23}$', '$x_{24}$', '$x_{25}$', '$x_{26}$', '$x_{27}$', '$x_{28}$', '$x_{29}$', '$x_{30}$',
                   '$r_{1}$', '$r_{2}$', '$r_{3}$', '$r_{4}$', '$r_{5}$', '$r_{6}$', '$r_{7}$', '$r_{8}$']
    paras = [ jams.str2tex(ii,usetex=usetex) for ii in paras ]

    keys = sobol_indexes['paras']['wsi'].keys()
    ikey = variable
    if not( variable in keys):
        print("")
        print("Variables in pickle: ",keys)
        print("Variable given:      ",variable)
        raise ValueError("Variable given is not available in Pickle!")
    
    npara = np.shape(sobol_indexes['paras']['wsi'][ikey])[0]
    mark1 = sub.bar(np.arange(npara), sobol_indexes['paras']['wsti'][ikey], align='center', alpha=0.3,
                    color=(infil_color, infil_color, infil_color,
                           quick_color, quick_color, quick_color, quick_color,
                           evapo_color, evapo_color, evapo_color,
                           basef_color, basef_color,
                           snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, snowb_color,
                           convs_color,convs_color,
                           convd_color,convd_color,
                           potme_color,potme_color,potme_color,potme_color,
                           perco_color,
                           soilm_color,soilm_color,
                           # weights generating random numbers
                           infil_color, infil_color,
                           quick_color, quick_color,
                           evapo_color,
                           basef_color,
                           snowb_color, snowb_color, snowb_color) )    # STi wmean
    mark2 = sub.bar(np.arange(npara), sobol_indexes['paras']['wsi'][ikey], align='center', alpha=1.0,
                    color=(infil_color, infil_color, infil_color,
                           quick_color, quick_color, quick_color, quick_color,
                           evapo_color, evapo_color, evapo_color,
                           basef_color, basef_color,
                           snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, snowb_color, snowb_color,
                           convs_color,convs_color,
                           convd_color,convd_color,
                           potme_color,potme_color,potme_color,potme_color,
                           perco_color,
                           soilm_color,soilm_color,
                           # weights generating random numbers
                           infil_color, infil_color,
                           quick_color, quick_color,
                           evapo_color,
                           basef_color,
                           snowb_color, snowb_color, snowb_color) )    # Si  wmean

    sub.set_ylim(ylim)

    npara = len(paras)
    plt.xticks(np.arange(npara), paras,rotation=90,fontsize='x-small')
    
    plt.title(jams.str2tex('Sensitivities of Model Variables',usetex=usetex))
    plt.ylabel(jams.str2tex("Sobol' Index",usetex=usetex))

    jams.abc2plot(sub,dxabc/2,dyabc,iplot,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='top')

    # -------------
    # Process option sensitivities
    # -------------
    iplot += 2
    sub = fig.add_axes(jams.position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    procopt = ['INF_HMETS', 'INF_VIC_ARNO', 'INF_HBV', 'BASE_LINEAR_ANALYTIC', 'BASE_VIC', 'BASE_TOPMODEL',
               'SOILEVAP_ALL', 'SOILEVAP_TOPMODEL', 'BASE_LINEAR_ANALYTIC', 'BASE_POWER_LAW',
               'SNOBAL_HMETS', 'SNOBAL_SIMPLE_MELT', 'SNOBAL_HBV', 'CONVOL_GAMMA', 'CONVOL_GAMMA_2', 'POTMELT_HMETS', 'PERC_LINEAR',
               'Infiltration weight gen. $r_{1}$', 'Infiltration weight gen. $r_{2}$', 'Quickflow weight gen. $r_{3}$', 'Quickflow weight gen. $r_{4}$',
               'Evaporation weight gen. $r_{5}$', 'Baseflow weight gen. $r_{6}$', 'Snow Balance weight gen. $r_{7}$', 'Snow Balance weight gen. $r_{8}$']  
    procopt = [ jams.str2tex(ii,usetex=usetex) for ii in procopt ]

    keys = sobol_indexes['process_options']['wsi'].keys()
    ikey = variable
    if not( variable in keys):
        print("")
        print("Variables in pickle: ",keys)
        print("Variable given:      ",variable)
        raise ValueError("Variable given is not available in Pickle!")
    
    nopt = np.shape(sobol_indexes['process_options']['wsi'][ikey])[0]
    mark1 = sub.bar(np.arange(nopt), sobol_indexes['process_options']['wsti'][ikey], align='center', alpha=0.6,
                    color=(infil_color, infil_color, infil_color,
                           quick_color, quick_color, quick_color,
                           evapo_color, evapo_color,
                           basef_color, basef_color,
                           snowb_color, snowb_color, snowb_color, 
                           convs_color, convd_color, potme_color, perco_color,
                           infil_color, infil_color,
                           quick_color, quick_color,
                           evapo_color,
                           basef_color,
                           snowb_color, snowb_color, snowb_color))    # STi wmean
    mark2 = sub.bar(np.arange(nopt), sobol_indexes['process_options']['wsi'][ikey], align='center', alpha=1.0,
                    color=(infil_color, infil_color, infil_color,
                           quick_color, quick_color, quick_color,
                           evapo_color, evapo_color,
                           basef_color, basef_color,
                           snowb_color, snowb_color, snowb_color, 
                           convs_color, convd_color, potme_color, perco_color,
                           infil_color, infil_color,
                           quick_color, quick_color,
                           evapo_color,
                           basef_color,
                           snowb_color, snowb_color, snowb_color))        # Si  wmean

    sub.set_ylim(ylim)

    nopt = len(procopt)
    plt.xticks(np.arange(nopt), procopt,rotation=90,fontsize='x-small')
    
    plt.title(jams.str2tex('Sensitivities of Process Options',usetex=usetex))
    plt.ylabel(jams.str2tex("Sobol' Index",usetex=usetex))

    jams.abc2plot(sub,dxabc,dyabc,iplot-1,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='top')

    # -------------
    # Process sensitivities
    # -------------
    iplot += 1
    sub = fig.add_axes(jams.position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    processes = ['Infiltration','Quickflow','Evaporation','Baseflow','Snow Balance', 'Convolution\n(Surface Runoff)', 'Convolution\n(Delayed Runoff)', 'Potential Melt', 'Percolation'] #, 'Soil Model']
    processes = [ jams.str2tex(ii,usetex=usetex) for ii in processes ]

    keys = sobol_indexes['processes']['wsi'].keys()
    ikey = variable
    if not( variable in keys):
        print("")
        print("Variables in pickle: ",keys)
        print("Variable given:      ",variable)
        raise ValueError("Variable given is not available in Pickle!")
    
    nproc = np.shape(sobol_indexes['processes']['wsi'][ikey])[0]    
    mark1 = sub.bar(    np.arange(nproc), sobol_indexes['processes']['wsti'][ikey], align='center', alpha=0.6,
                        color=(infil_color, quick_color, evapo_color, basef_color, snowb_color,
                               convs_color, convd_color, potme_color, perco_color)) #, soilm_color))    # STi wmean
    mark2 = sub.bar(    np.arange(nproc), sobol_indexes['processes']['wsi'][ikey], align='center', alpha=1.0,
                        color=(infil_color, quick_color, evapo_color, basef_color, snowb_color,
                               convs_color, convd_color, potme_color, perco_color)) #, soilm_color))    # Si  wmean

    sub.set_ylim(ylim)
    
    plt.xticks(np.arange(nproc), processes,rotation=90,fontsize='x-small')
    plt.title(jams.str2tex('Sensitivities of Processes',usetex=usetex))
    #plt.ylabel(jams.str2tex("Sobol' Index",usetex=usetex))

    jams.abc2plot(sub,dxabc,dyabc,iplot-1,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='top')
    
    # Create custom artists
    #      (left, bottom), width, height
    boxSi_1   = patches.Rectangle( (0.00, -0.70), 0.03, 0.05, color = infil_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_2   = patches.Rectangle( (0.04, -0.70), 0.03, 0.05, color = quick_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_3   = patches.Rectangle( (0.08, -0.70), 0.03, 0.05, color = evapo_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_4   = patches.Rectangle( (0.12, -0.70), 0.03, 0.05, color = basef_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_5   = patches.Rectangle( (0.16, -0.70), 0.03, 0.05, color = snowb_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_6   = patches.Rectangle( (0.20, -0.70), 0.03, 0.05, color = convs_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_7   = patches.Rectangle( (0.24, -0.70), 0.03, 0.05, color = convd_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_8   = patches.Rectangle( (0.28, -0.70), 0.03, 0.05, color = potme_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_9   = patches.Rectangle( (0.32, -0.70), 0.03, 0.05, color = perco_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSi_10  = patches.Rectangle( (0.36, -0.70), 0.03, 0.05, color = soilm_color, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_1  = patches.Rectangle( (0.00, -0.83), 0.03, 0.05, color = infil_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_2  = patches.Rectangle( (0.04, -0.83), 0.03, 0.05, color = quick_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_3  = patches.Rectangle( (0.08, -0.83), 0.03, 0.05, color = evapo_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_4  = patches.Rectangle( (0.12, -0.83), 0.03, 0.05, color = basef_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_5  = patches.Rectangle( (0.16, -0.83), 0.03, 0.05, color = snowb_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_6  = patches.Rectangle( (0.20, -0.83), 0.03, 0.05, color = convs_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_7  = patches.Rectangle( (0.24, -0.83), 0.03, 0.05, color = convd_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_8  = patches.Rectangle( (0.28, -0.83), 0.03, 0.05, color = potme_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_9  = patches.Rectangle( (0.32, -0.83), 0.03, 0.05, color = perco_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_10 = patches.Rectangle( (0.36, -0.83), 0.03, 0.05, color = soilm_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    sub.add_patch(boxSi_1)  ;  sub.add_patch(boxSTi_1)
    sub.add_patch(boxSi_2)  ;  sub.add_patch(boxSTi_2)
    sub.add_patch(boxSi_3)  ;  sub.add_patch(boxSTi_3)
    sub.add_patch(boxSi_4)  ;  sub.add_patch(boxSTi_4)
    sub.add_patch(boxSi_5)  ;  sub.add_patch(boxSTi_5)
    sub.add_patch(boxSi_6)  ;  sub.add_patch(boxSTi_6)
    sub.add_patch(boxSi_7)  ;  sub.add_patch(boxSTi_7)
    sub.add_patch(boxSi_8)  ;  sub.add_patch(boxSTi_8)
    sub.add_patch(boxSi_9)  ;  sub.add_patch(boxSTi_9)
    sub.add_patch(boxSi_10) ;  sub.add_patch(boxSTi_10)
    
    sub.text(0.42, -0.67, jams.str2tex("Sobol' main effect $\overline{S_i^w}$",usetex=usetex), horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.42, -0.80, jams.str2tex("Sobol' total effect $\overline{ST_i^w}$",usetex=usetex), horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)


    if (outtype == 'pdf'):
        pdf_pages.savefig(fig)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)



    # -------------------------------------------------
    # Sobol' indexes of all processes over time (stacked)
    # -------------------------------------------------
    
    ifig += 1
    iplot = 0
    print('Plot - Fig ', ifig)
    fig = plt.figure(ifig)

    # -----------------------
    # plot
    # -----------------------
    ylim = [0.0, 1.0]

    # -------------
    # Parameter sensitivities
    # -------------
    iplot += 1
    sub = fig.add_axes(jams.position(nrow, 1, iplot, hspace=hspace/2, vspace=vspace)-[0.02,0,0,0])

    keys = sobol_indexes['processes']['sti'].keys()
    ikey = variable
    if not( variable in keys):
        print("")
        print("Variables in pickle: ",keys)
        print("Variable given:      ",variable)
        raise ValueError("Variable given is not available in Pickle!")

    ntime = np.shape(sobol_indexes['processes']['sti'][ikey])[0]
    nproc = np.shape(sobol_indexes['processes']['sti'][ikey])[1]
    ntime_doy = 365

    # determine leap days and discrad them
    start_date = datetime.datetime(1991,1,1,0,0)
    times = np.array([start_date+datetime.timedelta(ii) for ii in range(ntime)])
    leap = np.array([ True if (times[ii].month == 2 and times[ii].day == 29) else False for ii in range(ntime) ])
    # leap[-1] = True  # by accident last day is 2011-01-01 instaed of 2010-12-31

    # weights used for STi in 'Mai1999'
    varA    = np.var(setup['f_a'][ikey][~leap,:],axis=1)
    denomA  = 1./np.sum(varA)
    weights = varA*denomA # (ntime, nproc)

    # reshape such that (ntime,nproc) --> (nyears, 365, nproc)
    tmp_sobol   = copy.deepcopy(sobol_indexes['processes']['sti'][ikey][~leap,:])     # (ntime, nproc)
    tmp_weights = copy.deepcopy(weights)                                    # (ntime)
    sobol_doy   = np.ones([np.int(ntime/ntime_doy),ntime_doy,nproc]) * -9999.0
    weights_doy = np.ones([np.int(ntime/ntime_doy),ntime_doy]) * -9999.0
    for iproc in range(nproc):
        sobol_doy[:,:,iproc]   = np.reshape(tmp_sobol[:,iproc],  [np.int(ntime/ntime_doy),ntime_doy])
        # sobol_doy[:,:,iproc] = np.reshape(tmp_sobol[:,iproc],  [np.int(ntime/ntime_doy),ntime_doy])
        sobol_doy[:,:,iproc]   = np.where(np.isinf(np.reshape(tmp_sobol[:,iproc],  [np.int(ntime/ntime_doy),ntime_doy])),np.nan,np.reshape(tmp_sobol[:,iproc],  [np.int(ntime/ntime_doy),ntime_doy]))
    weights_doy[:,:] = np.reshape(tmp_weights[:],[np.int(ntime/ntime_doy),ntime_doy])

    # average over years
    sobol_doy_mean   = np.nanmean(sobol_doy,axis=0)
    weights_doy_mean = np.nanmean(weights_doy,axis=0)

    # to scale Sobol' indexes such that they sum up to 1.0 (over all processes)
    csum   = np.sum(sobol_doy_mean,axis=1)

    # colors for all processes
    colors = [infil_color, quick_color, evapo_color, basef_color, snowb_color, convs_color, convd_color, potme_color, perco_color, soilm_color]

    
    width = 1.0
    for iproc in range(nproc):
        p1 = sub.bar(np.arange(ntime_doy),
                         sobol_doy_mean[:,iproc]/csum,
                         width,
                         color=colors[iproc],
                         bottom=np.sum(sobol_doy_mean[:,0:iproc],axis=1)/csum)

    sub2 = sub.twinx()
    sub2.plot(np.arange(ntime_doy),weights_doy_mean,color='black',linewidth=lwidth*2)
    sub2.set_ylabel(jams.str2tex('Weight',usetex=usetex), color='black')

    sub.set_xlim([0,ntime_doy])
    sub.set_ylim(ylim)

    #npara = len(paras)
    #plt.xticks(np.arange(npara), paras,rotation=90,fontsize='x-small')

    basin = picklefile.split('/')[-2]
    sub.set_title(jams.str2tex('Basin: '+basin,usetex=usetex))
    sub.set_xlabel(jams.str2tex("Day of Year",usetex=usetex))
    sub.set_ylabel(jams.str2tex("(normalized) Total\n Sobol' Index $ST_i$",usetex=usetex))

    # Create custom artists
    #      (left, bottom), width, height
    boxSTi_1  = patches.Rectangle( (0.00, -0.54), 0.02, 0.05, color = infil_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_2  = patches.Rectangle( (0.00, -0.62), 0.02, 0.05, color = quick_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_3  = patches.Rectangle( (0.00, -0.70), 0.02, 0.05, color = evapo_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_4  = patches.Rectangle( (0.00, -0.78), 0.02, 0.05, color = basef_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_5  = patches.Rectangle( (0.00, -0.86), 0.02, 0.05, color = snowb_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_6  = patches.Rectangle( (0.22, -0.54), 0.02, 0.05, color = convs_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_7  = patches.Rectangle( (0.22, -0.62), 0.02, 0.05, color = convd_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_8  = patches.Rectangle( (0.22, -0.70), 0.02, 0.05, color = potme_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    boxSTi_9  = patches.Rectangle( (0.22, -0.78), 0.02, 0.05, color = perco_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    #boxSTi_10 = patches.Rectangle( (0.22, -0.86), 0.02, 0.05, color = soilm_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
    line      = patches.Rectangle( (0.72, -0.52), 0.02, 0.00, color = 'black',     alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    sub.add_patch(boxSTi_1)
    sub.add_patch(boxSTi_2)
    sub.add_patch(boxSTi_3)
    sub.add_patch(boxSTi_4)
    sub.add_patch(boxSTi_5)
    sub.add_patch(boxSTi_6)
    sub.add_patch(boxSTi_7)
    sub.add_patch(boxSTi_8)
    sub.add_patch(boxSTi_9)
    #sub.add_patch(boxSTi_10)
    sub.add_patch(line)

    sub.text(0.00, -0.40, jams.str2tex("Processes:",usetex=usetex),                   horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.04, -0.53, jams.str2tex("Infiltration",usetex=usetex),                 fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.04, -0.61, jams.str2tex("Quickflow",usetex=usetex),                    fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.04, -0.69, jams.str2tex("Evaporation",usetex=usetex),                  fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.04, -0.77, jams.str2tex("Baseflow",usetex=usetex),                     fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.04, -0.85, jams.str2tex("Snow Balance",usetex=usetex),                 fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.26, -0.53, jams.str2tex("Convolution (Surface Runoff)",usetex=usetex), fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.26, -0.61, jams.str2tex("Convolution (Delayed Runoff)",usetex=usetex), fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.26, -0.69, jams.str2tex("Potential Melt",usetex=usetex),               fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.26, -0.77, jams.str2tex("Percolation",usetex=usetex),                  fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    #sub.text(0.26, -0.85, jams.str2tex("Soil Model",usetex=usetex),                   fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.76, -0.53, jams.str2tex("Weight of Timestep",usetex=usetex),           fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    
    #jams.abc2plot(sub,dxabc/2,dyabc,iplot,bold=True,usetex=usetex,mathrm=True, large=True, parenthesis='none',verticalalignment='top')

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)





    # # -------------------------------------------------
    # # actual model outputs f_a and f_b per model output key
    # # -------------------------------------------------
    
    # ifig += 1
    # iplot = 0
    # print('Plot - Fig ', ifig)
    # fig = plt.figure(ifig)

    # # -----------------------
    # # plot
    # # -----------------------
    # keys = sobol_indexes['paras']['msi'].keys()
    # ikey = variable
    # if not( variable in keys):
    #     print("")
    #     print("Variables in pickle: ",keys)
    #     print("Variable given:      ",variable)
    #     raise ValueError("Variable given is not available in Pickle!")

    # # -------------
    # # model outputs f_a and f_b
    # # -------------
    # for ikey in keys:
        
    #     iplot += 1
    #     sub = fig.add_axes(jams.position(nrow, 1, iplot, hspace=hspace/2, vspace=vspace))

    #     p1 = sub.plot(setup['f_a'][ikey],color='gray',alpha=0.6)
    #     p2 = sub.plot(setup['f_b'][ikey],color='gray',alpha=0.6)
    #     active_snowproc = "+".join(picklefile.split('.')[0].split('_')[3:])
    #     sub.set_title(jams.str2tex("Processes: "+active_snowproc+"  Model output: "+ikey,usetex=usetex))

    # if (outtype == 'pdf'):
    #     pdf_pages.savefig(fig)
    #     plt.close(fig)
    # elif (outtype == 'png'):
    #     pngfile = pngbase+"{0:04d}".format(ifig)+".png"
    #     fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
    #     plt.close(fig)


        
      

    # --------------------------------------
    # Finish
    # --------------------------------------
    if (outtype == 'pdf'):
        pdf_pages.close()
    elif (outtype == 'png'):
        pass
    else:
        plt.show()

    
    t2  = time.time()
    str = '  Time plot [m]: '+jams.astr((t2-t1)/60.,1) if (t2-t1)>60. else '  Time plot [s]: '+jams.astr(t2-t1,0)
    print(str)


