#!/usr/bin/env python

# Copyright 2019-2021 Juliane Mai - juliane.mai(at)uwaterloo.ca
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
# python environment
#     source ~/projects/rpp-hwheater/julemai/xSSA-North-America/env-3.5/bin/activate
#
# run with:
#     run figure_2.py -p figure_2.pdf


#!/usr/bin/env python
from __future__ import print_function

"""

Plots calibration and validation results on a map

History
-------
Written,  JM, Jan 2021
"""

# -------------------------------------------------------------------------
# Command line arguments - if script
#

# Comment|Uncomment - Begin
if __name__ == '__main__':

    import argparse
    import numpy as np

    pngbase   = ''
    pdffile   = ''
    usetex    = False

    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Plots calibration and validation results on a map.''')
    parser.add_argument('-g', '--pngbase', action='store',
                        default=pngbase, dest='pngbase', metavar='pngbase',
                        help='Name basis for png output files (default: open screen window).')
    parser.add_argument('-p', '--pdffile', action='store',
                        default=pdffile, dest='pdffile', metavar='pdffile',
                        help='Name of pdf output file (default: open screen window).')
    parser.add_argument('-t', '--usetex', action='store_true', default=usetex, dest="usetex",
                    help="Use LaTeX to render text in pdf.")

    args      = parser.parse_args()
    pngbase   = args.pngbase
    pdffile   = args.pdffile
    usetex    = args.usetex
    dobw      = False
    donolabel = True

    if pdffile != '':
        outtype   = 'pdf'
    elif pngbase != '':
        outtype   = 'png'
    else:
        outtype   = 'X'

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

    import numpy as np
    import glob
    # import jams
    # import copy
    import time
    # import re
    # import datetime
    import color                            # in lib/
    from position      import position      # in lib/
    from abc2plot      import abc2plot      # in lib/
    from brewer        import get_brewer    # in lib/
    from autostring    import astr          # in lib/
    from str2tex       import str2tex       # in lib/
    from fread         import fread         # in lib/
    from sread         import sread         # in lib/
    from fsread        import fsread        # in lib/

    t1 = time.time()

    # -------------------------------------------------------------------------
    # Read list of basins that got calibrated
    # -------------------------------------------------------------------------
    basin_ids_cal = np.transpose(sread("../data/basins_calibration.dat",skip=0))[0]
    basin_ids_val = np.transpose(sread("../data/basins_validation.dat",skip=0))[0]

    # -------------------------------------------------------------------------
    # Read basin metadata
    # -------------------------------------------------------------------------

    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac
    head = fread("../data/basin_physical_characteristics.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data/basin_physical_characteristics.txt",skip=1,separator=';',cname=["lat","lon","area_km2","elevation_m","slope_deg","forest_frac"],sname=["basin_id","basin_name"])

    dict_metadata = {}
    for ibasin in range(len(meta_float)):

        if meta_string[ibasin][0] in basin_ids_cal:
            dict_basin = {}
            dict_basin["name"]        = meta_string[ibasin][1]
            dict_basin["lat"]         = meta_float[ibasin][0]
            dict_basin["lon"]         = meta_float[ibasin][1]
            dict_basin["area_km2"]    = meta_float[ibasin][2]
            dict_basin["elevation_m"] = meta_float[ibasin][3]
            dict_basin["slope_deg"]   = meta_float[ibasin][4]
            dict_basin["forest_frac"] = meta_float[ibasin][5]

            dict_metadata[meta_string[ibasin][0]] = dict_basin


    # -------------------------------------------------------------------------
    # Read basin calibration results
    # -------------------------------------------------------------------------

    # basin_id; basin_name; nse; rmse; kge
    head = fread("../data/basin_calibration_results.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data/basin_calibration_results.txt",skip=1,separator=';',cname=["nse","rmse","kge"],sname=["basin_id","basin_name"])

    meta_string = [ [ iitem.strip() for iitem in iline ] for iline in meta_string ]

    dict_calibration = {}
    for ibasin in range(len(meta_float)):
        dict_basin = {}
        dict_basin["name"]          = meta_string[ibasin][1]
        dict_basin["nse"]           = meta_float[ibasin][0]
        dict_basin["rmse"]          = meta_float[ibasin][1]
        dict_basin["kge"]           = meta_float[ibasin][2]

        if meta_string[ibasin][0] in basin_ids_cal:
            dict_calibration[meta_string[ibasin][0]] = dict_basin


    # -------------------------------------------------------------------------
    # Read basin validation results
    # -------------------------------------------------------------------------

    # basin_id; basin_name; nse; rmse; kge
    head = fread("../data/basin_validation_results.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data/basin_validation_results.txt",skip=1,separator=';',cname=["nse","rmse","kge"],sname=["basin_id","basin_name"])

    meta_string = [ [ iitem.strip() for iitem in iline ] for iline in meta_string ]

    dict_validation = {}
    for ibasin in range(len(meta_float)):
        dict_basin = {}
        dict_basin["name"]          = meta_string[ibasin][1]
        dict_basin["nse"]           = meta_float[ibasin][0]
        dict_basin["rmse"]          = meta_float[ibasin][1]
        dict_basin["kge"]           = meta_float[ibasin][2]

        if meta_string[ibasin][0] in basin_ids_val:
            dict_validation[meta_string[ibasin][0]] = dict_basin

    # -------------------------------------------------------------------------
    # sort basins starting with largest
    # -------------------------------------------------------------------------
    areas = []
    for ibasin_id,basin_id in enumerate(basin_ids_cal):
        areas.append(dict_metadata[basin_id]["area_km2"])
    areas = np.array(areas)
    idx_areas = np.argsort(areas)[::-1]
    basin_ids_cal = np.array(basin_ids_cal)[idx_areas]


    # -------------------------------------------------------------------------
    # Setup
    #
    dowhite    = False  # True: black background, False: white background
    title      = False   # True: title on plots, False: no plot titles
    textbox    = False  # if true: additional information is set as text box within plot
    textbox_x  = 0.95
    textbox_y  = 0.85


    # -------------------------------------------------------------------------
    # Plotting of results
    # -------------------------------------------------------------------------

    # Main plot
    nrow        = 5           # # of rows of subplots per figure
    ncol        = 2           # # of columns of subplots per figure
    hspace      = 0.02         # x-space between subplots
    vspace      = 0.05        # y-space between subplots
    right       = 0.9         # right space on page
    textsize    = 6           # standard text size
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
    msize       = 3.0         # marker size
    mwidth      = 1.0         # marker edge width
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

    from matplotlib.patches import Rectangle, Circle, Polygon
    from mpl_toolkits.basemap import Basemap

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
        c  = get_brewer('rdylbu11', rgb=True)
        tmp = c.pop(5)   # rm yellow
        np.random.shuffle(c)

        #c.insert(2,c[2]) # same colour for both soil moistures
        ocean_color = (151/256., 183/256., 224/256.)
        # ocean_color = color.get_brewer('accent5', rgb=True)[-1]

        # rainbow colors
        cc = color.get_brewer('dark_rainbow_256', rgb=True)
        cc = cc[::-1] # reverse colors
        cmap = mpl.colors.ListedColormap(cc)

        # blue-yellow-red colors
        cc = color.get_brewer('rdylbu10', rgb=True)
        cmap = mpl.colors.ListedColormap(cc)

        # green-pink colors
        cc = color.get_brewer('piyg10', rgb=True)
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

    if (outtype == 'pdf'):
        # pdffile = plotname+'.pdf'
        print('Plot PDF ', pdffile)
        pdf_pages = PdfPages(pdffile)
    elif (outtype == 'png'):
        print('Plot PNG ', pngbase)
    else:
        print('Plot X')

    t1  = time.time()
    ifig = 0

    figsize = mpl.rcParams['figure.figsize']
    mpl.rcParams['axes.linewidth'] = lwidth


    ifig = 0

    # -------------------------------------------------------------------------
    # Fig 1
    #
    ifig += 1
    iplot = 0
    print('Plot - Fig ', ifig)

    fig = plt.figure(ifig)

    max_nse = 1.0
    min_nse = 0.0

    # -----------------------------------------
    # Figure 1a: Calibration results
    # -----------------------------------------
    iplot += 1

    #     [left, bottom, width, height]
    pos = [0.1,0.8,0.45,0.15]
    # sub = fig.add_axes(pos, projection=ccrs.LambertConformal())
    sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace),frameon=False) #, axisbg='none')
    # print("position: ",position(nrow,ncol,iplot,hspace=hspace,vspace=vspace))

    # Map: North America [-140, -65, 22, 77]
    llcrnrlon =  -120.
    urcrnrlon =   -30.  #0
    llcrnrlat =   15.
    urcrnrlat =   58.  #68
    lat_1     =   45.   #(llcrnrlat+urcrnrlat)/2.  # first  "equator"
    lat_2     =   55.   #(llcrnrlat+urcrnrlat)/2.  # second "equator"
    lat_0     =   50.   #(llcrnrlat+urcrnrlat)/2.  # center of the map
    lon_0     =   -90.  #(llcrnrlon+urcrnrlon)/2.  # center of the map
    # Map: North America
    llcrnrlon =  -116.
    urcrnrlon =   -30.   # maybe -30 (more space lower right)
    llcrnrlat =   21.
    urcrnrlat =   58.
    lat_1     =   45.   #(llcrnrlat+urcrnrlat)/2.  # first  "equator"
    lat_2     =   55.   #(llcrnrlat+urcrnrlat)/2.  # second "equator"
    lat_0     =   50.   #(llcrnrlat+urcrnrlat)/2.  # center of the map
    lon_0     =   -90.  #(llcrnrlon+urcrnrlon)/2.  # center of the map
    map4 = Basemap(projection='lcc', area_thresh=10000.,ax=sub,suppress_ticks=True,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
                resolution='i') # Lambert conformal

    map4.ax.set_frame_on(False)
    for spine in map4.ax.spines.values():
        spine.set_linewidth(0.0)

    # draw parallels and meridians.
    # labels: [left, right, top, bottom]
    # map4.drawparallels(np.arange(-80.,81.,10.),  labels=[1,0,0,0], linewidth=0.1, color='0.5') #dashes=[1,1],
    # map4.drawmeridians(np.arange(-170.,181.,20.),labels=[0,0,0,1], linewidth=0.1, color='0.5') #dashes=[1,1],

    # draw cooastlines and countries
    # map4.drawcoastlines(linewidth=0.3)
    map4.drawmapboundary(fill_color='white', linewidth=0.2)
    map4.ax.set_frame_on(False)
    map4.drawcountries(color='0.8', linewidth=0.2)
    map4.ax.set_frame_on(False)
    map4.fillcontinents(color='gray', lake_color='white')
    map4.ax.set_frame_on(False)


    # plt.title(str2tex("Model performance",usetex=usetex))

    # add label with current number of basins
    sub.text(1.0,0.02,str2tex("$\mathrm{N}_\mathrm{basins}="+str(len(basin_ids_cal))+"$",usetex=usetex),
                 verticalalignment='bottom',horizontalalignment='right',
                 fontsize=textsize,transform=sub.transAxes)

    # add ABC
    sub.text(1.0,0.96,str2tex("a",usetex=usetex),
                 verticalalignment='top',horizontalalignment='right',
                 fontweight='bold',
                 fontsize=textsize+2,transform=sub.transAxes)

    # adjust frame linewidth
    # sub.outline_patch.set_linewidth(lwidth)

    coord_catch   = []
    for ibasin_id,basin_id in enumerate(basin_ids_cal):

        shapefilename =  "wrong-directory"+basin_id+"/shape.dat"    # "observations/"+basin_id+"/shape.dat"

        # color based on NSE
        icolor = 'red'
        inse = dict_calibration[basin_id]['nse']
        if inse < min_nse:
            icolor = cc[0]
        elif inse > max_nse:
            icolor = cc[-1]
        else:
            icolor = cc[int((inse-min_nse)/(max_nse-min_nse)*(len(cc)-1))]

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
                    sub.add_patch(Polygon(np.transpose(map4(coords[start:end,0],coords[start:end,1])),
                                              facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))
                    #sub.add_patch(Polygon(np.transpose([coords[start:end,0],coords[start:end,1]]),
                    #                          transform=ccrs.PlateCarree(), facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))

                    xxmin = np.nanmin(coords[start:end,0])
                    xxmax = np.nanmax(coords[start:end,0])
                    yymin = np.nanmin(coords[start:end,1])
                    yymax = np.nanmax(coords[start:end,1])

                    if not(donolabel):
                        # annotate
                        #xpt, ypt  = map4(np.mean(coords[start:end,0]), np.mean(coords[start:end,1]))   # center of shape
                        xpt, ypt  = [ np.nanmean(coords[start:end,0]), np.nanmean(coords[start:end,1]) ]      # center of shape
                        x2,  y2   = (1.1,0.95-ibasin_id*0.1)                                            # position of text
                        sub.annotate(basin_id,
                            xy=(xpt, ypt),   xycoords='data',
                            xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                            fontsize=8,
                            verticalalignment='center',horizontalalignment='left',
                            arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                            zorder=400
                            )

            # print("Basin: ",basin_id)
            # print("   --> area      =  ",dict_metadata[basin_id]["area_km2"])
            # print("   --> lon range = [",xxmin,",",xxmax,"]")
            # print("   --> lat range = [",yymin,",",yymax,"]")

        # shapefile doesnt exist only plot dot at location
        else:
            xpt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[0]
            ypt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[1]
            x2,  y2   = (1.1,0.95-ibasin_id*0.1)

            sub.plot(xpt, ypt,
                     linestyle='None', marker='o', markeredgecolor=icolor, markerfacecolor=icolor,
                     markersize=0.7, markeredgewidth=0.0)

            if not(donolabel):
                sub.annotate(basin_id,
                        xy=(xpt, ypt),   xycoords='data',
                        xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                        fontsize=8,
                        # transform=transform,
                        verticalalignment='center',horizontalalignment='left',
                        arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                        zorder=400
                        )
            # print("Basin: ",basin_id)
            # print("   --> lat/lon = [",dict_metadata[basin_id]["lat"],",",dict_metadata[basin_id]["lon"],"]")

    # only way to turn of frame is to plot a white line :(
    sub.plot([0,1,1,0,0],[0,0,1,1,0],transform=sub.transAxes,color='white')

    # -----------------------------------------
    # Figure 1b: Validation results
    # -----------------------------------------
    iplot += 1

    #     [left, bottom, width, height]
    pos = [0.1,0.8,0.45,0.15]
    # sub = fig.add_axes(pos, projection=ccrs.LambertConformal())
    sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)+[-0.15,0.0,0.0,0.0],frameon=False) #, axisbg='none')
    # print("position: ",position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)+[-0.15,0.0,0.0,0.0])

    # Map: North America [-140, -65, 22, 77]
    llcrnrlon =  -120.
    urcrnrlon =   -30.  #0
    llcrnrlat =   15.
    urcrnrlat =   58.  #68
    lat_1     =   45.   #(llcrnrlat+urcrnrlat)/2.  # first  "equator"
    lat_2     =   55.   #(llcrnrlat+urcrnrlat)/2.  # second "equator"
    lat_0     =   50.   #(llcrnrlat+urcrnrlat)/2.  # center of the map
    lon_0     =   -90.  #(llcrnrlon+urcrnrlon)/2.  # center of the map
    # Map: North America
    llcrnrlon =  -116.
    urcrnrlon =   -30.   # maybe -30 (more space lower right)
    llcrnrlat =   21.
    urcrnrlat =   58.
    lat_1     =   45.   #(llcrnrlat+urcrnrlat)/2.  # first  "equator"
    lat_2     =   55.   #(llcrnrlat+urcrnrlat)/2.  # second "equator"
    lat_0     =   50.   #(llcrnrlat+urcrnrlat)/2.  # center of the map
    lon_0     =   -90.  #(llcrnrlon+urcrnrlon)/2.  # center of the map
    map4 = Basemap(projection='lcc', area_thresh=10000.,ax=sub,suppress_ticks=True,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
                resolution='i') # Lambert conformal

    map4.ax.set_frame_on(False)
    for spine in map4.ax.spines.values():
        spine.set_linewidth(0.0)

    # draw parallels and meridians.
    # labels: [left, right, top, bottom]
    # map4.drawparallels(np.arange(-80.,81.,10.),  labels=[0,1,0,0], linewidth=0.1, color='0.5') #dashes=[1,1],
    # map4.drawmeridians(np.arange(-170.,181.,20.),labels=[0,0,0,1], linewidth=0.1, color='0.5') #dashes=[1,1],

    # draw cooastlines and countries
    # map4.drawcoastlines(linewidth=0.3)
    map4.drawmapboundary(fill_color='white', linewidth=0.2)
    map4.ax.set_frame_on(False)
    map4.drawcountries(color='0.8', linewidth=0.2)
    map4.ax.set_frame_on(False)
    map4.fillcontinents(color='gray', lake_color='white')
    map4.ax.set_frame_on(False)


    # plt.title(str2tex("Model performance",usetex=usetex))

    # add label with current number of basins
    sub.text(1.0,0.02,str2tex("$\mathrm{N}_\mathrm{basins}="+str(len(basin_ids_val))+"$",usetex=usetex),
                 verticalalignment='bottom',horizontalalignment='right',
                 fontsize=textsize,transform=sub.transAxes)

    # add ABC
    sub.text(1.0,0.96,str2tex("b",usetex=usetex),
                 verticalalignment='top',horizontalalignment='right',
                 fontweight='bold',
                 fontsize=textsize+2,transform=sub.transAxes)

    coord_catch   = []
    for ibasin_id,basin_id in enumerate(basin_ids_val):

        shapefilename =  "wrong-directory"+basin_id+"/shape.dat"    # "observations/"+basin_id+"/shape.dat"

        # color based on NSE
        icolor = 'red'
        inse = dict_validation[basin_id]['nse']
        if inse < min_nse:
            icolor = cc[0]
        elif inse > max_nse:
            icolor = cc[-1]
        else:
            icolor = cc[int((inse-min_nse)/(max_nse-min_nse)*(len(cc)-1))]

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
                    sub.add_patch(Polygon(np.transpose(map4(coords[start:end,0],coords[start:end,1])),
                                              facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))
                    #sub.add_patch(Polygon(np.transpose([coords[start:end,0],coords[start:end,1]]),
                    #                          transform=ccrs.PlateCarree(), facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))

                    xxmin = np.nanmin(coords[start:end,0])
                    xxmax = np.nanmax(coords[start:end,0])
                    yymin = np.nanmin(coords[start:end,1])
                    yymax = np.nanmax(coords[start:end,1])

                    if not(donolabel):
                        # annotate
                        #xpt, ypt  = map4(np.mean(coords[start:end,0]), np.mean(coords[start:end,1]))   # center of shape
                        xpt, ypt  = [ np.nanmean(coords[start:end,0]), np.nanmean(coords[start:end,1]) ]      # center of shape
                        x2,  y2   = (1.1,0.95-ibasin_id*0.1)                                            # position of text
                        sub.annotate(basin_id,
                            xy=(xpt, ypt),   xycoords='data',
                            xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                            fontsize=8,
                            verticalalignment='center',horizontalalignment='left',
                            arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                            zorder=400
                            )

            # print("Basin: ",basin_id)
            # print("   --> area      =  ",dict_metadata[basin_id]["area_km2"])
            # print("   --> lon range = [",xxmin,",",xxmax,"]")
            # print("   --> lat range = [",yymin,",",yymax,"]")

        # shapefile doesnt exist only plot dot at location
        else:
            xpt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[0]
            ypt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[1]
            x2,  y2   = (1.1,0.95-ibasin_id*0.1)

            sub.plot(xpt, ypt,
                     linestyle='None', marker='o', markeredgecolor=icolor, markerfacecolor=icolor,
                     markersize=0.7, markeredgewidth=0.0)

            if not(donolabel):
                sub.annotate(basin_id,
                        xy=(xpt, ypt),   xycoords='data',
                        xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                        fontsize=8,
                        # transform=transform,
                        verticalalignment='center',horizontalalignment='left',
                        arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                        zorder=400
                        )
            # print("Basin: ",basin_id)
            # print("   --> lat/lon = [",dict_metadata[basin_id]["lat"],",",dict_metadata[basin_id]["lon"],"]")

    # only way to turn of frame is to plot a white line :(
    sub.plot([0,1,1,0,0],[0,0,1,1,0],transform=sub.transAxes,color='white')

    # -------------------------------------------------------------------------
    # Figure 1c: Colorbar calibration
    # -------------------------------------------------------------------------
    iplot += 1
    csub_cal    = fig.add_axes(position(1,1,1,hspace=hspace,vspace=vspace, left=0.21, right=0.418, top=0.776, bottom=0.770) )   # top=0.642, bottom=0.632

    cbar = mpl.colorbar.ColorbarBase(csub_cal, norm=mpl.colors.Normalize(vmin=min_nse, vmax=max_nse), cmap=cmap, orientation='horizontal', extend='min')
    cbar.set_label(str2tex("NSE [-]",usetex=usetex))

    nses= np.array([ dict_calibration[basin_id]['nse'] for basin_id in basin_ids_cal ])
    print('')
    # print('-----------------------------------')
    print('Median NSE calibration: ',np.median(nses))
    # print('Mean   NSE: ',np.mean(nses))
    # print('NSEs      : ',np.sort(nses)[0:10])
    # print('-----------------------------------')
    #print('')
    cticks = [ min_nse+ii*(max_nse-min_nse)/len(cc) for ii in range(len(cc)+1) ]  # cbar.get_ticks()      # tick labels
    percent_in_cat = np.diff([0]+ [ np.sum(nses < ctick) for ctick in cticks ])*100.0/np.shape(nses)[0]

    # add percentages of basins with performance
    for iitick,itick in enumerate(cticks):
        if iitick == 0:
            continue
        dist_from_middle = np.abs(iitick-((len(cticks))/2.))*1.0 / ((len(cticks)-1)/2.)
        icolor = (dist_from_middle,dist_from_middle,dist_from_middle)
        csub_cal.text(1.0/(len(cticks)-1)*(iitick-0.5), 0.5,
                      str2tex(astr(percent_in_cat[iitick],prec=1)+'%',usetex=usetex),
                      color=icolor,
                      fontsize=textsize-3,
                      va='center',
                      ha='center',
                      rotation=0)

    # add small triangle for median
    csub_cal.add_patch(Polygon(np.array([[np.median(nses),0.0],[np.median(nses)-0.015,0.3],[np.median(nses)+0.015,0.3]]),
                               facecolor='black', edgecolor='black', linewidth=0.0,zorder = 800, alpha=0.8))

    # -------------------------------------------------------------------------
    # Figure 1d: Colorbar validation
    # -------------------------------------------------------------------------
    # iplot += 1
    csub_val    = fig.add_axes(position(1,1,1,hspace=hspace,vspace=vspace, left=0.4575, right=0.6655, top=0.776, bottom=0.770) )   # left=0.21, right=0.418

    cbar = mpl.colorbar.ColorbarBase(csub_val, norm=mpl.colors.Normalize(vmin=min_nse, vmax=max_nse), cmap=cmap, orientation='horizontal', extend='min')
    cbar.set_label(str2tex("NSE [-]",usetex=usetex))

    nses= np.array([ dict_validation[basin_id]['nse'] for basin_id in basin_ids_val ])
    # print('')
    # print('-----------------------------------')
    print('Median NSE validation: ',np.median(nses))
    # print('Mean   NSE: ',np.mean(nses))
    # print('NSEs      : ',np.sort(nses)[0:10])
    # print('-----------------------------------')
    print('')
    cticks = [ min_nse+ii*(max_nse-min_nse)/len(cc) for ii in range(len(cc)+1) ]  # cbar.get_ticks()      # tick labels
    percent_in_cat = np.diff([0]+ [ np.sum(nses < ctick) for ctick in cticks ])*100.0/np.shape(nses)[0]

    # add percentages of basins with performance
    for iitick,itick in enumerate(cticks):
        if iitick == 0:
            continue
        dist_from_middle = np.abs(iitick-((len(cticks))/2.))*1.0 / ((len(cticks)-1)/2.)
        icolor = (dist_from_middle,dist_from_middle,dist_from_middle)
        csub_val.text(1.0/(len(cticks)-1)*(iitick-0.5), 0.5,
                      str2tex(astr(percent_in_cat[iitick],prec=1)+'%',usetex=usetex),
                      color=icolor,
                      fontsize=textsize-3,
                      va='center',
                      ha='center',
                      rotation=0)

    # add small triangle for median
    csub_val.add_patch(Polygon(np.array([[np.median(nses),0.0],[np.median(nses)-0.015,0.3],[np.median(nses)+0.015,0.3]]),
                               facecolor='black', edgecolor='black', linewidth=0.0,zorder = 800, alpha=0.8))


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
