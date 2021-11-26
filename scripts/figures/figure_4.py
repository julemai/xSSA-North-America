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
#     run figure_4.py -p figure_4.pdf


#!/usr/bin/env python
from __future__ import print_function

"""

Plots time-dependent sensitivity plots for selected basins (representative of XX clusters of Knoben climate index)

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
                                      description='''Plots time-dependent sensitivity plots for selected basins (representative of XX clusters of Knoben climate index).''')
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
    import time
    import skfuzzy as fuzz                  # c-means fuzzy clustering
    import json
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
    basin_ids_sa = np.transpose(sread("../data/basins_5yr_nse-gt-05.dat",skip=0))[0]

    # -------------------------------------------------------------------------
    # Read basin metadata
    # -------------------------------------------------------------------------

    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac
    head = fread("../data/basin_physical_characteristics.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data/basin_physical_characteristics.txt",skip=1,separator=';',cname=["lat","lon","area_km2","elevation_m","slope_deg","forest_frac"],sname=["basin_id","basin_name"])

    dict_metadata = {}
    for ibasin in range(len(meta_float)):

        if meta_string[ibasin][0] in basin_ids_sa:
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
    # Read basin climate index information results
    # -------------------------------------------------------------------------

    # file is a JSON that starts with "var myData = { ... }" as this is how it is used by HTML javascript
    file_sa = open("../data/mydata_download_Point.js", 'r')
    tmp = file_sa.readline()
    file_sa.close()

    tmp = json.loads(tmp.split("=")[1])['features']

    dict_climate = {}
    for ii in range(len(tmp)):

        if tmp[ii]['properties']['is_xSSA']:
            #
            # contains: dict_keys(['id', 'name', 'elevation_m', 'area_km2', 'slope_deg',
            #                      'snowice_frac', 'urban_frac', 'crops_frac', 'grass_frac', 'forest_frac', 'perimeter_km', 'aspect_deg', 'wetland_frac', 'shrubs_frac', 'gravelius_frac', 'water_frac',
            #                      'lat_deg', 'lon_deg', 'lat_gauge_deg', 'lon_gauge_deg', 'shapefile',
            #                      'is_xSSA', 'figure_wSi_wSTi_all', 'wsti_processes', 'figure_wSTi_processes', 'wsi_processes',
            #                       'is_calibrated', 'calibrated_model_setup', 'kge', 'nse', 'rmse',
            #                      'rgb_climate_zone', 'hex_climate_zone', 'aridity',  'seasonality', 'frac_p_as_snow'])
            #
            dict_basin                     = {}
            dict_basin["lat"]              = tmp[ii]['properties']['lat_gauge_deg']
            dict_basin["lon"]              = tmp[ii]['properties']['lon_gauge_deg']
            dict_basin['aridity']          = tmp[ii]['properties']['aridity']
            dict_basin['seasonality']      = tmp[ii]['properties']['seasonality']
            dict_basin['frac_p_as_snow']   = tmp[ii]['properties']['frac_p_as_snow']
            dict_basin['rgb_climate_zone'] = tmp[ii]['properties']['rgb_climate_zone']

            dict_climate[tmp[ii]['properties']['id']] = dict_basin

    del tmp

    # -------------------------------------------------------------------------
    # sort basins starting with largest
    # -------------------------------------------------------------------------
    areas = []
    for ibasin_id,basin_id in enumerate(basin_ids_sa):
        areas.append(dict_metadata[basin_id]["area_km2"])
    areas = np.array(areas)
    idx_areas = np.argsort(areas)[::-1]
    basin_ids_sa = np.array(basin_ids_sa)[idx_areas]


    # --------------------------------------------------------------------------
    # Clustering of climate indicators and select representative basins
    # --------------------------------------------------------------------------
    latlon  = np.transpose(np.array([[ dict_climate[bb]['lat'] for bb in basin_ids_sa ],
                                     [ dict_climate[bb]['lon'] for bb in basin_ids_sa ]]))
    alldata = np.vstack(([ dict_climate[bb]['aridity']        for bb in basin_ids_sa ],
                         [ dict_climate[bb]['seasonality']    for bb in basin_ids_sa ],
                         [ dict_climate[bb]['frac_p_as_snow'] for bb in basin_ids_sa ]))

    ncenters = 8
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None, seed=42)


    # -------------------------------------------------------------------------
    # Setup
    #
    dowhite    = False   # True: black background, False: white background
    title      = False   # True: title on plots,   False: no plot titles
    textbox    = False   # if true: additional information is set as text box within plot
    textbox_x  = 0.95
    textbox_y  = 0.85


    # -------------------------------------------------------------------------
    # Plotting of results
    # -------------------------------------------------------------------------

    # Main plot
    nrow        = ncenters #4           # # of rows of subplots per figure
    ncol        = 6           # # of columns of subplots per figure
    hspace      = 0.01         # x-space between subplots
    vspace      = -0.14        # y-space between subplots
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
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D

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

    # -------------------------------------------------------------------------
    # Colors for processes
    # -------------------------------------------------------------------------
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
    soilm_color = (0.7,0.7,0.7) #color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[255]

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

    max_sti = 1.0
    min_sti = 0.01 # 0.00001  # 0.0

    # -----------------------------------------
    # Figure: Map of all basins per category on map and stacked sensitivities of one representative basin
    # -----------------------------------------

    regions = ['Coastal and Interior Plains',
               'Arid Regions and Florida',
               'Mediterranean California and Temperate Sierra',
               'Temperate Broadleaf and Mixed Forests',
               'Boreal Forest',
               'Temperate Coniferous Forests',
               'Strongly Seasonal and Snow-Dominated Regions',
               'Montane Cordillera']

    for iicenter,icenter in enumerate([1,0,6,2,7,3,4,5]): #enumerate([2]): #range(ncenters):  #

        iplot = iicenter*ncol + 1

        cluster_membership = np.argmax(u, axis=0)
        basin_ids_sa_in_cat = basin_ids_sa[cluster_membership == icenter]

        # representative basin is the one that is closest to the center
        basin_ref_in_cat = basin_ids_sa_in_cat[np.argmin([ np.sum(([dict_climate[bb]['aridity'],dict_climate[bb]['seasonality'],dict_climate[bb]['frac_p_as_snow']]-cntr[icenter])**2) for bb in basin_ids_sa_in_cat ])]

        print("Category #",icenter+1,"   --> representative basin: ",basin_ref_in_cat, "   (",regions[iicenter],")")

        # ------------------------------------------------------------
        # (A) Map with basins in current cluster
        # ------------------------------------------------------------

        #     [left, bottom, width, height]
        # print("iplot = ",iplot)
        pos = position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)
        # print("pos plot: ",pos)
        sub = fig.add_axes(pos,frameon=False) #, axisbg='none')

        for spine in sub.spines.values():
            spine.set_linewidth(0.0)

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
        # map4.drawparallels(np.arange(-80.,81.,10.),  labels=[1,1,0,0], linewidth=0.1, color='0.5') #dashes=[1,1],
        # map4.drawmeridians(np.arange(-170.,181.,20.),labels=[0,0,1,1], linewidth=0.1, color='0.5') #dashes=[1,1],
        # map4.drawparallels(np.arange(-80.,81.,10.),  labels=[0,0,0,0], linewidth=0.1, color='0.5') #dashes=[1,1],
        # map4.drawmeridians(np.arange(-170.,181.,20.),labels=[0,0,0,0], linewidth=0.1, color='0.5') #dashes=[1,1],

        # draw cooastlines and countries
        # map4.drawcoastlines(linewidth=0.3)
        map4.drawmapboundary(fill_color='white', linewidth=0.2, ax=sub)
        map4.ax.set_frame_on(False)
        # map4.drawmapboundary(fill_color=ocean_color, linewidth=0.2)
        map4.drawcountries(color='0.8', linewidth=0.2, ax=sub)
        map4.ax.set_frame_on(False)
        # map4.fillcontinents(color='white', lake_color=ocean_color)
        map4.fillcontinents(color='gray', lake_color='white', ax=sub)
        map4.ax.set_frame_on(False)

        # add label with current number of basins
        #if iplot == ((ncenters-1)*ncol + 1):
        sub.text(0.00,0.0,str2tex("$\mathrm{N}_\mathrm{basins}="+str(len(basin_ids_sa_in_cat))+"$",usetex=usetex),
                     verticalalignment='bottom',horizontalalignment='right',rotation=90,
                     fontsize=textsize-2,transform=sub.transAxes)

        # # add region name (below map)
        # sub.text(0.00,-0.03,str2tex(regions[iicenter],usetex=usetex),
        #              verticalalignment='top',horizontalalignment='left',rotation=0,
        #              fontsize=textsize-2,transform=sub.transAxes)

        # add region name (above map)
        sub.text(0.0,1.0,str2tex(regions[iicenter],usetex=usetex),
                     verticalalignment='bottom',horizontalalignment='left',
                     #fontweight='bold',
                     fontsize=textsize-0,transform=sub.transAxes)

        # add ABC
        sub.text(-0.05,1.0,str2tex(chr(96+(iicenter+1)),usetex=usetex),
                     verticalalignment='top',horizontalalignment='right',
                     fontweight='bold',
                     fontsize=textsize+2,transform=sub.transAxes)

        cticks_log   = np.array([ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ])
        coord_catch   = []
        for ibasin_id,basin_id in enumerate(basin_ids_sa_in_cat):

            icolor = tuple(dict_climate[basin_id]['rgb_climate_zone'])

            xpt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[0]
            ypt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[1]
            x2,  y2   = (1.1,0.95-ibasin_id*0.1)

            sub.plot(xpt, ypt,
                     linestyle='None', marker='o', markeredgecolor=icolor, markerfacecolor=icolor,
                     markersize=1.7, markeredgewidth=0.0)

            # mark representative basin
            xpt_rep = map4(dict_metadata[basin_ref_in_cat]['lon'],dict_metadata[basin_ref_in_cat]['lat'])[0]
            ypt_rep = map4(dict_metadata[basin_ref_in_cat]['lon'],dict_metadata[basin_ref_in_cat]['lat'])[1]

            sub.plot(xpt_rep, ypt_rep,
                     linestyle='None', marker='x', markeredgecolor='black', markerfacecolor='None',
                     markersize=3.7, markeredgewidth=0.5)

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

        # only way to turn of frame is to plot a white line :(
        sub.plot([0,1,1,0,0],[0,0,1,1,0],transform=sub.transAxes,color='white',linewidth=2.0)


        # --------------------------------------------
        # (B) stacked sensitivities for representative basin in the current cluster
        # --------------------------------------------

        #     [left, bottom, width, height]
        pos = position(nrow,ncol,iplot,hspace=hspace,vspace=vspace) + [0.18,0.078,0.4,-0.156]
        # print("pos plot: ",pos)
        sub = fig.add_axes(pos,frameon=False) #, axisbg='none')

        # -------------------------------------------------------------------
        # read SA data and weights from CSVs for this specific basin
        # -------------------------------------------------------------------
        # date, Precipitation Correction $W$, Rain-Snow Partitioning $V$, Percolation $U$, Potential Melt $T$,
        #       Convolution (dlyd runoff) $S$, Convolution (srfc runoff) $R$, Snow Balance $Q$, Baseflow $P$,
        #       Evaporation $O$, Quickflow $N$, Infiltration $M$
        basin_ref_in_cat_sa_data = fread('../data/xSSA_analysis/'+basin_ref_in_cat+'_nsets1000_wSTi_processes.csv',skip=1,cskip=1)[:,::-1]
        basin_ref_in_cat_weights = fread('../data/xSSA_analysis/'+basin_ref_in_cat+'_nsets1000_wSTi_weights.csv',  skip=1,cskip=1)
        ntime_doy = np.shape(basin_ref_in_cat_sa_data)[0]
        nproc     = np.shape(basin_ref_in_cat_sa_data)[1]

        # add gauge id as label
        sub.text(1.0,1.03,str2tex("Representative basin: "+basin_ref_in_cat,usetex=usetex),
                     verticalalignment='bottom',horizontalalignment='right',
                     #fontweight='bold',
                     fontsize=textsize-0,transform=sub.transAxes)

        # colors for all processes
        colors = [infil_color, quick_color, evapo_color, basef_color, snowb_color, convs_color, convd_color, potme_color, perco_color, rspar_color, rscor_color, soilm_color]

        # stacked sensitivities
        width = 1.0
        for iproc in range(nproc):
            p1 = sub.bar(np.arange(ntime_doy),
                         basin_ref_in_cat_sa_data[:,iproc],
                         width,
                         color=colors[iproc],
                         bottom=np.sum(basin_ref_in_cat_sa_data[:,0:iproc],axis=1))

        # weights
        sub2 = sub.twinx()
        sub2.plot(np.arange(ntime_doy),basin_ref_in_cat_weights,color='black',linewidth=lwidth*2)
        sub2.set_ylabel(str2tex('Weight',usetex=usetex), color='black')

        # only for month ticks instead of DOY
        monthlength = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        sub.set_xticks(np.cumsum(monthlength)) #, ['J','F','M','A','M','J','J','A','S','O','N','D'])
        sub.set_xticklabels('')
        sub.set_xticks((monthlength*1.0/2.)+np.cumsum(np.append(0,monthlength)[0:12]), minor=True)
        if iplot == ((ncenters-1)*ncol + 1):
            sub.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], minor=True)
        else:
            sub.set_xticklabels(['','','','','','','','','','','',''], minor=True)
        sub.tick_params(which='minor',length=0)   # dont show minor ticks; only labels

        # align y-axis ticks
        for itick,tick in enumerate(sub.yaxis.get_major_ticks()):
            if (itick==0):  # 0.0
                tick.label1.set_verticalalignment('bottom')
            if (itick==1):  # 0.5
                tick.label1.set_verticalalignment('center')
            if (itick==2):  # 1.0
                tick.label1.set_verticalalignment('top')

        # range of axis
        ylim  = [0.0, 1.0]
        ylim2 = [0.0, 0.0018]
        sub.set_xlim([0,ntime_doy])
        sub.set_ylim(ylim)
        sub2.set_ylim(ylim2)

        # axis label
        if iplot == ((ncenters-1)*ncol + 1):
            sub.set_xlabel(str2tex("Day of Year",usetex=usetex))
        else:
            sub.set_xlabel(str2tex("",usetex=usetex))
        if iplot%2 == 1:
            sub.set_ylabel(str2tex("$ST_i$",usetex=usetex))  # (normalized) Total\n
        else:
            sub.set_ylabel(str2tex("",usetex=usetex))

        # DOY on ticks only at last plot
        if iplot != ((ncenters-1)*ncol + 1):
            sub.set_xticklabels(str2tex("",usetex=usetex))

        # # add ABC
        # sub.text(0.98,0.96,str2tex(chr(96+(2*iicenter+2)),usetex=usetex),
        #              verticalalignment='top',horizontalalignment='right',
        #              fontweight='bold',
        #              fontsize=textsize+2,transform=sub.transAxes)

        # legend for climate indexes
        if iplot == (ncenters-1)*ncol + 1:
            # print("plot legend for iplot = ",iplot)
            #     [left, bottom, width, height]
            pos_col_legend = position(nrow,ncol,iplot,hspace=hspace,vspace=vspace, top=0.076, bottom=0.070)
            #print("pos_col_legend = ",pos_col_legend)

            # w/ ticklabels  w/o ticks
            # csub_red    = fig.add_axes( pos_col_legend - [-0.01,-0.08+0.00,0.02,0.113]  )
            # csub_green  = fig.add_axes( pos_col_legend - [-0.01,-0.08+0.03,0.02,0.113]  )
            # csub_blue   = fig.add_axes( pos_col_legend - [-0.01,-0.08+0.06,0.02,0.113]  )

            # w/o ticklabels  w/o ticks
            csub_red    = fig.add_axes( pos_col_legend - [-0.032,-0.08+0.014,0.03,0.113]  )
            csub_green  = fig.add_axes( pos_col_legend - [-0.032,-0.08+0.026,0.03,0.113]  )
            csub_blue   = fig.add_axes( pos_col_legend - [-0.032,-0.08+0.038,0.03,0.113]  )

            # colormaps
            cc         = plt.get_cmap('Reds')   # color.get_brewer('BlWhRe', rgb=True)[52:][::-1]       # Red --> White
            cmap_red   = cc.reversed()          # mpl.colors.ListedColormap(cc)
            cc         = plt.get_cmap('Greens') # color.get_brewer('WhiteGreen', rgb=True)[2:]          # White --> Green
            cmap_green = cc                     # mpl.colors.ListedColormap(cc)
            cc         = plt.get_cmap('Blues')  # color.get_brewer('BlWhRe', rgb=True)[2:53][::-1]      # White --> Blue
            cmap_blue  = cc                     # mpl.colors.ListedColormap(cc)

            cbar_red   = mpl.colorbar.ColorbarBase(csub_red,   norm=mpl.colors.Normalize(vmin=-1.0, vmax=1.0), cmap=cmap_red,   orientation='horizontal', alpha=0.5)
            cbar_green = mpl.colorbar.ColorbarBase(csub_green, norm=mpl.colors.Normalize(vmin= 0.0, vmax=2.0), cmap=cmap_green, orientation='horizontal', alpha=0.5)
            cbar_blue  = mpl.colorbar.ColorbarBase(csub_blue,  norm=mpl.colors.Normalize(vmin= 0.0, vmax=1.0), cmap=cmap_blue,  orientation='horizontal', alpha=0.5)

            cbar_red.ax.tick_params(labelsize=textsize-2)
            cbar_green.ax.tick_params(labelsize=textsize-2)
            cbar_blue.ax.tick_params(labelsize=textsize-2)

            cbar_red.ax.text( -0.55,1.2,str2tex("Climate Index (RGB):",usetex=usetex),         fontsize=textsize-2, color='black', horizontalalignment='left', verticalalignment='bottom', transform=cbar_red.ax.transAxes)
            cbar_red.ax.text(  0.5,0.5,str2tex("Aridity",usetex=usetex),         fontsize=textsize-2, color='black', horizontalalignment='center', verticalalignment='center', transform=cbar_red.ax.transAxes)
            cbar_green.ax.text(0.5,0.5,str2tex("Seasonality",usetex=usetex),     fontsize=textsize-2, color='black', horizontalalignment='center', verticalalignment='center', transform=cbar_green.ax.transAxes)
            cbar_blue.ax.text( 0.5,0.5,str2tex("Precip. as snow",usetex=usetex), fontsize=textsize-2, color='black', horizontalalignment='center', verticalalignment='center', transform=cbar_blue.ax.transAxes)

            cbar_red.ax.text(    -0.05,0.5,str2tex("arid",usetex=usetex),        fontsize=textsize-2, color='black', horizontalalignment='right', verticalalignment='center', transform=cbar_red.ax.transAxes)
            cbar_red.ax.text(     1.05,0.5,str2tex("wet",usetex=usetex),         fontsize=textsize-2, color='black', horizontalalignment='left', verticalalignment='center', transform=cbar_red.ax.transAxes)
            cbar_green.ax.text(  -0.05,0.5,str2tex("constant",usetex=usetex),    fontsize=textsize-2, color='black', horizontalalignment='right', verticalalignment='center', transform=cbar_green.ax.transAxes)
            cbar_green.ax.text(   1.05,0.5,str2tex("seasonal",usetex=usetex),    fontsize=textsize-2, color='black', horizontalalignment='left', verticalalignment='center', transform=cbar_green.ax.transAxes)
            cbar_blue.ax.text(   -0.05,0.5,str2tex("no snow",usetex=usetex),     fontsize=textsize-2, color='black', horizontalalignment='right', verticalalignment='center', transform=cbar_blue.ax.transAxes)
            cbar_blue.ax.text(    1.05,0.5,str2tex("all snow",usetex=usetex),    fontsize=textsize-2, color='black', horizontalalignment='left', verticalalignment='center', transform=cbar_blue.ax.transAxes)

            cbar_red.ax.tick_params(size=0)    # remove ticks
            cbar_green.ax.tick_params(size=0)  # remove ticks
            cbar_blue.ax.tick_params(size=0)   # remove ticks
            cbar_red.set_ticks([])             # remove ticklabels
            cbar_green.set_ticks([])           # remove ticklabels
            cbar_blue.set_ticks([])            # remove ticklabels


        # legend for processes
        if iplot == ((ncenters-1)*ncol + 1):

            xshift=0.1
            yshift=-0.1

            # Create custom artists
            #      (left, bottom), width, height
            boxSTi_1  = patches.Rectangle( (0.00+xshift, -0.57+yshift), 0.02, 0.05, color = infil_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_2  = patches.Rectangle( (0.00+xshift, -0.67+yshift), 0.02, 0.05, color = quick_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_3  = patches.Rectangle( (0.00+xshift, -0.77+yshift), 0.02, 0.05, color = evapo_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_4  = patches.Rectangle( (0.00+xshift, -0.87+yshift), 0.02, 0.05, color = basef_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_5  = patches.Rectangle( (0.22+xshift, -0.57+yshift), 0.02, 0.05, color = snowb_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_6  = patches.Rectangle( (0.22+xshift, -0.67+yshift), 0.02, 0.05, color = convs_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_7  = patches.Rectangle( (0.22+xshift, -0.77+yshift), 0.02, 0.05, color = convd_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_8  = patches.Rectangle( (0.22+xshift, -0.87+yshift), 0.02, 0.05, color = potme_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_9  = patches.Rectangle( (0.55+xshift, -0.57+yshift), 0.02, 0.05, color = perco_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_10 = patches.Rectangle( (0.55+xshift, -0.67+yshift), 0.02, 0.05, color = rspar_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            boxSTi_11 = patches.Rectangle( (0.55+xshift, -0.77+yshift), 0.02, 0.05, color = rscor_color, alpha=0.6, fill  = True, transform=sub.transAxes, clip_on=False )
            line      = patches.Rectangle( (0.55+xshift, -0.84+yshift), 0.02, 0.00, color = 'black',     alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
            sub.add_patch(boxSTi_1)
            sub.add_patch(boxSTi_2)
            sub.add_patch(boxSTi_3)
            sub.add_patch(boxSTi_4)
            sub.add_patch(boxSTi_5)
            sub.add_patch(boxSTi_6)
            sub.add_patch(boxSTi_7)
            sub.add_patch(boxSTi_8)
            sub.add_patch(boxSTi_9)
            sub.add_patch(boxSTi_10)
            sub.add_patch(boxSTi_11)
            sub.add_patch(line)

            sub.text(0.00+xshift, -0.42+yshift, str2tex("(Normalized) Total Sobol' Index $ST_i$ of Processes:",usetex=usetex), fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.04+xshift, -0.56+yshift, str2tex("Infiltration $M$",usetex=usetex),             fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.04+xshift, -0.66+yshift, str2tex("Quickflow $N$",usetex=usetex),                fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.04+xshift, -0.76+yshift, str2tex("Evaporation $O$",usetex=usetex),              fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.04+xshift, -0.86+yshift, str2tex("Baseflow $P$",usetex=usetex),                 fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.26+xshift, -0.56+yshift, str2tex("Snow Balance $Q$",usetex=usetex),             fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            #sub.text(0.26+xshift, -0.66+yshift, str2tex("Surface Runoff $R$",usetex=usetex),           fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            #sub.text(0.26+xshift, -0.76+yshift, str2tex("Delayed Runoff $S$",usetex=usetex),           fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.26+xshift, -0.66+yshift, str2tex("Convolution (srfc runoff) $R$",usetex=usetex), fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.26+xshift, -0.76+yshift, str2tex("Convolution (dlyd runoff) $S$",usetex=usetex), fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.26+xshift, -0.86+yshift, str2tex("Potential Melt $T$",usetex=usetex),           fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.59+xshift, -0.56+yshift, str2tex("Percolation $U$",usetex=usetex),              fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.59+xshift, -0.66+yshift, str2tex("Rain-Snow Partitioning $V$",usetex=usetex),   fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.59+xshift, -0.76+yshift, str2tex("Precipitation Correction $W$",usetex=usetex), fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
            sub.text(0.59+xshift, -0.86+yshift, str2tex("Weight of Timestep",usetex=usetex),           fontsize=textsize-2, horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)


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
