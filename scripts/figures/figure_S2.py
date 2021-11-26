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
#     run figure_S2.py

#    run figure_S2.py -g figure_S2_


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
donolabel = True  # True: does not plot labels at points on map

"""
Plots comparison with Markstrom FAST estimates over CONUS

History
-------
Written,  JM, Apr 2021
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

    # -----------------------
    # add subolder scripts/lib to search path
    # -----------------------
    import sys
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path+'/lib')

    import numpy as np
    from scipy.stats import spearmanr
    from scipy.stats import pearsonr
    import time

    t1 = time.time()

    from fsread                 import fsread                  # in lib/
    from fread                  import fread                   # in lib/
    from   autostring           import astr                    # in lib/
    import color                                               # in lib/
    from   position             import position                # in lib/
    from   abc2plot             import abc2plot                # in lib/
    from   brewer               import get_brewer              # in lib/
    from   str2tex              import str2tex                 # in lib/

    # -------------------------------------------------------------------------
    # Read Markstrom FAST results
    # -------------------------------------------------------------------------

    meta_float, meta_string = fsread("../data/sa_for_markstrom_FAST.dat",skip=1,separator=',',snc=1,nc=2)
    meta_string = np.transpose(meta_string)[0]

    dict_sa_results_FAST = {}
    for ii,ibasin in enumerate(meta_string):

        dict_sa_results_FAST[ibasin] = meta_float[ii][0]

    # -------------------------------------------------------------------------
    # Read xSSA results
    # -------------------------------------------------------------------------

    meta_float, meta_string = fsread("../data/sa_for_markstrom_xSSA.dat",skip=1,separator=',',snc=1,nc=6)
    meta_string = np.transpose(meta_string)[0]

    idx = np.where(np.array([ meta_float[ii][4] for ii,ibasin in enumerate(meta_string) ])>1.0)[0]
    nbasins_Si_too_large = np.shape(idx)[0]
    print("Number of basins where sum_Si > 1.0: ",nbasins_Si_too_large," (of ",len(meta_float)," basins)")
    median_Si_too_large = np.median(np.array([ meta_float[ii][4] for ii,ibasin in enumerate(meta_string) ])[idx])
    print("Median sum of Si in those basins: ",median_Si_too_large)

    dict_sa_results_xSSA = {}
    for ii,ibasin in enumerate(meta_string):

        tmp_dict = {}
        tmp_dict['msi']       = meta_float[ii][0]
        tmp_dict['msti']      = meta_float[ii][1]
        tmp_dict['wsi']       = meta_float[ii][2]
        tmp_dict['wsti']      = meta_float[ii][3]
        tmp_dict['si_qmean']  = min(meta_float[ii][4],1.0) #meta_float[ii][4] #
        tmp_dict['sti_qmean'] = meta_float[ii][5]
        dict_sa_results_xSSA[ibasin] = tmp_dict

    # -------------------------------------------------------------------------
    # Read basin metadata
    # -------------------------------------------------------------------------

    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac
    head = fread("../data/basin_physical_characteristics.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("../data/basin_physical_characteristics.txt",skip=1,separator=';',cname=["lat","lon","area_km2","elevation_m","slope_deg","forest_frac"],sname=["basin_id","basin_name"])

    dict_metadata = {}
    for ibasin in range(len(meta_float)):

        if meta_string[ibasin][0] in dict_sa_results_FAST:
            dict_basin = {}
            dict_basin["name"]        = meta_string[ibasin][1]
            dict_basin["lat"]         = meta_float[ibasin][0]
            dict_basin["lon"]         = meta_float[ibasin][1]
            dict_basin["area_km2"]    = meta_float[ibasin][2]
            dict_basin["elevation_m"] = meta_float[ibasin][3]
            dict_basin["slope_deg"]   = meta_float[ibasin][4]
            dict_basin["forest_frac"] = meta_float[ibasin][5]

            dict_metadata[meta_string[ibasin][0]] = dict_basin

    # all basins in list
    basin_ids_sa = list(dict_sa_results_FAST.keys())

    # dont use basins with FAST -9999 (basins not analysed by Markstrom)
    sa_results_valid = np.array([ [ dict_sa_results_FAST[ibasin],
                                    dict_sa_results_xSSA[ibasin]["msi"],
                                    dict_sa_results_xSSA[ibasin]["msti"],
                                    dict_sa_results_xSSA[ibasin]["wsi"],
                                    dict_sa_results_xSSA[ibasin]["wsti"],
                                    dict_sa_results_xSSA[ibasin]["si_qmean"],
                                    dict_sa_results_xSSA[ibasin]["sti_qmean"],
                                    dict_metadata[ibasin]["lat"],
                                    dict_metadata[ibasin]["lon"],
                                    dict_metadata[ibasin]["area_km2"] ] for ibasin in basin_ids_sa if dict_sa_results_FAST[ibasin] > 0.0 ])
    corr_Markstrom_xSSA_msi       = pearsonr(sa_results_valid[:,0],sa_results_valid[:,1])[0]
    corr_Markstrom_xSSA_msti      = pearsonr(sa_results_valid[:,0],sa_results_valid[:,2])[0]
    corr_Markstrom_xSSA_wsi       = pearsonr(sa_results_valid[:,0],sa_results_valid[:,3])[0]
    corr_Markstrom_xSSA_wsti      = pearsonr(sa_results_valid[:,0],sa_results_valid[:,4])[0]
    corr_Markstrom_xSSA_si_qmean  = pearsonr(sa_results_valid[:,0],sa_results_valid[:,5])[0]
    corr_Markstrom_xSSA_sti_qmean = pearsonr(sa_results_valid[:,0],sa_results_valid[:,6])[0]

    # print("Correlation FAST vs msi:       ",corr_Markstrom_xSSA_msi)
    # print("Correlation FAST vs msti:      ",corr_Markstrom_xSSA_msti)
    # print("Correlation FAST vs wsi:       ",corr_Markstrom_xSSA_wsi)
    # print("Correlation FAST vs wsti:      ",corr_Markstrom_xSSA_wsti)
    print("Correlation FAST vs si_qmean:  ",corr_Markstrom_xSSA_si_qmean,"   <<<<<<<<<<<")
    # print("Correlation FAST vs sti_qmean: ",corr_Markstrom_xSSA_sti_qmean)








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
    msize       = 0.3         # marker size
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

    from matplotlib.patches import Rectangle, Circle, Polygon
    from mpl_toolkits.basemap import Basemap

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

    idx_xSSA = 5   # 1=msi, 2=msti, 3=wsi, 4=wsti, 5=si_qmean, 6=sti_qmean
    min_xSSA = np.percentile(sa_results_valid[:,idx_xSSA],5.0)  # np.min(sa_results_valid[:,idx_xSSA])
    max_xSSA = np.percentile(sa_results_valid[:,idx_xSSA],95.0) # np.max(sa_results_valid[:,idx_xSSA])

    min_FAST = np.percentile(sa_results_valid[:,0],5.0) #np.min(sa_results_valid[:,0])
    max_FAST = np.percentile(sa_results_valid[:,0],95.0)

    # -----------------------------------------
    # Figure 1a: xSSA results
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
    sub.text(1.0,0.02,str2tex("$\mathrm{N}_\mathrm{basins}="+str(len(basin_ids_sa))+"$",usetex=usetex),
                 verticalalignment='bottom',horizontalalignment='right',rotation=90,
                 fontsize=textsize,transform=sub.transAxes)

    # add ABC
    sub.text(1.0,0.96,str2tex("a",usetex=usetex),
                 verticalalignment='top',horizontalalignment='right',
                 fontweight='bold',
                 fontsize=textsize+2,transform=sub.transAxes)

    # adjust frame linewidth
    # sub.outline_patch.set_linewidth(lwidth)

    coord_catch   = []
    for ibasin_id,basin_id in enumerate(basin_ids_sa):

        shapefilename =  "wrong-directory"+basin_id+"/shape.dat"    # "observations/"+basin_id+"/shape.dat"

        # color based on xSSA estimate (sum of mSI of paras)
        icolor = 'red'

        # 1=msi, 2=msti, 3=wsi, 4=wsti, 5=lat, 6=lon, 7=area_km2
        if idx_xSSA == 4:
            ixSSA = dict_sa_results_xSSA[basin_id]['wsti']
        elif idx_xSSA == 3:
            ixSSA = dict_sa_results_xSSA[basin_id]['wsi']
        elif idx_xSSA == 2:
            ixSSA = dict_sa_results_xSSA[basin_id]['msti']
        elif idx_xSSA == 1:
            ixSSA = dict_sa_results_xSSA[basin_id]['msi']
        elif idx_xSSA == 5:
            ixSSA = dict_sa_results_xSSA[basin_id]['si_qmean']
        elif idx_xSSA == 6:
            ixSSA = dict_sa_results_xSSA[basin_id]['sti_qmean']
        else:
            raise ValueError("dont know which xSSA index to plot")

        if ixSSA < min_xSSA:
            icolor = cc[0]
        elif ixSSA > max_xSSA:
            icolor = cc[-1]
        else:
            icolor = cc[int((ixSSA-min_xSSA)/(max_xSSA-min_xSSA)*(len(cc)-1))]

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

    # colorbar
    # [left, bottom, width, height]
    pos_cbar = [0.13,0.76,0.2,0.006]
    print("pos cbar: ",pos_cbar)
    csub1    = fig.add_axes( pos_cbar )

    inorm = 'linear'
    if inorm == 'linear':
        ticks = [ min_xSSA + (ii*(max_xSSA - min_xSSA))/len(cc) for ii in range(len(cc)+1) ]
        cbar1 = mpl.colorbar.ColorbarBase(csub1, cmap=cmap, norm=mpl.colors.Normalize(vmin=min_xSSA, vmax=max_xSSA), ticks=ticks, orientation='horizontal') #, extend='min')
        cbar1.set_ticklabels([ "{:.2f}".format(itick) if (iitick%2 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
    elif inorm == 'log':
        ticks = [ 10.0**(np.log10(min_xSSA) + ii*(np.log10(max_xSSA) - np.log10(min_xSSA))/len(cc)) for ii in range(len(cc)+1) ]
        cbar1 = mpl.colorbar.ColorbarBase(csub1, norm=norm, cmap=cmap, orientation='horizontal', extend='min')
        cbar1.set_ticklabels([ "{:.1e}".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
    elif inorm == 'pow':
        ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
        ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
        cbar1 = mpl.colorbar.ColorbarBase(csub1, norm=norm, ticks=ticks, cmap=cmap, orientation='horizontal', extend='max')
        cbar1.set_ticklabels([ "{:.0e}".format(itick) for itick in ticks ])
    else:
        raise ValueError('Norm for colormap not known.')
    cbar1.set_label(str2tex("xSSA sensitivity $\sum S_i(\overline{Q_t})$ [-]",usetex=usetex))

    # -----------------------------------------
    # Figure 1b: FAST results
    # -----------------------------------------
    iplot += 1

    #     [left, bottom, width, height]
    pos = [0.7,0.8,0.45,0.15]
    # sub = fig.add_axes(pos, projection=ccrs.LambertConformal())
    sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)+[0.0,0.0,0.0,0.0],frameon=False) #, axisbg='none')
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
    sub.text(1.0,0.02,str2tex("$\mathrm{N}_\mathrm{basins}="+str(len(sa_results_valid))+"$",usetex=usetex),
                 verticalalignment='bottom',horizontalalignment='right',rotation=90,
                 fontsize=textsize,transform=sub.transAxes)

    # add ABC
    sub.text(1.0,0.96,str2tex("b",usetex=usetex),
                 verticalalignment='top',horizontalalignment='right',
                 fontweight='bold',
                 fontsize=textsize+2,transform=sub.transAxes)

    coord_catch   = []
    for ibasin_id,basin_id in enumerate(basin_ids_sa):

        shapefilename =  "wrong-directory"+basin_id+"/shape.dat"    # "observations/"+basin_id+"/shape.dat"

        # color based on NSE
        icolor = 'red'
        iFAST = dict_sa_results_FAST[basin_id]
        if iFAST < min_FAST:
            icolor = cc[0]
        elif iFAST > max_FAST:
            icolor = cc[-1]
        else:
            icolor = cc[int((iFAST-min_FAST)/(max_FAST-min_FAST)*(len(cc)-1))]

        if iFAST >= 0.0: # some FAST results are -9999.0 because they were not analysed by Markstrom

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

    # colorbar
    # [left, bottom, width, height]
    pos_cbar = [0.408,0.76,0.2,0.006]
    print("pos cbar: ",pos_cbar)
    csub    = fig.add_axes( pos_cbar )

    inorm = 'linear'
    if inorm == 'linear':
        ticks = [ min_FAST + (ii*(max_FAST - min_FAST))/len(cc) for ii in range(len(cc)+1) ]
        cbar = mpl.colorbar.ColorbarBase(csub, cmap=cmap, norm=mpl.colors.Normalize(vmin=min_FAST, vmax=max_FAST), ticks=ticks, orientation='horizontal') #, extend='min')
        cbar.set_ticklabels([ "{:.2f}".format(itick) if (iitick%2 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
    elif inorm == 'log':
        ticks = [ 10.0**(np.log10(min_FAST) + ii*(np.log10(max_FAST) - np.log10(min_FAST))/len(cc)) for ii in range(len(cc)+1) ]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='horizontal', extend='min')
        cbar.set_ticklabels([ "{:.1e}".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
    elif inorm == 'pow':
        ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
        ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, ticks=ticks, cmap=cmap, orientation='horizontal', extend='max')
        cbar.set_ticklabels([ "{:.0e}".format(itick) for itick in ticks ])
    else:
        raise ValueError('Norm for colormap not known.')
    cbar.set_label(str2tex("FAST sensitivity $\sum F_i(\overline{R_t})$ [-]",usetex=usetex))

    # # -----------------------------------
    # # scatterplot xSSA vs FAST
    # # -----------------------------------

    # iplot += 1

    # sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace))

    # idx_color=8  # 1=msi, 2=msti, 3=wsi, 4=wsti, 5=si_qmean, 6=sti_qmean, 7=lat, 8=lon, 9=area_km2
    # idx_xSSA=5
    # plot1 = sub.scatter(sa_results_valid[:,0],sa_results_valid[:,idx_xSSA],
    #                      #linewidth=0.0,
    #                      #marker='o',
    #                      #markersize=msize,
    #                      alpha=0.7,
    #                      s=msize*0.5,
    #                      c=(sa_results_valid[:,idx_color]-np.min(sa_results_valid[:,idx_color]))/(np.max(sa_results_valid[:,idx_color])-np.min(sa_results_valid[:,idx_color])),
    #                         cmap=cmap)

    # minval_xy = np.min(sa_results_valid[:,[0,idx_xSSA]])
    # maxval_xy = np.max(sa_results_valid[:,[0,idx_xSSA]])

    # # 1:1 line for reference
    # #sub.plot([minval_xy,maxval_xy],[minval_xy,maxval_xy],linewidth=lwidth,color='k')

    # # # title = process
    # # sub.text(0.5, 1.0, str2tex(processes_clear[iprocess],usetex=usetex), transform=sub.transAxes,
    # #              rotation=0, fontsize=textsize+1,
    # #              horizontalalignment='center', verticalalignment='bottom')

    # # sub.text(0.96, 0.12, str2tex('$\\rho_{cal} = '+astr(np.round(np.mean(coef_calib),3),prec=3)+' \pm '+astr(np.round(np.std(coef_calib),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
    # #              rotation=0, fontsize=textsize-2,
    # #              horizontalalignment='right', verticalalignment='bottom')
    # # sub.text(0.96, 0.04, str2tex('$\\rho_{val} = '+astr(np.round(np.mean(coef_valid),3),prec=3)+' \pm '+astr(np.round(np.std(coef_valid),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
    # #              rotation=0, fontsize=textsize-2,
    # #              horizontalalignment='right', verticalalignment='bottom')
    # # sub.text(0.04, 0.96, str2tex('$MAE_{cal} = '+astr(np.round(np.mean(mae_calib),3),prec=3)+' \pm '+astr(np.round(np.std(mae_calib),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
    # #              rotation=0, fontsize=textsize-2,
    # #              horizontalalignment='left', verticalalignment='top')
    # # sub.text(0.04, 0.88, str2tex('$MAE_{val} = '+astr(np.round(np.mean(mae_valid),3),prec=3)+' \pm '+astr(np.round(np.std(mae_valid),3),prec=3)+'$',usetex=usetex), transform=sub.transAxes,
    # #              rotation=0, fontsize=textsize-2,
    # #              horizontalalignment='left', verticalalignment='top')

    # # x-labels
    # if (iplot == 1):
    #     xlabel=str2tex('FAST-derived sensitivity [-]',usetex=usetex)
    # else:
    #     xlabel=''

    # # y-labels
    # if (iplot == 1):
    #     ylabel=str2tex("xSSA-derived sensitivity $\sum ST_i^m$ [-]",usetex=usetex)
    # else:
    #     ylabel=''

    # # # 1:1 line for reference
    # # sub.plot([0.0,maxval_xy],[0.0,maxval_xy],linewidth=lwidth,color='k')

    # # # # x-ticks only in last row
    # # # if ((iplot-1) // ncol < nrow - dummy_rows - 1):
    # # #     # sub.axes.get_xaxis().set_visible(False)
    # # #     sub.axes.get_xaxis().set_ticks([])
    # # # else:
    # # #     plt.xticks(rotation=90)

    # # # # y-ticks only in first column
    # # # if (iplot-1)%ncol != 0:
    # # #     # sub.axes.get_yaxis().set_visible(False)
    # # #     sub.axes.get_yaxis().set_ticks([])

    # # sub.set_xlim([0.0,maxval_xy])
    # # sub.set_ylim([0.0,maxval_xy])

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)

    # # # add ABC
    # # sub.text(1.00,1.0,str2tex(chr(96+iplot),usetex=usetex),
    # #                      verticalalignment='bottom',horizontalalignment='right',
    # #                      fontweight='bold',
    # #                      fontsize=textsize+2,transform=sub.transAxes)

    # # # colorbar
    # # # [left, bottom, width, height]
    # # pos_cbar = [0.2,0.38,0.6,0.01]
    # # print("pos cbar: ",pos_cbar)
    # # csub    = fig.add_axes( pos_cbar )

    # # if inorm == 'log':
    # #     ticks = [ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ]
    # #     cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='horizontal', extend='min')
    # #     cbar.set_ticklabels([ "{:.1e}".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
    # #     cbar.set_label(str2tex("Density [-]",usetex=usetex))
    # # elif inorm == 'pow':
    # #     ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
    # #     ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
    # #     cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, ticks=ticks, cmap=cmap, orientation='horizontal', extend='max')
    # #     cbar.set_ticklabels([ "{:.0e}".format(itick) for itick in ticks ])
    # #     cbar.set_label(str2tex("Frequency [-]",usetex=usetex))
    # # else:
    # #     raise ValueError('Norm for colormap not known.')




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
