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
# python environment
#     source ~/projects/rpp-hwheater/julemai/xSSA-North-America/env-3.5/bin/activate
#
# run with:
#     run figure_2.py -p figure_2-3layer.pdf
#     run figure_2.py -g figure_2-3layer


#!/usr/bin/env python
from __future__ import print_function

"""

Plots calibration results on a map

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

    pngbase   = ''
    pdffile   = ''
    usetex    = False
    
    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Benchmark example to test Sensitivity Analysis for models with multiple process options.''')
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
    from fsread        import fsread        # in lib/
    
    t1 = time.time()

    # -------------------------------------------------------------------------
    # Find basins with calibration results
    # -------------------------------------------------------------------------
    diag_files = glob.glob('ostrich-weighted-*/best/Diagnostics.csv')
    basins = [ diag_file.split('/')[0].split('-')[2] for  diag_file in diag_files ]

    # all 149 basins with NSE below 0.2 in first trial
    # basins = ["09518500", "09479350", "08124000", "09404110", "11271290", "11303000", "11234750", "11302000", "11517500", "11525500", "06848000", "11517000", "11401112", "11399500", "11413517", "12100496", "14316455", "11367800", "12100000", "11126000", "11367760", "11070210", "14315700", "14315500", "11293200", "14091500", "11216500", "11372000", "14313700", "07143375", "07141300", "08362500", "07229200", "07229050", "07228500", "08136700", "08088610", "06856000", "06853500", "06853020", "06849500", "07227500", "08405200", "08404000", "08402000", "08401500", "08126380", "06844500", "08123850", "02172002", "07242000", "06843500", "07241800", "07241550", "07241520", "07241000", "07239700", "07239500", "07239450", "06837000", "08386000", "07227000", "08123800", "02246500", "08080500", "07234000", "07298500", "06863000", "06862850", "06862700", "08136000", "08206910", "06861000", "10219000", "12472600", "08384500", "10080000", "08383500", "08121000", "08120700", "06847500", "09416000", "06860000", "06835500", "06827500", "06131000", "08382830", "09419756", "13108150", "02187252", "07312100", "07232900", "06827000", "12362500", "09401260", "09513650", "06846500", "11246700", "06846000", "06834000", "08080700", "02394000", "07233500", "07119500", "01325000", "07142575", "09537500", "08343000", "09386030", "09400568", "09385700", "06838000", "09487000", "06354580", "09330000", "07141175", "12471400", "06307740", "09415550", "08408500", "09400562", "08433000", "08317950", "08145000", "08397600", "06177500", "06425720", "08212400", "09401110", "06307600", "09486800", "08405150", "06844900", "06344300", "08334000", "08128000", "01018500", "06678000", "08405500", "08405105", "12465400", "08401900", "09404208", "09306255", "08198500", "08317200", "01019000", "08401200", "08189300"]

    # 100 random basins 
    # basins = ["09147025", "08040600", "06472000", "02294898", "13309220", "14174000", "07040450", "09304200", "05483600", "09520280", "06639000", "06836500", "13011000", "03053500", "02313000", "02312720", "02446500", "06403700", "13022500", "06885500", "13337500", "06679500", "03597860", "02440500", "05066500", "08NG053", "02OE032", "06447230", "12039500", "11255575", "04292000", "08020000", "02054530", "10AC004", "10141000", "07029500", "05586100", "02131500", "08064100", "06635000", "06625000", "11458000", "03171000", "07140000", "10239000", "07049000", "08062700", "10351700", "07064533", "02FC016", "11451000", "04181500", "05059700", "04235440", "06915800", "05496000", "04124200", "07019000", "10318500", "05300000", "13250000", "03341300", "13069500", "01534000", "02196000", "05064000", "03320000", "03434500", "12342500", "07264000", "03069500", "06482020", "06439000", "01536000", "02312000", "05554500", "03604400", "10CB001", "10146400", "06883000", "04CA003", "11390500", "12354000", "14076500", "08073600", "05382000", "03289500", "12452800", "06025500", "12344000", "05331580", "07222500", "03221000", "08KD007", "08015500", "12061500", "05419000", "06725450", "02YN002", "05TD001"] 

    # -------------------------------------------------------------------------
    # Read basin metadata
    # -------------------------------------------------------------------------

    # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac
    head = fread("basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',header=True)
    meta_float, meta_string = fsread("basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',cname=["lat","lon","area_km2","elevation_m","slope_deg","forest_frac"],sname=["basin_id","basin_name"])

    dict_metadata = {}
    for ibasin in range(len(meta_float)):

        if meta_string[ibasin][0] in basins:
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
    # Read best NSE
    # -------------------------------------------------------------------------
    dict_nse = {}
    for bb in basins:

        idx_nse = fread('ostrich-weighted-'+bb+'/best/Diagnostics.csv',skip=1,cskip=2,header=True).index('DIAG_NASH_SUTCLIFFE')  
        nse     = fread('ostrich-weighted-'+bb+'/best/Diagnostics.csv',skip=1,cskip=2,fill=True)[0,idx_nse]

        dict_result = {}
        dict_result['nse'] = nse

        dict_nse[bb] = dict_result

        # Okklahoma and Kansas basins
        # if dict_metadata[bb]["lat"] > 34 and dict_metadata[bb]["lat"] < 40 and dict_metadata[bb]["lon"] < -94.5 and dict_metadata[bb]["lon"] > -102:
        #    print(bb, ';', dict_metadata[bb]["lat"],';',dict_metadata[bb]["lon"],' ; ',dict_nse[bb]['nse'])


    # sort basins starting with largest
    areas = []
    for ibasin_id,basin_id in enumerate(basins):
        areas.append(dict_metadata[basin_id]["area_km2"])
    areas = np.array(areas)
    idx_areas = np.argsort(areas)[::-1]
    basins = np.array(basins)[idx_areas]


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
        else:
            mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            #mpl.rc('font',**{'family':'serif','serif':['times']})
        mpl.rc('text.latex', unicode=True)
    elif (outtype == 'png'):
        mpl.use('TkAgg') # set directly after import matplotlib
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

    # -----------------------------------------
    # North America map - overview
    # -----------------------------------------
    iplot += 1

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

    # # Map: Europe
    # # m = Basemap(projection='lcc', llcrnrlon=-9, llcrnrlat=35.6, urcrnrlon=25.3, urcrnrlat=53,
    # #             lat_1=50, lat_2=70, lon_0=0, resolution='i') # Lambert conformal
    # # Map: USA
    # # m = Basemap(projection='lcc',
    # #             llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
    # #             lat_1=33, lat_2=45, lon_0=-95,
    # #             resolution='i') # Lambert conformal
    # # Map: Canada - Saguenay-Lac-St-Jean region
    # llcrnrlon =  -119.0
    # urcrnrlon =  -45.0
    # llcrnrlat =   22.0
    # urcrnrlat =   60.0
    # lat_1     =   (llcrnrlat+urcrnrlat)/2.0  # first  "equator"
    # lat_2     =   (llcrnrlat+urcrnrlat)/2.0  # second "equator"
    # lat_0     =   (llcrnrlat+urcrnrlat)/2.0  # center of the map
    # lon_0     =   (llcrnrlon+urcrnrlon)/2.0  # center of the map
    # # m = Basemap(projection='lcc',
    # #             llcrnrlon=-80, llcrnrlat=43, urcrnrlon=-75, urcrnrlat=47,
    # #             lon_0=-77.5, lat_0=43, 
    # #             lat_1=44, lat_2=44, 
    # #             resolution='i') # Lambert conformal
    # map4 = Basemap(projection='lcc',
    #                 width=9000000,height=7000000,
    #                 lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.,
    #                 area_thresh=6000., # only large lakes
    #                 resolution='i') # Lambert conformal
              
    # # draw parallels and meridians.
    # # labels: [left, right, top, bottom]
    # map4.drawparallels(np.arange(-80.,81.,6.),  labels=[1,0,0,0], dashes=[1,1], linewidth=0.25, color='0.5')
    # map4.drawmeridians(np.arange(-180.,181.,15.),labels=[0,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')

    # # draw cooastlines and countries
    # map4.drawcoastlines(linewidth=0.3)
    # map4.drawmapboundary(fill_color=ocean_color, linewidth=0.3)
    # map4.drawcountries(color='black', linewidth=0.3)
    # map4.fillcontinents(color='white', lake_color=ocean_color)


    # plt.title(str2tex("Model performance",usetex=usetex))

    # add label with current number of basins
    sub.text(0.05,0.05,str2tex("$\mathrm{N}_\mathrm{basins} = "+str(len(basins))+"$",usetex=usetex),transform=sub.transAxes)

    # adjust frame linewidth 
    sub.outline_patch.set_linewidth(lwidth)

    coord_catch   = []
    for ibasin_id,basin_id in enumerate(basins):

        shapefilename =  "wrong-directory"+basin_id+"/shape.dat"    # "observations/"+basin_id+"/shape.dat"

        # color based on NSE
        icolor = 'red'
        max_nse = 1.0
        min_nse = 0.0
        inse = dict_nse[basin_id]['nse']
        if inse < min_nse:
            icolor = cc[0]
        elif inse > max_nse:
            icolor = cc[-1]
        else:
            icolor = cc[np.int((inse-min_nse)/(max_nse-min_nse)*(len(cc)-1))]        

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

            print("Basin: ",basin_id)
            print("   --> area      =  ",dict_metadata[basin_id]["area_km2"])
            print("   --> lon range = [",xxmin,",",xxmax,"]")
            print("   --> lat range = [",yymin,",",yymax,"]")

        # shapefile doesnt exist only plot dot at location
        else:   
            # xpt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[0]
            # ypt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[1]
            xpt = dict_metadata[basin_id]["lon"]
            ypt = dict_metadata[basin_id]["lat"]
            x2,  y2   = (1.1,0.95-ibasin_id*0.1)
            
            sub.plot(xpt, ypt,
                     linestyle='None', marker='o', markeredgecolor=icolor, markerfacecolor=icolor,
                     transform=ccrs.PlateCarree(), 
                     markersize=0.7, markeredgewidth=0.0)

            if not(donolabel):
                transform = ccrs.PlateCarree()._as_mpl_transform(sub)
                sub.annotate(basin_id,
                        xy=(xpt, ypt),   xycoords='data',
                        xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                        fontsize=8,
                        transform=transform,
                        verticalalignment='center',horizontalalignment='left',
                        arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                        zorder=400
                        )
            print("Basin: ",basin_id)
            print("   --> lat/lon = [",dict_metadata[basin_id]["lat"],",",dict_metadata[basin_id]["lon"],"]")
            
    # -------------------------------------------------------------------------
    # (1b) Colorbar
    # -------------------------------------------------------------------------
    iplot += 1
    csub    = fig.add_axes(position(1,1,1,hspace=hspace,vspace=vspace, left=0.2, right=0.45, top=0.778, bottom=0.772) )   # top=0.642, bottom=0.632

    cbar = mpl.colorbar.ColorbarBase(csub, norm=mpl.colors.Normalize(vmin=min_nse, vmax=max_nse), cmap=cmap, orientation='horizontal', extend='min')
    cbar.set_label(str2tex("NSE [-]",usetex=usetex))

    nses= np.array([ dict_nse[basin_id]['nse'] for basin_id in basins ])
    print('')
    print('-----------------------------------')
    print('Median NSE: ',np.median(nses))
    print('Mean   NSE: ',np.mean(nses))
    print('NSEs      : ',np.sort(nses)[0:10])
    print('-----------------------------------')
    print('')
    cticks = [ min_nse+ii*(max_nse-min_nse)/len(cc) for ii in range(len(cc)+1) ]  # cbar.get_ticks()      # tick labels
    percent_in_cat = np.diff([0]+ [ np.sum(nses < ctick) for ctick in cticks ])*100.0/np.shape(nses)[0]  
    
    # add percentages of basins with performance
    for iitick,itick in enumerate(cticks):
        if iitick == 0:
            continue
        dist_from_middle = np.abs(iitick-((len(cticks))/2.))*1.0 / ((len(cticks)-1)/2.)
        icolor = (dist_from_middle,dist_from_middle,dist_from_middle)
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

