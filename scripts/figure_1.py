#!/usr/bin/env python

# Copyright 2016-2018 Juliane Mai - juliane.mai(at)uwaterloo.ca
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

# source ~/python_env/bin/activate    (has basemap)
# run figure_1.py -p test.pdf     -i "01010000 02135200"
# run figure_1.py -p figure_1.pdf -i "14189000 03MD001 06306300 02YA001 11433500 05387440 09404900 02100500 03518000 02196484"
# run figure_1.py -p figure_1.pdf -i "01049500 01085500 01389800 01449000 01533400 01BG009 01BJ007 02012500 02034000 02100500 02196484 02197320 02358789 02404400 02433000 02469761 02OG026 02PL005 02YA001 03065000 03085000 03090500 03202400 03212980 03319000 03351000 03362500 03404500 03518000 03565000 03BF001 03MB002 03MD001 04062011 04087170 04212000 04293500 05051300 05051522 05074500 05082625 05247500 05330000 05369000 05387440 05487980 05505000 05TG003 06102000 06208500 06306300 06308500 06334630 06347000 06436800 06467600 06600100 06651500 06690500 06710000 06756100 06791800 06862850 06873460 06920500 06GA001 07260500 07311800 07363400 07BB003 07EB002 08022500 08079575 08111010 08143600 08164500 08177000 08179000 08317400 08401900 08GD008 08LB047 08NA002 08NA006 08NH130 09058030 09288100 09304600 09332100 09404900 09508500 10016900 10028500 10308200 11128000 11152050 11333500 11397500 11433500 12452800 13037500 13152500 13302000 13309220 13316500 13317000 14026000 14054000 14120000 14189000 14238000 14315700" -n 

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

# -------------------------------------------------------------------------
# Command line arguments
#

import argparse

pngbase   = ''
pdffile   = ''
usetex    = False
basin_ids = None
donolabel = False

parser  = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description='''Plot basin shape.''')
parser.add_argument('-g', '--pngbase', action='store',
                    default=pngbase, dest='pngbase', metavar='pngbase',
                    help='Name basis for png output files (default: open screen window).')
parser.add_argument('-p', '--pdffile', action='store',
                    default=pdffile, dest='pdffile', metavar='pdffile',
                    help='Name of pdf output file (default: open screen window).')
parser.add_argument('-t', '--usetex', action='store_true', default=usetex, dest="usetex",
                    help="Use LaTeX to render text in pdf.")
parser.add_argument('-i', '--basin_ids', action='store',
                    default=basin_ids, dest='basin_ids', metavar='basin_ids',
                    help='Basin ID of basins to plot. Mandatory. (default: None).')
parser.add_argument('-n', '--donolabel', action='store_true', default=donolabel, dest="donolabel",
                  help="If set, catchments are not annoteted.")

args      = parser.parse_args()
pngbase   = args.pngbase
pdffile   = args.pdffile
usetex    = args.usetex
basin_ids = args.basin_ids
donolabel = args.donolabel

# convert basin_ids to list 
basin_ids = basin_ids.strip()
basin_ids = " ".join(basin_ids.split())
basin_ids = basin_ids.split()
# print('basin_ids: ',basin_ids)

# Basin ID need to be specified
if basin_ids is None:
    raise ValueError('Basin ID (option -i) needs to given. Basin ID needs to correspond to ID of a CANOPEX basin.')

if (pdffile != '') & (pngbase != ''):
    print('\nError: PDF and PNG are mutually exclusive. Only either -p or -g possible.\n')
    parser.print_usage()
    import sys
    sys.exit()

del parser, args

# -----------------------
# add subolder scripts/lib to search path
# -----------------------
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('lib')

import color                            # in lib/
from position      import position      # in lib/
from abc2plot      import abc2plot      # in lib/
from brewer        import get_brewer    # in lib/
from autostring    import astr          # in lib/
from str2tex       import str2tex       # in lib/
from fread         import fread         # in lib/
from fsread        import fsread        # in lib/

import numpy as np
import copy                       # deep copy objects, arrays etc
import time
import os
from matplotlib.patches import Polygon, Ellipse
t1 = time.time()

# -------------------------------------------------------------------------
# Read basin metadata
# -------------------------------------------------------------------------

# basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac
head = fread("../data_in/basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',header=True)
meta_float, meta_string = fsread("../data_in/basin_metadata/basin_physical_characteristics.txt",skip=1,separator=';',cname=["lat","lon","area_km2","elevation_m","slope_deg","forest_frac"],sname=["basin_id","basin_name"])

dict_metadata = {}
for ibasin in range(len(meta_float)):
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
nrow        = 5           # # of rows of subplots per figure
ncol        = 2           # # of columns of subplots per figure
hspace      = 0.02         # x-space between subplots
vspace      = 0.05        # y-space between subplots
right       = 0.9         # right space on page
textsize    = 8           # standard text size
dxabc       = 1.0         # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
# dyabc       = -13       # % of (max-min) shift up from lower x-axis for a,b,c,... labels
dyabc       = 0.0         # % of (max-min) shift up from lower x-axis for a,b,c,... labels
dxsig       = 1.23        # % of (max-min) shift to the right from left y-axis for signature
dysig       = -0.05       # % of (max-min) shift up from lower x-axis for signature
dxtit       = 0           # % of (max-min) shift to the right from left y-axis for title
dytit       = 1.3         # % of (max-min) shift up from lower x-axis for title

lwidth      = 1.5         # linewidth
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
dpi         = 300         # 150 for testing
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
    ocean_color = color.get_brewer('accent5', rgb=True)[-1]

    cc = color.get_brewer('dark_rainbow_256', rgb=True)
    cc = cc[::-1] # reverse colors
    cmap = mpl.colors.ListedColormap(cc)


    # colors for each sub-basin from uwyellow4 (228,180,42) to gray
    graylevel = 0.2
    uwyellow = [251,213,79]
    cc = [ (    (uwyellow[0]+ii/(22.-1)*(256*graylevel-uwyellow[0]))/256.,
                (uwyellow[1]+ii/(22.-1)*(256*graylevel-uwyellow[1]))/256.,
                (uwyellow[2]+ii/(22.-1)*(256*graylevel-uwyellow[2]))/256.) for ii in range(22) ]
    cmap = mpl.colors.ListedColormap(cc)
    
    cc = color.get_brewer('Paired6', rgb=True)
    cmap = mpl.colors.ListedColormap(cc)

    
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
# North America map - overview
# -----------------------------------------
iplot += 1

#     [left, bottom, width, height]
pos = [0.1,0.8,0.45,0.15]
sub = fig.add_axes(pos)
#sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)) #, axisbg='none')

# Map: Europe
# m = Basemap(projection='lcc', llcrnrlon=-9, llcrnrlat=35.6, urcrnrlon=25.3, urcrnrlat=53,
#             lat_1=50, lat_2=70, lon_0=0, resolution='i') # Lambert conformal
# Map: USA
# m = Basemap(projection='lcc',
#             llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             lat_1=33, lat_2=45, lon_0=-95,
#             resolution='i') # Lambert conformal
# Map: Canada - Saguenay-Lac-St-Jean region
llcrnrlon =  -119.0
urcrnrlon =  -45.0
llcrnrlat =   22.0
urcrnrlat =   60.0
lat_1     =   (llcrnrlat+urcrnrlat)/2.0  # first  "equator"
lat_2     =   (llcrnrlat+urcrnrlat)/2.0  # second "equator"
lat_0     =   (llcrnrlat+urcrnrlat)/2.0  # center of the map
lon_0     =   (llcrnrlon+urcrnrlon)/2.0  # center of the map
# m = Basemap(projection='lcc',
#             llcrnrlon=-80, llcrnrlat=43, urcrnrlon=-75, urcrnrlat=47,
#             lon_0=-77.5, lat_0=43, 
#             lat_1=44, lat_2=44, 
#             resolution='i') # Lambert conformal
map4 = Basemap(projection='lcc',
                width=9000000,height=7000000,
                lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.,
                area_thresh=6000., # only large lakes
                resolution='i') # Lambert conformal
          
# draw parallels and meridians.
# labels: [left, right, top, bottom]
map4.drawparallels(np.arange(-80.,81.,6.),  labels=[1,0,0,0], dashes=[1,1], linewidth=0.25, color='0.5')
map4.drawmeridians(np.arange(-180.,181.,15.),labels=[0,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')

# draw cooastlines and countries
map4.drawcoastlines(linewidth=0.3)
map4.drawmapboundary(fill_color=ocean_color, linewidth=0.3)
map4.drawcountries(color='black', linewidth=0.3)
map4.fillcontinents(color='white', lake_color=ocean_color)

coord_catch   = []
for ibasin_id,basin_id in enumerate(basin_ids):

    shapefilename =  "../data_in/data_obs/"+basin_id+"/shape.dat"
    icolor = cc[ibasin_id%len(cc)]

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
                sub.add_patch(Polygon(np.transpose(map4(coords[start:end,0],coords[start:end,1])), facecolor=icolor, edgecolor='black', linewidth=0.0,zorder = 300, alpha=0.8))

                xxmin = np.nanmin(coords[start:end,0])
                xxmax = np.nanmax(coords[start:end,0])
                yymin = np.nanmin(coords[start:end,1])
                yymax = np.nanmax(coords[start:end,1])

                if not(donolabel):
                    # annotate
                    xpt, ypt  = map4(np.mean(coords[start:end,0]), np.mean(coords[start:end,1]))   # center of shape
                    x2,  y2   = (1.1,0.95-ibasin_id*0.1)                                           # position of text
                    sub.annotate(basin_id,
                        xy=(xpt, ypt),   xycoords='data',
                        xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                        fontsize=8,
                        verticalalignment='center',horizontalalignment='left',
                        arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                        zorder=400
                        )

        print("Basin: ",basin_id)
        print("   --> lon range = [",xxmin,",",xxmax,"]")
        print("   --> lat range = [",yymin,",",yymax,"]")

    # shapefile doesnt exist only plot dot at location
    else:   
        xpt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[0]
        ypt = map4(dict_metadata[basin_id]["lon"],dict_metadata[basin_id]["lat"])[1]
        x2,  y2   = (1.1,0.95-ibasin_id*0.1)
        
        sub.plot(xpt, ypt,
                 linestyle='None', marker='o', markeredgecolor=icolor, markerfacecolor=icolor,
                 markersize=2.0, markeredgewidth=0.0)

        if not(donolabel):
            sub.annotate(basin_id,
                    xy=(xpt, ypt),   xycoords='data',
                    xytext=(x2, y2), textcoords='axes fraction', #textcoords='data',
                    fontsize=8,
                    verticalalignment='center',horizontalalignment='left',
                    arrowprops=dict(arrowstyle="->",relpos=(0.0,1.0),linewidth=0.6),
                    zorder=400
                    )
        print("Basin: ",basin_id)
        print("   --> lat/lon = [",dict_metadata[basin_id]["lat"],",",dict_metadata[basin_id]["lon"],"]")
        
    

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




    
