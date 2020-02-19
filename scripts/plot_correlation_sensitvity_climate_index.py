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
#     run plot_correlation_sensitvity_climate_index.py -i "01010000 01010500 01013500 01014000 01015800 01018500 01019000 01020000 01021000 01021500 01030500 01031500 01033000 01033500 01034000 01037000 01041000 01042500 01043500 01045000 01049205 01049265 01049500 01053500 01053600"

#     run plot_correlation_sensitvity_climate_index.py -g plot_correlation_sensitvity_climate_index_ -i "01010000 01010500 01013500 01014000 01015800 01018500 01019000 01020000 01021000 01021500 01030500 01031500 01033000 01033500 01034000 01037000 01041000 01042500 01043500 01045000 01049205 01049265 01049500 01053500 01053600 01054000 01064140 01064500 01065000 01065500 01066000 01085000 01085500 01088000 01091500 01108000 01111212 01112500 01113895 01116500 01118500 01127000 01129200 01129500 01130000 01131500 01144000 01144500 01151500 01152500 01154500 01161000 01166500 01168151 01168500 01170000 01177000 01183500 01187000 01187980 01188090 01199000 01200000 01200500 01321000 01325000 01327750 01329490 01329500 01343060 01344000 01346000 01347000 01348000 01351500 01357500 01358000 01361000 01364500 01375000 01381900 01388500 01389500 01389800 01400500 01417500 01420500 01420980 01421000 01422700 01427405 01427510 01428500 01438500 01442500 01446500 01446700 01447800 01449000 01463500 01465500 01470500 01471000 01471510 01481000 01481500 01497500 01505000 01505810 01507000 01509000 01510500 01518000 01518700 01520000 01520500 01525500 01526500 01529500 01529950 01531000 01531500 01532000 01533400 01534000 01536000 01541200 01541303 01542500 01543000 01543500 01547500 01548000 01548005 01548500 01549700 01555000 01556000 01558000 01568000 01570000 01570500 01571500 01573000 01575585 01576000 01576500 01576754 01578310 01595800 01598500 01600000 01601500 01603000 01608070 01608500 01610000 01611500 01613000 01619500 01622000 01625000 01627500 01628500 01634000 01636500 01638500 01657500 01662000 01663500 01671025 01672500 01673000 01AM001 01AN002 01AP002 01BG005 01BG009 01BH001 01BH002 01BH005 01BH007 01BJ007 01BO002 01BO003 01BP001 01BQ001 01BU002 01DG006 01EC001 01ED005 01EG002 01EN002 01EO001 01FB001 01FB003 02011800 02012500 02013100 02016000 02018000 02019500 02021500 02023000 02024000 02029000 02030500 02032500 02032515 02034000 02034500 02041500 02041650 02042500 02044500 02045500 02051500 02052000 02053200 02054500 02054510 02054530 02055000 02056000 02057500 02058400 02058500 02060500 02061500 02062500 02065200 02066000 02070500 02071000 02072000 02072500 02073000 02075500 02077000 02077303 02082506 02082585 02083000 02087570 02088500 02089000 02089500 02091814 02098200 02100500 02102000 02102500 02103000 02103500 02105500 02105769 02106500 02113500 02113850 02115360 02116500 02118000 02119000 02122500 02123500 02135200 02136000 0213903612 02140991 02142500 02145000 02146000 02146800 021556525 02156000 02156500 02160105 02160390 02162500 02163000 02163001 02163500 02165000 02169000 02169500 02172002 02172500 02173000 02175000 02175500 02176000 02191300 02191743 02192000 02192500 02193500 02196484 02197320 02198000 02198920 02200120 02203000 02204500 02207220 02207335 02207500 02213000 02215000 02215500 02215900 02217475 02219500 02220900 02223000 02223056 02223248 02226000 02226100 02226160 02226362 02226500 02231253 02231289 02231600 02234500 02236000 02236125 02237293 02238000 02243000 02244040 02244440 02245050 02246025 02272500 02294650 02294655 02294775 02294898 02298830 02301500 02301719 02303000 02303330 02312300 02312500 02312600 02312700 02312720 02319500 02319800 02320000 02320500 02321500 02326512 02326900 02327355 02335450 02335500 02335815 02336000 02336490 02344000 02344500 02346180 02352500 02353000 02353265 02353500 02354500 02354800 02355350 02355662 02358700 02358754 02358789 02359000 02359170 02360500 02365200 02365500 02366000 02366500 02367800 02372000 02372250 02372422 02372430 02372500 02373000 02373500 02373800 02374250 02374660 02374700 02375500 02376033 02384500 02387000 02387500 02388350 02388500 02394000 02395000 02395980 02404400 02411000 02411930 02412000 02413300 02414500 02421000 02422000 02422500 02423000 02423425 02433000 02437100 02437500 02438000 02439400 02440500 02450180 02450500 02452000 02453500 02455000 02464000 02464500 02465000 02466000 02467000 02469761 02472000 02472500 02472850 02473000 02473500 02477000 02477500 02477990 02478500 02479000 02479500 02479560 02481500 02481510 02481880 02483500 02484000 02484630 02485498 02485500 02489000 02489500 02490500 02491500 02492000 02BD003 02EA005 02EC018 02ED101 02FC016 02GB007 02GC002 02GC010 02GD004 02JB004 02LD001 02LD005 02LG005 02LH004 02MC001 02OE032 02OG026 02OJ024 02PA007 02PJ007 02PJ030 02PK005 02PL005 02PL007 02QA002 02RC011 02RD002 02RD003 02RH047 02RH049 02UC002 02VA001 02VB004 02YA001 02YC001 02YG001 02YJ001 02YN002 02YO008 02YQ001 02YR003 02YS001 03010500 03010800 03011020 03012550 03012600 03015500 03016000 03017500 03020000 03020500 03025500 03028500 03029500 03036500 03039000 03040000 03041029 03041500 03044000 03047000 03048500 03053500 03054500 03056000 03057000 03058975 03059000 03061000 03062000 03065000 03082500 03083500 03085000 03085500 03086000 03090500 03091500 03114500 03115400 03117000 03131500 03133500 03135000 03142000 03143500 03144500 03146500 03147000 03152000 03153500 03154000 03155000 03155220 03159530 03160000 03161000 03162500 03164000 03184000 03184500 03185400 03185500 03187000 03192000 03193000 03194700 03195000 03195500 03199400 03199700 03200500 03201000 03201902 03202400 03204500 03205470 03207000 03207020 03207500 03207800 03209200 03209300 03209500 03209800 03211500 03212980 03214000 03214500 03214900 03219500 03221000 03225500 03226500 03226800 03230700 03230800 03230900 03231000 03231500 03246200 03246500 03247050 03247500 03248000 03249500 03250500 03251200 03252000 03252500 03253500 03254520 03261500 03309000 03310300 03310500 03319000 03320500 03321060 03322100 03325000 03326000 03326500 03327500 03328000 03351000 03362500 03402900 03403000 03403500 03403910 03404000 03404500 03405000 03406500 03408500 03518000 03565000 03571000 03574500 03575000 03AD001 03BA003 03BB002 03BC002 03BD002 03BF001 03CC001 03CE001 03DC002 03DD002 03DD003 03ED004 03FA003 03FC007 03MB002 03MD001 03NF001 03OC003 03PB002 04024000 04024430 04027000 04027500 04029990 04034500 04035000 04039500 04041500 04045500 04056500 04057000 04058100 04059000 04059500 04062011 04062230 04062270 04062300 04062400 04062500 04065106 04065500 04065722 04066000 04066003 04067500 04067651 04067958 04087170 04121500 04122000 04122200 04122500 04130500 04132000 04133500 04133501 04135000 04135700 04146063 04147500 04148500 04149000 04150000 04150500 04150800 04151500 04152500 04154000 04154500 04155000 04159500 04160000 04161820 04164000 04164500 04174500 04174800 04176000 04176500 04178000 04181500 04182000 04183000 04183500 04184500 04189000 04191500 04192500 04193500 04195500 04197000 04198000 04199000 04199500 04200500 04208000 04208504 04209000 04212000 04212100 04213500 04221500 04223000 04227000 04232482 04232730 04235271 04235440 04235500 04247000 04247055 04249000 04250200 04252500 04262000 04262500 04263000 04265000 04265432 04268700 04269000 04273500 04275500 04276500 04293500 04AC007 04AD002 04CA002 04CA003 04CB001 04CC001 04CD002 04CE002 04DA001 04DB001 04DC001 04DC002 04EA001 04FC001 04GA002 04GB004 04NB001 05017500 05030500 05051300 05051522 05051700 05052000 05052500 05053000 05054000 05055400 05056000 05057200 05059700 05060000 05062000 05062200 05062500 05064000 05065500 05074500 05078000 05078230 05078500 05082625 05084000 05087500 05089000 05090000 05092000 05112000 05116150 05116500 05123400 05123510 05124500 05125550 05127000 05127500 05129000 05211000 05212700 05217000 05220500 05227500 05247500 05267000 05270500 05270700 05275500 05278500 05278930 05279000 05286000 05288500 05290000 05291000 05292000 05292500 05292704 05294000 05300000 05304000 05304500 05311000 05313500 05315000 05316580 05316770 05317000 05319500 05320000 05320500 05325000 05330000 05330920 05331000 05331580 05332500 05333500 05336000 05336700 05338500 05339500 05340050 05340500 05341500 05341752 05344500 05353800 05355200 05356000 05356500 05357500 05358500 05359500 05360000 05360500 05362000 05364000 05365500 05366500 05367426 05367500 05368000 05369000 05369500 05372995 05373000 05374000 05374500 05374900 05379500 05381000 053813595 05382000 05383000 05384000 05385000 05385500 05387440 05393000 05395000 05396000 05397110 05400800 05402000 05403000 05403500 05404000 05408000 05410000 05410490 05410500 05411850 05416900 05417000 05418400 05418450 05418500 05421000 05421740 05422000 05427085 05427530 05427570 05428500 05430500 05431486 05431500 05432500 05433000 05453100 05453520 05454500 05455100 05455500 05455700 05457700 05459500 05462000 05463000 05463500 05464000 05464220 05464315 05464500 05465500 05466500 05469000 05470000 05470500 05471000 05471200 05471500 05472500 05473400 05474000 05476000 05476500 05476750 05478000 05479000 05480000 05480500 05481000 05481300 05481500 05481650 05481950 05482000 05482135 05482300 05482500 05483450 05483600 05484000 05484500 05484650 05484900 05485500 05486490 05487470 05487980 05488000 05488110 05488500 05489500 05490500 05490600 05495000 05495500 05496000 05497000 05498000 05498150 05498700 05500000 05501000 05502300 05502500 05504800 05504900 05505000 05506350 05506500 05506800 05507000 05507500 05508805 05514500 05515500 05516500 05517000 05517500 05517530 05520500 05522500 05524500 05525000 05525500 05529000 05532500 05536290 05536995 05540500 05546500 05550000 05550001 05551000 05551540 05555500 05556500 05558300 05563500 05567500 05570000 05570910 05571000 05576000 05576500 05578500 05579500 05580000 05583000 05584500 05585000 05585500 05585830 05590950 05591200 05592000 05592100 05593000 05594000 05594100 05594800 05595000 05596000 05RA002 05RB003 05RC001 05RD008 05RE001 05SA002 05TE002 05TF002 05TG002 05TG003 05UA003 05UF004 05UG001 06011000 06012500 06013500 06014000 06014500 06015400 06102000 06208500 06306300 06308500 06334630 06347000 06436800 06467600 06600100 06651500 06690500 06710000 06756100 06791800 06862850 06873460 06920500 06GA001 07260500 07311800 07363400 07BB003 07EB002 08022500 08079575 08111010 08143600 08164500 08177000 08179000 08317400 08401900 08GD008 08LB047 08NA002 08NA006 08NH130 09058030 09288100 09304600 09332100 09404900 09508500 10016900 10028500 10308200 11128000 11152050 11333500 11397500 11433500 12452800 13037500 13152500 13302000 13309220 13316500 13317000 14026000 14054000 14120000 14189000 14238000 14315700"

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

    basin_ids = None
    pngbase   = ''
    pdffile   = ''
    usetex    = False
    
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

    args      = parser.parse_args()
    basin_ids = args.basin_ids
    pngbase   = args.pngbase
    pdffile   = args.pdffile
    usetex    = args.usetex

    # convert basin_ids to list 
    basin_ids = basin_ids.strip()
    basin_ids = " ".join(basin_ids.split())
    basin_ids = basin_ids.split()
    # print('basin_ids: ',basin_ids)

    del parser, args
# Comment|Uncomment - End


    # -------------------------------------------------------------------------
    # Function definition - if function
    #    

    # Basin ID need to be specified
    if basin_ids is None:
        raise ValueError('Basin ID (option -i) needs to given. Basin ID needs to correspond to ID of a CAMELS basin.')

    # -----------------------
    # add subolder scripts/lib to search path
    # -----------------------
    import sys
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path+'/lib')

    import numpy    as np
    import datetime as datetime
    import pandas   as pd
    import time

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

    # ----------------------------------
    # read climate indicators
    # ----------------------------------
    climate_indexes = {}
    indicators = ['aridity', 'seasonality', 'precip_as_snow']

    file_climate_indexes = '../data_in/basin_metadata/basin_climate_index_knoben.txt'
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
    processes = ['Infiltration','Quickflow','Evaporation','Baseflow','Snow Balance', 'Convolution (Surface Runoff)', 'Convolution (Delayed Runoff)', 'Potential Melt', 'Percolation']
    sobol_indexes = {}
    for ibasin_id,basin_id in enumerate(basin_ids):

        import pickle
        picklefile = "../data_out/"+basin_id+"/sensitivity_nsets1000.pkl"
        setup         = pickle.load( open( picklefile, "rb" ) )
        # sobol_indexes.append( setup['sobol_indexes']['processes']['wsti']['Q'] )
        sobol_indexes[basin_id] = np.array(setup['sobol_indexes']['processes']['wsti']['Q'])

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
nrow        = len(processes)    # # of rows of subplots per figure
ncol        = len(indicators)   # # of columns of subplots per figure
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



# sort basins starting with largest
areas = []
for ibasin_id,basin_id in enumerate(basin_ids):
    areas.append(dict_metadata[basin_id]["area_km2"])
areas = np.array(areas)
idx_areas = np.argsort(areas)[::-1]
basin_ids = np.array(basin_ids)[idx_areas]




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

for iprocess, process in enumerate(processes):
    for iindicator, indicator in enumerate(indicators):

        iplot += 1

        sub = fig.add_axes(position(nrow,ncol,iplot,hspace=hspace,vspace=vspace))

        wsti  = [ sobol_indexes[basin_id][iprocess]    for basin_id in basin_ids ]
        indic = [ climate_indexes[basin_id][indicator] for basin_id in basin_ids ]
        mark1 = sub.scatter(indic,wsti)
        plt.setp(mark1, linestyle='None', color=mcol[idischarge], linewidth=0.0,
                     marker='o', markeredgecolor=mcol[idischarge], markerfacecolor='None',
                     markersize=msize, markeredgewidth=mwidth,
                     label=str2tex(head_sink[idischarge],usetex=usetex))
    

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

