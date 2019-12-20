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
#     python figure_3.py -t pdf -p figure_3 -i "14189000 03MD001 06306300 02YA001 11433500 05387440 09404900 02100500 03518000 02196484"
#     python figure_3.py -t pdf -p figure_3 -i "01049500 01085500 01389800 01449000 01533400 01BG009 01BJ007 02012500 02034000 02100500 02196484 02197320 02358789 02404400 02433000 02469761 02OG026 02PL005 02YA001 03065000 03085000 03090500 03202400 03212980 03319000 03351000 03362500 03404500 03518000 03565000 03BF001 03MB002 03MD001 04062011 04087170 04212000 04293500 05051300 05051522 05074500 05082625 05247500 05330000 05369000 05387440 05487980 05505000 05TG003 06102000 06208500 06306300 06308500 06334630 06347000 06436800 06467600 06600100 06651500 06690500 06710000 06756100 06791800 06862850 06873460 06920500 06GA001 07260500 07311800 07363400 07BB003 07EB002 08022500 08079575 08111010 08143600 08164500 08177000 08179000 08317400 08401900 08GD008 08LB047 08NA002 08NA006 08NH130 09058030 09288100 09304600 09332100 09404900 09508500 10016900 10028500 10308200 11128000 11152050 11333500 11397500 11433500 12452800 13037500 13152500 13302000 13309220 13316500 13317000 14026000 14054000 14120000 14189000 14238000 14315700"

#!/usr/bin/env python
from __future__ import print_function

"""

Plots results of RAVEN sensitivity analysis and correlates it to:
- temperature
- precipitation
- elevation
- slope


History
-------
Written,  JM, Nov 2019
"""

# -------------------------------------------------------------------------
# Command line arguments - if script
#

# Comment|Uncomment - Begin
if __name__ == '__main__':

    import argparse
    import numpy as np

    pngbase     = ''
    pdffile     = ''
    usetex      = False
    serif       = False
    nsets       = 1000          # number of Sobol sequences
    variable    = 'Q'           # model output variable
    basin_ids   = None
    
    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Plots results of RAVEN sensitivity analysis and correlates it to different indicators such as temeparture.''')
    parser.add_argument('-s', '--serif', action='store_true', default=serif, dest="serif",
                    help="Use serif font; default sans serif.")
    parser.add_argument('-g', '--pngbase', action='store',
                    default=pngbase, dest='pngbase', metavar='pngbase',
                    help='Name basis for png output files (default: open screen window).')
    parser.add_argument('-p', '--pdffile', action='store',
                    default=pdffile, dest='pdffile', metavar='pdffile',
                    help='Name of pdf output file (default: open screen window).')
    parser.add_argument('-u', '--usetex', action='store_true', default=usetex, dest="usetex",
                        help="Use LaTeX to render text in pdf, png and html.")
    parser.add_argument('-n', '--nsets', action='store',
                        default=nsets, dest='nsets', metavar='nsets',
                        help='Number of sensitivity samples (default: nsets=10).')
    parser.add_argument('-v', '--variable', action='store',
                        default=variable, dest='variable', metavar='variable',
                        help='Model output variable name. Must match key in dictionaries in Pickle. (default: variable="Q").')
    parser.add_argument('-i', '--basin_ids', action='store',
                    default=basin_ids, dest='basin_ids', metavar='basin_ids',
                    help='Basin ID of basins to plot. Mandatory. (default: None).')

    args     = parser.parse_args()
    pdffile = args.pdffile
    pngbase = args.pngbase
    serif    = args.serif
    usetex   = args.usetex
    nsets    = np.int(args.nsets)
    variable = args.variable
    basin_ids = args.basin_ids

    # convert basin_ids to list 
    basin_ids = basin_ids.strip()
    basin_ids = " ".join(basin_ids.split())
    basin_ids = basin_ids.split()
    # print('basin_ids: ',basin_ids)

    # Basin ID need to be specified
    if basin_ids is None:
        raise ValueError('Basin ID (option -i) needs to given. Basin ID needs to correspond to ID of a CANOPEX basin.')

    del parser, args
    # Comment|Uncomment - End

    # -----------------------
    # add subolder scripts/lib to search path
    # -----------------------
    import sys
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append('lib')

    import pickle
    import datetime
    import time
    import color                      # in lib/
    from autostring import astr       # in lib/
    from position   import position   # in lib/
    

    # ---------------------------------
    # Read Sobol' results
    # ---------------------------------
    print("Read Sobol' results ...")
    sobol_indexes = {}
    for ibasin in basin_ids:
        picklefile = "../data_out/"+ibasin+"/sensitivity_nsets"+str(nsets)+".pkl"
        setup = pickle.load( open( picklefile, "rb" ) )
        sobol_indexes[ibasin] = setup['sobol_indexes']

    # sobol_indexes[ibasin]['processes']['wsti'][variable]   --> wsti per process:          shape = (9,)
    # sobol_indexes[ibasin]['processes']['sti'][variable]    --> sti over time per process: shape = (7305,9)

    # ---------------------------------
    # Read forcing data
    # ---------------------------------
    print("Read forcings ...")
    forcings = {}
    for ibasin in basin_ids:
        forcing_file = "../data_in/data_obs/"+ibasin+"/model_basin_mean_forcing_Santa-Clara.rvt"
        start_date = datetime.datetime(1991,1,1)
        end_date   = datetime.datetime(2010,12,31)

        ff = open(forcing_file, "r")
        line = ff.readline().strip()
        line = ff.readline().strip()
        variables = ff.readline().strip()
        ff.close()

        variables = variables.split()[1:]

        # :MultiData 
        #  1950-01-01  00:00:00  1  22280 
        # :Parameters     TEMP_DAILY_MIN   TEMP_DAILY_MAX   PRECIP   
        # :Units          C                C                mm/d     
        #     -8.32681814       1.17227271       1.31227272 
        #    -17.77545435      -6.48409090       0.88181818

        headerlines = 4
        datastart = datetime.datetime(np.int(line.split()[0].split('-')[0]),       # year
                                        np.int(line.split()[0].split('-')[1]),   # month
                                        np.int(line.split()[0].split('-')[2]),   # day
                                        np.int(line.split()[1].split(':')[0]),   # hour
                                        np.int(line.split()[1].split(':')[1]))   # minute
        delta_t = np.float(line.split()[2]) # dt in [days]

        start_idx = np.int((start_date - datastart).days / delta_t)
        end_idx   = np.int((end_date - datastart).days / delta_t) + 1

        ff = open(forcing_file, "r")
        lines = ff.readlines()
        ff.close()

        lines = lines[start_idx+headerlines:end_idx+headerlines]
        lines = [ [ np.float(ill) for ill in ll.strip().split() ] for ll in lines ]
        lines =  np.array(lines)
        forcing = {}
        for iivar,ivar in enumerate(variables):
            forcing[ivar] = lines[:,iivar] 
        forcings[ibasin] = forcing
        
    # ---------------------------------
    # Read basin characteristics
    # ---------------------------------
    print("Read basin characteristics ...")
    basin_props = {}

    file_properties = '../data_in/basin_metadata/basin_physical_characteristics.txt'
    ff = open(file_properties, "r")
    lines = ff.readlines()
    ff.close()

    for ill,ll in enumerate(lines):
        if ill > 0:

            # basin_id; basin_name; lat; lon; area_km2; elevation_m; slope_deg; forest_frac 
            tmp = ll.strip().split(';')
            basin_id = str(tmp[0])

            if (basin_id in basin_ids):

                basin_prop = {}
                basin_prop['name']         = str(tmp[1].strip())
                basin_prop['lat_deg']      = np.float(tmp[2].strip())
                basin_prop['lon_deg']      = np.float(tmp[3].strip())
                basin_prop['area_km2']     = np.float(tmp[4].strip())
                basin_prop['elevation_m']  = np.float(tmp[5].strip())
                basin_prop['slope_deg']    = np.float(tmp[6].strip()) 
                basin_prop['forest_frac']  = np.float(tmp[7].strip())
                
                basin_props[basin_id] = basin_prop


    # gathered:
    #      basin_props.keys()                                                                                                                                                
    #      forcings.keys()                                                                                                                                                   
    #      sobol_indexes.keys()


    # ------------------------------------
    # Correlation of wSTi with basin characteristics
    basin_chars = ['lat_deg', 'lon_deg', 'area_km2', 'elevation_m', 'slope_deg', 'forest_frac']
    processes   = ['Infiltration','Quickflow','Evaporation','Baseflow','Snow Balance', 'Convolution\n(surface runoff)', 'Convolution\n(delayed runoff)', 'Potential melt', 'Percolation']
    

    correlation_characteristics = np.ones([len(basin_chars),len(processes)]) * -9999.0
    for iichar,ichar in enumerate(basin_chars):
        for iiproc,iproc in enumerate(processes):

            correlation_characteristics[iichar,iiproc] = np.corrcoef( np.array([ basin_props[ib][ichar] for ib in basin_ids ]),
                                                                      np.array([ sobol_indexes[ib]['processes']['wsti'][variable][iiproc] for ib in basin_ids ]) )[1,0]


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
    nrow        = 3           # # of rows of subplots per figure
    ncol        = 2           # # of columns of subplots per figure
    hspace      = 0.10        # x-space between subplots
    vspace      = 0.05        # y-space between subplots
    textsize    = 13          # standard text size
    dxabc       = 0.90        # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
    dyabc       = 0.05        # % of (max-min) shift up from lower x-axis for a,b,c,... labels

    lwidth      = 1.5         # linewidth
    elwidth     = 1.0         # errorbar line width
    alwidth     = 1.0         # axis line width
    msize       = 1.0         # marker size
    mwidth      = 1.0         # marker edge width
    mcol1       = color.colours('red')        # primary marker colour
    mcol2       = '0.0'                     # secondary
    mcol3       = (202/255.,0/255.,32/255.) # third
    mcols       = color.colours(['blue','red','darkgray','orange','darkblue','black'])
    lcol1       = color.colours('blue')   # primary line colour
    lcol2       = '0.0'
    lcol3       = '0.0'
    lcols       = mcols

    # Map
    delon   = 60.        # spacing between longitude labels
    delat   = 60.        # spacing between latitude labels
    xsize   = textsize   # axis label size
    ncolor  = 10         # # of colors in plot
    cbsize  = textsize   # colorbar label size

    # Legend
    llxbbox     = -0.01       # y-anchor legend bounding box
    llybbox     = 0.04        # y-anchor legend bounding box
    llrspace    = 0.          # spacing between rows in legend
    llcspace    = 1.0         # spacing between columns in legend
    llhtextpad  = 0.4         # the pad between the legend handle and text
    llhlength   = 1.5         # the length of the legend handles
    frameon     = False       # if True, draw a frame around the legend. If None, use rc

    # PNG
    dpi         = 300
    transparent = False
    bbox_inches = 'tight'
    pad_inches  = 0

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
            #mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            mpl.rc('font',**{'family':'serif','serif':['times']})
        mpl.rc('text.latex', unicode=True)
    elif (outtype == 'png'):
        mpl.use('Agg') # set directly after import matplotlib
        import matplotlib.pyplot as plt
        mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
        if usetex:
            mpl.rc('text', usetex=True)
        else:
            #mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            mpl.rc('font',**{'family':'serif','serif':['times']})
        mpl.rc('text.latex', unicode=True)
        mpl.rc('savefig', dpi=dpi, format='png')
    else:
        import matplotlib.pyplot as plt
        mpl.rc('figure', figsize=(4./5.*8.27,4./5.*11.69)) # a4 portrait
    mpl.rc('font', size=textsize)
    mpl.rc('lines', linewidth=lwidth, color='black')
    mpl.rc('axes', linewidth=alwidth, labelcolor='black')
    mpl.rc('path', simplify=False) # do not remove





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
    # Colors
    # -------------------------------------------------------------------------
    infil_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[20]
    quick_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[55]
    evapo_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[80]
    basef_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[105]
    snowb_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[130]
    convs_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[155]
    convd_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[180]
    potme_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[205]
    perco_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[230]
    soilm_color = (0.7,0.7,0.7) #jams.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[255]


    # -------------------------------------------------------------------------
    # Plot
    #
    t1 = time.time()
    
    if (outtype == 'pdf'):
        print('Plot PDF ', pdffile)
        pdf_pages = PdfPages(pdffile)
    elif (outtype == 'png'):
        print('Plot PNG ', pngbase)
    else:
        print('Plot X')
    # figsize = mpl.rcParams['figure.figsize']

    
    ifig = 0

    # -------------------------------------------------
    # Correlation between basin properties and process sensitivities
    # -------------------------------------------------
    
    ifig += 1
    iplot = 0
    print('Plot - Fig ', ifig)
    fig = plt.figure(ifig)

    # -----------------------
    # plot
    # -----------------------

    # -------------
    # Correlation matrix
    # -------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    cc   = color.get_brewer('GrayWhiteGray', rgb=True)
    cc   = cc[2:]
    cmap = mpl.colors.ListedColormap(cc)

    xx = np.array([ np.arange(len(processes)+1) for ichar in range(len(basin_chars)+1) ])
    yy = np.array([ np.ones(len(processes)+1)*ichar for ichar in range(len(basin_chars)+1) ]) 
    sub.pcolor(xx,yy, correlation_characteristics, cmap=cmap)

    sub.set_xticks(np.arange(len(processes)+1))
    sub.set_xticklabels(['' for ii in np.arange(len(processes)+1)])
    sub.set_xticks(np.arange(len(processes))+0.5, minor=True)
    sub.set_xticklabels(processes, minor=True)

    sub.set_yticks(np.arange(len(basin_chars)+1))
    sub.set_yticklabels(['' for ii in np.arange(len(basin_chars)+1)])
    sub.set_yticks(np.arange(len(basin_chars))+0.5, minor=True)
    sub.set_yticklabels(basin_chars, minor=True)
    
    sub.tick_params(which='minor',length=0)   # dont show minor ticks; only labels
    plt.setp(sub.xaxis.get_minorticklabels(), rotation=90, fontsize='x-small')
    plt.setp(sub.yaxis.get_minorticklabels(), rotation= 0, fontsize='x-small')

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)
      
      

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
    str = '  Time plot [m]: '+astr((t2-t1)/60.,1) if (t2-t1)>60. else '  Time plot [s]: '+astr(t2-t1,0)
    print(str)


