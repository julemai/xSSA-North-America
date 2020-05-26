#!/bin/bash
#
# Produces plots for presentation.
#
# submit with:
#       sbatch plot.sh   

#SBATCH --mem-per-cpu=10G                          # memory; default unit is megabytes
#SBATCH --output=/dev/null
#SBATCH --time=03:00:00
#SBATCH --account=rpp-hwheater 
#SBATCH --mail-user=juliane.mai@uwaterloo.ca
#SBATCH --mail-type=FAIL
#SBATCH --job-name=plotting

set -e
#
prog=$0
pprog=$(basename ${prog})
dprog=$(dirname ${prog})
isdir=${PWD}
pid=$$

# source  ~/projects/rpp-hwheater/julemai/xSSA-North-America/env-3.5/bin/activate



doclimindex=0   #                Derives Knoben climate indexes for all basins
dofig1=0  	# 		 Plot sensitivities of all 9 processes on a map with basin shapes
dofig3=0  	# 		 Plot correlations of climate indexes and properties with sensitivities of processes
dofig4=0  	# 		 Plot map of climate indexes of all basins
dofig5=1  	# On my Mac:     Plot comparison with mHM, Hype, and VIC
dofig6=1  	# On my Mac:     Plot comparison with PRMS



if [[ ${doclimindex} -eq 1 ]] ; then
    # calculates climate indexes of all basins
    #
    # all basins
    basins=$( \ls -d ~/projects/rpp-hwheater/julemai/xSSA-North-America/data_out/* | rev | cut -d '/' -f 1 | rev )
    python calculate_climate_indexes_knoben.py -i "${basins}" -o "../data_in/basin_metadata/basin_climate_index_knoben_snow-raven.txt"  -s "raven"
    python calculate_climate_indexes_knoben.py -i "${basins}" -o "../data_in/basin_metadata/basin_climate_index_knoben_snow-knoben.txt" -s "knoben"
fi

if [[ ${dofig1} -eq 1 ]] ; then
    # plot only basins already analysed
    basins=$( \ls ~/projects/rpp-hwheater/julemai/xSSA-North-America/data_out/*/results_nsets1000.pkl | rev | cut -d '/' -f 2 | rev )
    python figure_1.py -g figure_1_ -n -i "${basins}"
    mv figure_1_0001.png ../figures/figure_1_Infiltration.png
    mv figure_1_0002.png ../figures/figure_1_Quickflow.png
    mv figure_1_0003.png ../figures/figure_1_Evaporation.png
    mv figure_1_0004.png ../figures/figure_1_Baseflow.png
    mv figure_1_0005.png ../figures/figure_1_SnowBalance.png
    mv figure_1_0006.png ../figures/figure_1_ConvolutionSurfaceRunoff.png
    mv figure_1_0007.png ../figures/figure_1_ConvolutionDelayedRunoff.png
    mv figure_1_0008.png ../figures/figure_1_PotentialMelt.png
    mv figure_1_0009.png ../figures/figure_1_Percolation.png
fi

if [[ ${dofig3} -eq 1 ]] ; then
    # plot correlations of climate indexes and properties with sensitivities of processes
    #
    # plot only basins already analysed
    #basins=$( \ls ~/projects/rpp-hwheater/julemai/xSSA-North-America/data_out/0161*/results_nsets1000.pkl | rev | cut -d '/' -f 2 | rev )
    basins=$( \ls ~/projects/rpp-hwheater/julemai/xSSA-North-America/data_out/*/results_nsets1000.pkl | rev | cut -d '/' -f 2 | rev )
    python figure_3.py -p figure_3.pdf -i "${basins}" -a "../data_in/basin_metadata/basin_climate_index_knoben_snow-raven.txt"
    pdfcrop figure_3.pdf
    mv figure_3-crop.pdf ../figures/figure_3.pdf
    rm figure_3.pdf
fi

if [[ ${dofig4} -eq 1 ]] ; then
    # plot map of climate indexes of all basins
    #
    # all basins
    basins=$( \ls -d ~/projects/rpp-hwheater/julemai/xSSA-North-America/data_out/* | rev | cut -d '/' -f 1 | rev )
    python figure_4.py -g figure_4_ -i "${basins}" -a "../data_in/basin_metadata/basin_climate_index_knoben_snow-raven.txt"
    mv figure_4_0001.png ../figures/figure_4.png
fi


if [[ ${dofig5} -eq 1 ]] ; then
    #
    # plot scatterplots and CDFs of model performance compared to HYPE, VIC, and mHM
    #
    python figure_5.py -p figure_5.pdf
    pdfcrop figure_5.df
    mv figure_5-crop.pdf ../figures/figure_5.pdf
    rm figure_5.pdf
fi


if [[ ${dofig6} -eq 1 ]] ; then
    #
    # plot comparison of sum(si_m for all paras) with PRMS results in Markstrom et al. (2016)
    #
    python figure_6.py -p figure_6.pdf
    pdfcrop figure_6.df
    mv figure_6-crop.pdf ../figures/figure_6.pdf
    rm figure_6.pdf
fi
