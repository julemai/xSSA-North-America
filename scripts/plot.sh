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

source  ~/projects/rpp-hwheater/julemai/xSSA-North-America/env-3.5/bin/activate

dofig1=0  # Plot sensitivities of all 9 processes on a map with basin shapes
dofig2=1  # Plot climate indexes map

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

if [[ ${dofig2} -eq 1 ]] ; then
    # plot all basins
    basins=$( \ls -d ~/projects/rpp-hwheater/julemai/xSSA-North-America/data_out/* | rev | cut -d '/' -f 1 | rev )
    python figure_2.py -g figure_2_ -i "${basins}"
    mv figure_2_0001.png ../figures/figure_2.png
fi
