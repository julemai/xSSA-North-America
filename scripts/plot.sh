#!/bin/bash
#
# Produces plots for presentation.
#
set -e
#
prog=$0
pprog=$(basename ${prog})
dprog=$(dirname ${prog})
isdir=${PWD}
pid=$$

dofig1=1  # Plot sensitivities of all 9 processes on a map with basin shapes

if [[ ${dofig1} -eq 1 ]] ; then
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
