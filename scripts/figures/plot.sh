#!/bin/bash
#
# Produces plots for publication.

set -e
#
prog=$0
pprog=$(basename ${prog})
dprog=$(dirname ${prog})
isdir=${PWD}
pid=$$

# source  ~/projects/rpp-hwheater/julemai/xSSA-North-America/env-3.5/bin/activate

dofig1=0  	# Flowchart
dofig2=0  	# Plot calibration and validation results on a map
dofig3=0  	# Plots time-aggregated xSSA results for 9 processes on a map
dofig4=0        # Plots time-dependent sensitivity plots for selected basins (representative of XX clusters of Knoben climate index)
dofig5=0        # Plots xSSA-derived vs predicted STi for each process in calibration and validation setting for 100 trials
dofigS1=0        # Blended Model flowchart (same as S1 in Mai et al. (2020))
dofigS2=1        # Plots comparison with Markstrom's FAST (first-order) results regarding mean runoff vs. xSSA (first-order; Si) regarding mean Q


if [[ ${dofig1} -eq 1 ]] ; then
    echo ''
    echo 'Figure 1 in progress...'
    # --------------------------------------------------------------------
    # Flowchart
    # --------------------------------------------------------------------
    pdflatex figure_1/figure_1.tex  ${pipeit}
    pdfcrop figure_1.pdf
    mv figure_1-crop.pdf ../../figures/figure_1.pdf
    rm figure_1.pdf
    rm figure_1.log
    rm figure_1.aux
fi

if [[ ${dofig2} -eq 1 ]] ; then
    echo ''
    echo 'Figure 2 in progress...'
    # --------------------------------------------------------------------
    # plot calibration results of blended Raven on a map
    # --------------------------------------------------------------------
    # python figure_2.py -p figure_2.pdf
    # pdfcrop figure_2.pdf
    # mv figure_2-crop.pdf ../../figures/figure_2.pdf
    # rm figure_2.pdf

    python figure_2.py -g figure_2_
    #pdfcrop figure_2.pdf
    mv figure_2_0001.png ../../figures/figure_2.png
    #rm figure_2.pdf
fi

if [[ ${dofig3} -eq 1 ]] ; then
    echo ''
    echo 'Figure 3 in progress...'
    # --------------------------------------------------------------------
    # Plots time-aggregated xSSA results for 9 processes on a map
    # --------------------------------------------------------------------
    python figure_3.py -g figure_3_
    #pdfcrop figure_3.pdf
    mv figure_3_0001.png ../../figures/figure_3.png
    #rm figure_3.pdf
fi

if [[ ${dofig4} -eq 1 ]] ; then
    echo ''
    echo 'Figure 4 in progress...'
    # --------------------------------------------------------------------
    # Plots time-dependent sensitivity plots for selected basins (representative of XX clusters of Knoben climate index)
    # --------------------------------------------------------------------
    # python figure_4.py -p figure_4.pdf
    # pdfcrop figure_4.pdf
    # mv figure_4-crop.pdf ../../figures/figure_4.pdf
    # rm figure_4.pdf

    python figure_4.py -g figure_4_
    #pdfcrop figure_4.pdf
    mv figure_4_0001.png ../../figures/figure_4.png
    #rm figure_4.pdf
fi

if [[ ${dofig5} -eq 1 ]] ; then
    echo ''
    echo 'Figure 5 in progress...'
    # --------------------------------------------------------------------
    # Plots xSSA-derived vs predicted STi for each process in calibration and validation setting for 100 trials
    # --------------------------------------------------------------------
    python figure_5.py -g figure_5_
    #pdfcrop figure_5.pdf
    mv figure_5_0001.png ../../figures/figure_5.png
    #rm figure_5.pdf
fi

if [[ ${dofigS1} -eq 1 ]] ; then
    echo ''
    echo 'Figure S1 in progress...'
    # --------------------------------------------------------------------
    # Flowchart
    # --------------------------------------------------------------------
    pdflatex figure_S1/figure_S1.tex  ${pipeit}
    pdfcrop figure_S1.pdf
    mv figure_S1-crop.pdf ../../figures/figure_S1.pdf
    rm figure_S1.pdf
    rm figure_S1.log
    rm figure_S1.aux
fi

if [[ ${dofigS2} -eq 1 ]] ; then
    echo ''
    echo 'Figure S2 in progress...'
    # --------------------------------------------------------------------
    # Compare xSSA results with Markstrom results
    # --------------------------------------------------------------------
    python figure_S2.py -g figure_S2_
    #pdfcrop figure_S2.pdf
    mv figure_S2_0001.png ../../figures/figure_S2.png
    #rm figure_S2.pdf
fi
