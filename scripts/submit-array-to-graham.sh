#!/bin/bash

# submit with:
#       sbatch submit-array-to-graham.sh   

#SBATCH --mem-per-cpu=50G                          # memory; default unit is megabytes
#SBATCH --output=/dev/null
#SBATCH --time=3-00:00:00
#SBATCH --account=rpp-hwheater 
#SBATCH --mail-user=juliane.mai@uwaterloo.ca
#SBATCH --mail-type=FAIL
#SBATCH --job-name=sa-canopex
#SBATCH --array=1-601

# job-id  :: ${SLURM_ARRAY_JOB_ID}
# task-id :: ${SLURM_ARRAY_TASK_ID}

# ntasks=645 # (9 basins per task --> time=6-18:00:00 --> 5798 basins total)
ntasks=601   # (5 basins per task --> time=6-18:00:00 --> remaining 3002 basins)
nsets=1000

# nsets =   10 -->      7.0min/basin  --> 6.8h for each of 100 tasks                 #SBATCH --mem-per-cpu=3G
# nsets =    2 -->      1.6min/basin  --> 1h 50min for each of 100 tasks             #SBATCH --mem-per-cpu=3G
# nsets = 1000 -->    18:00:00/basin  --> 6-18:00:00 for 645 tasks (9 basins each)   #SBATCH --mem-per-cpu=50G
# nsets = 1000 -->    18:00:00/basin  --> 5-00:00:00 for 601 tasks (5 basins each)   #SBATCH --mem-per-cpu=50G

# change to right dir
cd /home/julemai/projects/rpp-hwheater/julemai/xSSA-North-America/scripts

# set Python env
source ../env-3.5/bin/activate

folders=$(  \ls -d ../data_in/data_obs/*/ | cut -d '/' -f 4 | sort )

# select only folders that don't have yet a results file
# >>>>> if run all basins again then uncomment following line
# folders=$( for ff in ${folders} ; do if [ ! -e ../data_out/${ff}/results_nsets${nsets}.pkl ] ; then echo ${ff} ; fi ; done )
folders=$( for ff in ${folders} ; do if [ ! -e ../data_out/${ff}/results_nsets${nsets}.token ] ; then echo ${ff} ; fi ; done )

nfolders=$( for ff in ${folders} ; do echo ${ff} ; done | wc -l )         # 5797 (or whatever is not done yet)
step=$( echo $(( ${nfolders}/${ntasks} + 1 )) )                           #   58 (or whatever is not done yet)
step=$( echo $( echo "${nfolders}/${ntasks}" | bc -l ) | cut -d '.' -f 1 )
step=$(( ${step} + 1 ))

start=$(($((${SLURM_ARRAY_TASK_ID} - 1)) * ${step}))
end=$((${start} + ${step}))

# last chunk might be a bit shorter
if ((${end} > ${nfolders})) ; then
   step=$(( ${step} - $(( ${end} - ${nfolders} )) ))    
   end=${nfolders}
fi

for ff in ${folders} ; do echo ${ff} ; done | sort
# echo 'nfolders = '${nfolders}
# echo 'start    = '${start}
# echo 'end      = '${end}
# echo 'step     = '${step}

if [[ ${step} -gt 0 ]] ; then

    # list of basins from start to end that do not have results yet
    basins_to_process=$( for ff in ${folders} ; do echo ${ff} ; done | sort | head -${end} | tail -${step} )
    
    # echo ${basins_to_process}

    # # ---------------------
    # # this can be used to run multiple basins in one shot
    # python raven_sa-usgs-canopex.py -i "${basins_to_process}" -n ${nsets}
    # # ---------------------

    # ---------------------
    # this is running one basin after the other
    for bb in ${basins_to_process} ; do

	# actual analysis
	python raven_sa-usgs-canopex.py -i ${bb} -n ${nsets} -t ${SLURM_TMPDIR} -o nc

	# # extract only sensitivity results from pickle file
	# pickle_all="../data_out/${bb}/results_nsets${nsets}.pkl"
	# pickle_si="../data_out/${bb}/sensitivity_nsets${nsets}.pkl"
	# python extract_si_results_from_pkl.py -i ${pickle_all} -o ${pickle_si}
    
	# plot results
	python figure_2.py -t pdf -p ../data_out/${bb}/${bb} -n ${nsets} -i ../data_out/${bb}/results_nsets${nsets}.nc -o nc 
	pdfcrop ../data_out/${bb}/${bb}.pdf
	mv ../data_out/${bb}/${bb}-crop.pdf ../data_out/${bb}/${bb}.pdf

    done
    # ---------------------
    
else
    echo 'Nothing to do!'
fi

# 2 sets
# job 21811270   :: run with 90min --> time out --> about 10/58 basins per tasks not finished  (4793 of 5797 basins)
# job 21815339   :: run with 30min --> only last 15 basins each                                (4808 of 5797 basins)
# job 21817272   :: redo only basins that are not finished yet 90min -->                       (5786 of 5797 basins)



# 06671000  --> ERROR : CGauge::Initialize: Raven cannot have blank data in daily temperature time series (Gauge: Gauge1, n=5537)
#           --> derived avg temp is nodata --> line 19787: -4.1490000 1.68000000 --> max temp changed to 1.681
# 08DD001   --> ERROR : CGauge::Initialize: excessively small or large average temperature (<-60C or >60C) reported at gauge
#           --> 119.46062622 --> 19.46062622



# 1000 sets, 5798 basins
# job 25208219

# 1000 sets, 3002 basins
# job 28161689
