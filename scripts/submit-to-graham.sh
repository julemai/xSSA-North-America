#!/bin/bash

# submit with:
#       sbatch submit-to-graham.sh     

#SBATCH --account=rpp-kshook                       # your group 
#SBATCH --mem-per-cpu=10G                          # memory; default unit is megabytes
#SBATCH --mail-user=juliane.mai@uwaterloo.ca       # email address for notifications
#SBATCH --mail-type=FAIL                           # email send only in case of failure
#SBATCH --time=0-22:00:00                          # time (DD-HH:MM:SS);
#SBATCH --job-name=sa-canopex                      # name of job in queque
#SBATCH --array=1-13

# -------------------
# testing NetCDF
# ------------------
# job ID: 30573974 - 10GB - 22hr
# used: max 4.2GB, ~9h
# ---> all crashed with permission denied error in tmp-dir

# job ID: 30593027 - 10GB - 22hr 


# job-id  :: ${SLURM_ARRAY_JOB_ID}
# task-id :: ${SLURM_ARRAY_TASK_ID}

nsets=1000

# change to right dir
cd /home/julemai/projects/rpp-hwheater/julemai/xSSA-North-America/scripts

# set Python env
source ../env-3.5/bin/activate

# set tasks to 13
basins=(08KC001 01608500 01643000 01668000 03054500 03179000 03364000 03451500 05455500 07186000 07378500 08167500 08172000)

if [ ! -e ../data_out/${bb}/results_nsets${nsets}.token ] ; then

    # actual analysis
    bb=$( echo ${basins[$(( ${SLURM_ARRAY_TASK_ID} - 1 ))]} )
    python raven_sa-usgs-canopex.py -i ${bb} -n ${nsets} -t ${SLURM_TMPDIR} -o nc

    # # extract only sensitivity results from pickle file
    # pickle_all="../data_out/${bb}/results_nsets${nsets}.pkl"
    # pickle_si="../data_out/${bb}/sensitivity_nsets${nsets}.pkl"
    # python extract_si_results_from_pkl.py -i ${pickle_all} -o ${pickle_si}
    
    # plot results
    python figure_2.py -t pdf -p ../data_out/${bb}/${bb} -n ${nsets} -i ../data_out/${bb}/results_nsets${nsets}.nc -o nc
    pdfcrop ../data_out/${bb}/${bb}.pdf
    mv ../data_out/${bb}/${bb}-crop.pdf ../data_out/${bb}/${bb}.pdf
    
fi

