#!/bin/bash

# submit with:
#       sbatch submit-to-graham.sh     

#SBATCH --account=rpp-hwheater                     # your group 
#SBATCH --mem-per-cpu=50G                          # memory; default unit is megabytes
#SBATCH --mail-user=juliane.mai@uwaterloo.ca       # email address for notifications
#SBATCH --mail-type=FAIL                           # email send only in case of failure
#SBATCH --time=0-18:00                             # time (DD-HH:MM);
#SBATCH --job-name=sa-canopex                      # name of job in queque
#SBATCH --array=1-100

# job ID: 21549268
# nsets =  10
# requested: 10min/basin     2GB
# used:       7min/basin   120MB

# job ID: 21884779   -> 10G
# nsets = 1000 --> 15h/basin 
# requested:   15h/basin    10GB
# used:       OUT OF MEMORY

# job ID: 21902322   -> 20G
# nsets = 1000 --> 15h/basin 
# requested:   15h/basin    20GB
# used:       OUT_OF_MEMORY

# job ID: 21919220   -> 50G             (changed run_ID folder names to a, b, and c only; no iset)
# nsets = 1000 --> 15h/basin 
# requested:   15h/basin    50GB
# used:       

# job ID: 22595423   -> 50G             (changed run_ID folder names to a, b, and c only; no iset)
# nsets = 1000 --> 18h/basin 
# requested:   18h/basin    50GB
# used:

# job ID: 22792833   -> 50G             (changed run_ID folder names to a, b, and c only; no iset)
# nsets = 1000 --> 18h/basin 
# requested:   18h/basin    50GB
# used:



# job-id  :: ${SLURM_ARRAY_JOB_ID}
# task-id :: ${SLURM_ARRAY_TASK_ID}

nsets=1000

# change to right dir
cd /home/julemai/projects/rpp-hwheater/julemai/sa-usgs-canopex/scripts

# set Python env
source ../env-3.5/bin/activate

# set tasks to 10
# basins=(03MD001 02196484 02100500 02YA001 03518000 05387440 06306300 09404900 11433500 14189000)

# set tasks to 85
basins=(01049500 01BG009 02197320 02358789 02404400 02433000 02469761 02OG026 02PL005 03065000 03085000 03090500 03202400 03212980 03319000 03351000 03362500 03404500 03565000 03BF001 03MB002 04062011 04087170 04212000 04293500 05051300 05051522 05082625 05247500 05369000 05487980 05505000 05TG003 06102000 06208500 06308500 06334630 06347000 06436800 06467600 06600100 06651500 06690500 06710000 06756100 06791800 06862850 06873460 06920500 06GA001 07260500 07311800 07363400 07BB003 07EB002 08079575 08111010 08143600 08164500 08317400 08401900 08GD008 08NA002 08NH130 09058030 09288100 09304600 09332100 09508500 10016900 10028500 11128000 11152050 11333500 11397500 12452800 13037500 13152500 13302000 13309220 13316500 13317000 14054000 14238000 14315700)

# set tasks to 100
basins=(01085500 01389800 01449000 01451000 01533400 01536500 01589000 01632000 01670400 01BJ007 01DG006 02012500 02034000 02173030 02227270 02298202 02319500 02326512 02336490 02PC009 03025500 03209000 03230500 03364650 03403910 03409500 03426310 03531500 04183500 04229500 05074500 05244000 05330000 05341500 05360000 05374000 05440700 05508805 05LE004 06015500 06017000 06026500 06078200 06274500 06306250 06340500 06354882 06425780 06695000 06709000 06747500 06803486 06874000 06879900 06899000 06919000 06DC001 07031650 07140900 07262500 07288500 07299670 07327000 07343450 07MA003 08022500 08150000 08172000 08177000 08179000 08250500 08CC001 08HA001 08LB047 08NA006 09144200 09144250 09260000 09293500 09306878 09309000 09354500 09486300 09497980 09516500 10308200 10371500 11097500 11158600 11213000 11295300 11349000 11421000 11472500 12324590 13183000 14026000 14120000 14163150 14165500)

# set tasks to 16
# basins=(01BG009 02338000 03090500 03403500 04212100 05484900 06264000 06709000 06EA007 07251500 08086050 08LA008 09419507 11187000 12472500 14339000)

if [ ! -e ../data_out/${bb}/results_nsets${nsets}.pkl ] ; then
    
    bb=$( echo ${basins[$(( ${SLURM_ARRAY_TASK_ID} - 1 ))]} )
    python raven_sa-usgs-canopex.py -i ${bb} -n ${nsets}
    
    python figure_2.py -t pdf -p ../data_out/${bb}/${bb} -i ../data_out/${bb}/results_nsets${nsets}.pkl
    pdfcrop ../data_out/${bb}/${bb}.pdf
    mv ../data_out/${bb}/${bb}-crop.pdf ../data_out/${bb}/${bb}.pdf
    
fi

