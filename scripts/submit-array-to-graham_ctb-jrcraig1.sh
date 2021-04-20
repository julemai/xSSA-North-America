#!/bin/bash

# submit with:
#       sbatch submit-array-to-graham_ctb-jrcraig1.sh   

#SBATCH --mem-per-cpu=7G                          # memory; default unit is megabytes
#SBATCH --output=/dev/null
#SBATCH --time=1-02:00:00
#SBATCH --account=ctb-jrcraig1
#SBATCH --mail-user=juliane.mai@uwaterloo.ca
#SBATCH --mail-type=FAIL
#SBATCH --job-name=sa-canopex
#SBATCH --array=1-2

# job-id  :: ${SLURM_ARRAY_JOB_ID}
# task-id :: ${SLURM_ARRAY_TASK_ID}

nsets=1000

# nsets =   10 -->      7.0min/basin  --> 6.8h for each of 100 tasks                 #SBATCH --mem-per-cpu=3G
# nsets =    2 -->      1.6min/basin  --> 1h 50min for each of 100 tasks             #SBATCH --mem-per-cpu=3G
# nsets = 1000 -->    18:00:00/basin  --> 6-18:00:00 for 645 tasks (9 basins each)   #SBATCH --mem-per-cpu=50G
# nsets = 1000 -->    18:00:00/basin  --> 5-00:00:00 for 601 tasks (5 basins each)   #SBATCH --mem-per-cpu=50G

# change to right dir
cd /home/julemai/projects/rpp-hwheater/julemai/xSSA-North-America/scripts

# set Python env
source ../env-3.5/bin/activate

# folders=$(  \ls -d ../data_in/data_obs/*/ | cut -d '/' -f 4 | sort )
# folders=$( cat basins_5yr_nse-gt-05.dat )                             # <<<<<<<<<<<< only basins with more than 5 years (calibrated) and NSE > 0.5

# set tasks to 139
# basins=(08189500 08193000 08206600 08206700 08207500 08208000 08210000 08211000 08211500 08217500 08219500 08220000 08227000 08246500 08249000 08255500 08268700 08276300 08276500 08279000 08279500 08284100 08289000 08313000 08317400 08319000 08323000 08324000 08329928 08330000 08332010 08379500 08401900 08CC001 08CD001 08CE001 08CG003 08DB001 08DD001 08EB004 08EB005 08EC013 08ED001 08ED002 08EE004 08EE013 08EE020 08EF001 08EF005 08FB006 08FB007 08FC003 08FE003 08FF001 08FF002 08GA071 08GD004 08GD008 08GE002 08HA001 08HA010 08HB002 08HD011 08HF005 08JB002 08JB003 08JE001 08KA001 08KA004 08KA005 08KA007 08KA008 08KB001 08KB003 08KC001 08KC003 08KD001 08KD006 08KD007 08KE009 08KE016 08KF001 08KG001 08KG003 08KH001 08KH006 08KH010 08KH019 08LA001 08LB020 08LB047 08LB064 08LB069 08LD001 08LE024 08LE027 08LE031 08LF051 08LG008 08LG048 08MA001 08MA002 08MB005 08MB006 08ME025 08MF065 08MG001 08MG005 08MG013 08NA002 08NA006 08NA045 08NB005 08NB012 08NB014 08NB019 08NC004 08ND012 08ND013 08NE001 08NE006 08NE039 08NG053 08NG065 08NH005 08NH007 08NH119 08NH130 08NJ013 08NK002 08NK016 08NK018 08NL024 08NL050 08NN002 08NN022 08NP001 08OA002 09033300)

# set tasks to 139
# basins=(09034250 09041400 09050700 09058000 09060770 09067005 09067020 09070000 09070500 09081000 09083800 09085000 09085100 09093700 09095500 09097900 09106150 09109000 09110000 09112200 09112500 09114500 09118450 09119000 09124500 09132500 09134100 09135950 09144250 09147025 09147500 09149500 09152500 09163500 09166500 09168730 09169500 09171100 09172500 09174600 09177000 09180000 09180500 09188500 09205000 09209400 09211200 09213500 09224700 09237450 09237500 09239500 09242500 09246400 09249750 09251000 09251100 09253000 09260000 09260050 09261000 09271550 09277500 09279150 09285900 09288180 09301500 09302000 09303000 09304115 09304200 09304500 09304800 09306200 09306222 09306290 09306500 09313000 09315000 09342500 09346400 09349800 09361500 09363500 09364010 09364500 09365000 09366500 09381800 09382000 09384000 09386950 09400350 09402000 09403600 09404900 09405500 09406000 09406100 09408135 09408150 09409880 09410100 09413000 09413700 09414900 09415000 09418500 09418700 09419700 09424447 09424900 09430500 09431500 09432000 09442000 09442680 09444000 09444200 09444500 09447000 09447800 09448500 09466500 09468500 09469500 09470000 09474000 09481740 09482000 09482500 09484500 09486055 09486500 09489000 09489500 09490500 09494000 09496500)

# set tasks to 139
# basins=(09497500 09497800 09497980 09498400 09499000 09502000 09502800 09503700 09504000 09504420 09504500 09505800 09506000 09507980 09508500 09510000 09511300 09512406 09512500 09512800 09513910 09514100 09516500 09518000 09519800 09520500 10016900 10020100 10020300 10028500 10038000 10039500 10092700 10109000 10109001 10113500 10118000 10126000 10129500 10130500 10131000 10132000 10136500 10140100 10141000 10150500 10155000 10155200 10155300 10159500 10174500 10183500 10217000 10239000 10251250 10251255 10251258 10261100 10293000 10296500 10297500 10300000 10308200 10309000 10310407 10311000 10311400 10311700 10312000 10315500 10317500 10318500 10321000 10321590 10321940 10321950 10322000 10322500 10324500 10329000 10337500 10338000 10339419 10346000 10347460 10348000 10348200 10350000 10350340 10350400 10351600 10351650 10351700 10352500 10396000 10AC004 10BE004 10BE007 10CB001 10CC002 10CD001 10CD004 11012000 11022480 11023000 11042000 11043000 11044000 11044300 11046000 11051499 11051500 11066460 11070150 11070365 11070500 11085000 11106550 11108500 11109000 11109375 11109525 11113000 11114000 11123000 11123500 11128500 11133000 11134000 11136800 11138500 11140000 11143250 11147500 11149900 11151300 11151700 11152000 11152050)

# set tasks to 139
# basins=(11152300 11152500 11156500 11157500 11158600 11159000 11172175 11176900 11177000 11179000 11180700 11186000 11186001 11189500 11192500 11192501 11216200 11218400 11234760 11238600 11242000 11246700 11251000 11261500 11266500 11270900 11272500 11276500 11276600 11276900 11278300 11278400 11289650 11289651 11289660 11290000 11292700 11292900 11295300 11302000 11303000 11303500 11316600 11316670 11319500 11323500 11325500 11335000 11342000 11345500 11348500 11367500 11367800 11368000 11370500 11372000 11374000 11376000 11376550 11377100 11382000 11383500 11389500 11390500 11413000 11413520 11417500 11418000 11421000 11424000 11427000 11433300 11443500 11444500 11446500 11451000 11452500 11458000 11462500 11463000 11463980 11464000 11467000 11468000 11469000 11470500 11471500 11473900 11475000 11475800 11476500 11477000 11478500 11481000 11482500 11493500 11501000 11502500 11507500 11517000 11517500 11519500 11522500 11525655 11525854 11526250 11527000 11528700 11530000 11532500 12017000 12027500 12031000 12035000 12035002 12039500 12040500 12041200 12044900 12045500 12061500 12086500 12089500 12097850 12098500 12099200 12101500 12105900 12106700 12113000 12134500 12144500 12149000 12150800 12167000 12189500 12194000 12200500 12210500)

# set tasks to 139
# basins=(12210700 12213100 12301300 12302055 12303500 12323600 12324200 12324590 12324680 12329500 12331500 12331800 12334510 12334550 12335100 12338300 12340000 12340500 12342500 12344000 12350250 12351200 12352500 12353000 12354000 12354500 12358500 12359800 12363000 12365000 12370000 12372000 12388200 12388700 12389000 12389500 12391400 12391950 12392000 12394000 12395000 12395500 12396500 12398600 12409000 12411000 12413000 12413470 12413500 12413860 12414500 12414900 12415140 12419000 12422500 12424000 12433000 12447383 12448000 12448500 12448998 12449500 12449950 12451000 12452800 12452990 12457000 12459000 12462500 12465000 13010065 13011900 13013650 13015000 13018750 13022500 13023000 13027500 13032500 13037500 13038500 13041010 13042500 13046000 13046995 13047500 13047600 13049500 13050500 13052200 13055000 13055340 13056500 13057000 13057940 13062500 13063000 13069500 13073000 13075000 13075500 13075910 13077000 13078000 13081500 13082500 13105000 13112000 13116500 13120500 13127000 13132500 13135000 13139500 13140800 13141000 13141500 13142500 13147900 13148500 13152500 13161500 13168500 13169500 13171620 13172500 13175100 13181000 13183000 13185000 13186000 13200000 13202000 13206000 13210050 13213000 13213100 13217500 13235000)

# set tasks to 134
# basins=(13237920 13246000 13247500 13249500 13250000 13251000 13258500 13265500 13266000 13269000 13273000 13277000 13285500 13286700 13290190 13292000 13296500 13302500 13305000 13305310 13307000 13309220 13310199 13310700 13313000 13314300 13316500 13317000 13329770 13331450 13331500 13333000 13335690 13335700 13336500 13337000 13337500 13338500 13339500 13340000 13340600 13341050 13342450 13342500 13344500 13345000 13351000 14018500 14020850 14033500 14038530 14044000 14046000 14046500 14048000 14076500 14087380 14087400 14092500 14097100 14103000 14111400 14113000 14120000 14123500 14137000 14137002 14142500 14144800 14148000 14152000 14154500 14155500 14157500 14162500 14163150 14163900 14166000 14169000 14170000 14174000 14178000 14183000 14184100 14187200 14187500 14189000 14190500 14191000 14194150 14197900 14200000 14201340 14202000 14207500 14209500 14210000 14211010 14211720 14220500 14226500 14231000 14231900 14232500 14233500 14238000 14242580 14243000 14301000 14305500 14306500 14307620 14308000 14310000 14312000 14316500 14316700 14319500 14320700 14321000 14328000 14330000 14337500 14337600 14339000 14357500 14359000 14361500 14362000 14366000 14369500 14372300 14377100 14400000)

# set tasks to 82 (remaining unfinished)
# basins=(01031500 01578310 01594440 01619500 02045500 02047000 02047500 02049500 02071000 02131000 02317500 03212980 03213700 03349000 03351000 03352500 03353000 03353611 03353800 03354000 03424730 03527000 03589500 04133501 04135700 04136000 04136500 04136900 04137005 04137500 04142000 04180500 04228500 04232000 05330000 06033000 06077200 06212500 06436760 06437000 06438000 06438500 06439000 06440200 06441500 06799100 06799350 06799500 06800500 06803080 06803486 06803513 06811500 06813500 06815000 06916600 07336200 07337900 07338500 08289000 08LA001 08LB020 08LB047 08LB064 08MB006 08ME025 09147025 09147500 09211200 09386950 09413000 09418500 09424447 09424900 09430500 09498400 09499000 10126000 10141000 10251258 10396000 13018750)

# set tasks to 11 (remaining unfinished)
# basins=(02OD003 02PJ030 02RH045 02RH049 02WB003 02YM001 03106000 08013000 08013500 08041780 10251258)

# set tasks to 2 (remaining unfinished)
basins=(05UH002 06012500)

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


# 1000 sets, package 1, 139 basins
# job ID: 30681534

# 1000 sets, package 2, 139 basins
# job ID: 30721121

# 1000 sets, package 3, 139 basins
# job ID: 30767434

# 1000 sets, package 4, 139 basins
# job ID: 30810119

# 1000 sets, package 5, 139 basins
# job ID: 30822779

# 1000 sets, package 6, 134 basins
# job ID: 30849620

# 1000 sets, package 7, 82 basins
# job ID: 30917251

# 1000 sets, package 8, 11 basins
# job ID: 30956187

# 1000 sets, package 9, 2 basins
# job ID: 31034374