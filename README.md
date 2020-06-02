# xSSA-North-America
The sensitivity of hydrologic parameters and processes across North America


```
cd scripts

bb='03054500'   # an example basin
nsets='1000'    # number of iterations for analysis --> nsets='1000' takes about 22h --> number of model runs = (35+2) x nsets
tmpdir='/tmp/test-xssa' # a temporary directory; use ${SLURM_TMPDIR} for jobs on Graham

python raven_sa-usgs-canopex.py -i ${bb} -n ${nsets} -t ${tmpdir} -o nc
```
