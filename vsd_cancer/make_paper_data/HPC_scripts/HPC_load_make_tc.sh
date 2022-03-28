#PBS -lwalltime=02:00:00
#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -J 1-255

LOGFILE=/rds/general/user/peq10/home/firefly_link/cancer/analysis/code_2022/cancer_vsd/vsd_cancer/make_paper_data/HPC_scripts/workdir/$PBS_ARRAY_INDEX.log

/rds/general/user/peq10/home/anaconda3/envs/cancer_HPC/bin/python /rds/general/user/peq10/projects/thefarm2/live/Firefly/cancer/analysis/code_2022/cancer_vsd/vsd_cancer/make_paper_data/create_paper_data.py $PBS_ARRAY_INDEX 0 2>&1 | tee $LOGFILE
