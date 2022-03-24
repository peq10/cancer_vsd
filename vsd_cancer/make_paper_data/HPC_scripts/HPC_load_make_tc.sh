#PBS -lwalltime=02:00:00
#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -J 1-254


/rds/general/user/peq10/home/anaconda3/envs/cancer_HPC/bin/python /rds/general/user/peq10/projects/thefarm2/live/Firefly/cancer/analysis/code_2022/cancer_vsd/vsd_cancer/make_paper_data/create_paper_data.py $PBS_ARRAY_INDEX 0 
