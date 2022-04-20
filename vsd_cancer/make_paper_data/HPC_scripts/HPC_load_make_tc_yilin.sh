#PBS -lwalltime=02:00:00
#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -J 1-2

/rds/general/user/ys5320/home/anaconda3/envs/cancer_HPC_tst/bin/python /rds/general/user/ys5320/projects/thefarm2/live/Firefly/cancer/analysis/code_2022/cancer_vsd/vsd_cancer/make_paper_data/create_paper_data.py $PBS_ARRAY_INDEX 1
