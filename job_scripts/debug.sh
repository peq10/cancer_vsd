#PBS -lwalltime=00:20:00
#PBS -lselect=1:ncpus=1:mem=40gb
#PBS -J 1-2
module load anaconda3/personal
source activate cancer_HPC


FIREFLY=$RDS_PROJECT/thefarm2/live/Firefly #helper
df_path=$FIREFLY/cancer/analysis/long_acqs_20201230_experiments_correct.csv 

python $FIREFLY/cancer/analysis/cancer/HPC_analyse.py $PBS_ARRAY_INDEX $df_path --redo_load True --redo_tc True

