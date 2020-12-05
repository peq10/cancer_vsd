#PBS -lwalltime=00:20:00
#PBS -lselect=1:ncpus=1:mem=32gb
#PBS -J 1-2
module load anaconda3/personal
source activate cancer_HPC


FIREFLY=$RDS_PROJECT/thefarm2/live/Firefly #helper
df_path=$FIREFLY/cancer/analysis/long_acqs_20201205_HPC.csv 

python $FIREFLY/cancer/analysis/cancer/create_paper_data.py $PBS_ARRAY_INDEX

