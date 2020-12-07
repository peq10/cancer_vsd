#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=1:mem=40gb
#PBS -J 1-80
module load anaconda3/personal
source activate cancer_HPC


FIREFLY=$RDS_PROJECT/thefarm2/live/Firefly #helper
df_path=$FIREFLY/cancer/analysis/long_acqs_20201207_HPC.csv 

echo "Doing "$PBS_ARRAY_INDEX
python $FIREFLY/cancer/analysis/cancer/create_paper_data.py $PBS_ARRAY_INDEX 0

