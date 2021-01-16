

module load anaconda3/personal
source activate cancer_HPC

FIREFLY=$RDS_PROJECT/thefarm2/live/Firefly #helper
df_path=$FIREFLY/cancer/analysis/long_acqs_20201205_HPC.csv 
num_runs=$(python -c "import pandas as pd;  df = pd.read_csv('${df_path}'); print(len(df))") 
echo $num_runs