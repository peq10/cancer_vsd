from pathlib import Path
import pandas as pd
from vsd_cancer.functions import cancer_functions as canf
import scipy.ndimage as ndimage
import sys
import numpy as np
import os

#redo = False

home = Path.home()
if "peq10" in str(home):
    HPC = True
    top_dir = Path(Path.home(), "firefly_link/cancer")

elif "ys5320" in str(home):
    HPC = True
    top_dir = Path(Path.home(), "firefly_link/cancer")

elif os.name == "nt":
    HPC = False
    top_dir = Path(r"R:\home\firefly_link\cancer")

elif "quickep" in str(home):
    HPC = False
    top_dir = Path("/Volumes/peq10/home/firefly_link/cancer")

else:
    HPC = False
    top_dir = Path("/home/peter/data/Firefly/cancer")



data_dir = Path(top_dir, "analysis", "full")
viewing_dir = Path(top_dir, "analysis", "full", "tif_viewing")


if not data_dir.is_dir():
    data_dir.mkdir()
    
print("Reading summary df.")

initial_df = Path(top_dir, "analysis", "long_acqs_20220420_HPC_labelled_complete.csv")

df_sum = pd.read_csv(initial_df)

df_sum = df_sum[df_sum['use']=='y']
df_sum = df_sum[df_sum['n_frames']== 10000]
#df.sort_values(by=['expt'], inplace=True)
#print(df_sum.shape)

for line in df_sum.expt.unique():
    print(f'For line {line}')
    
    save_dir = Path(data_dir,f'{line}')
    
    if not save_dir.is_dir():
        save_dir.mkdir()
    
    df = df_sum[df_sum['expt'] == line]
    #print(df.shape)

    ac=[]
    all_tcs=[]
    active_tcs = []
    cell_id = []
    cell_id_active = []
    expt = []
    expt_active = []
    
    print('Generating np.array of all_tcs and active_tcs.')
    for trial in df.trial_string:
        trial_string = trial
        #print(trial_string)
        e = df[df['trial_string'] == trial_string]['expt']
        e = np.array(e)
        #print(e)
        
        trial_save = Path (data_dir,"ratio_stacks")
        
        eve = np.load(Path(trial_save,trial_string,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
        
        tcs = np.load(Path(trial_save,trial_string,f'{trial_string}_all_eroded_median_tcs.npy'))
        #print(tcs.shape)
        
    
        
        events = eve['events'][1]
        ids = [x for x in events.keys() if type(x) != str]
            
            
        for i in range(tcs.shape[0]):
            all_tcs.append(tcs[i,:])
            cell_id.append(f'{trial_string}_cell_{i}')
            expt.append(e)
            #print(len(all_tcs))
            if i in ids:
                ac.append('active')
                active_tcs.append(tcs[i,:])
                expt_active.append(e)
                cell_id_active.append(f'{trial_string}_cell_{i}')
            else:
                ac.append('inactive')
        #print(len(expt))
            
    
    print('Applying Gaussian filter')       
    all_tcs = np.array(all_tcs)
    all_tcs_filt = ndimage.gaussian_filter(all_tcs,(0,3))
    all_tcs_filt=pd.DataFrame(all_tcs_filt)
    
    active_tcs = np.array(active_tcs)
    active_tcs_filt = ndimage.gaussian_filter(active_tcs,(0,3))
    active_tcs_filt=pd.DataFrame(active_tcs_filt)
    
    expt = pd.DataFrame(expt)
    expt.columns = ['expt']
    
    expt_active = pd.DataFrame(expt_active)
    expt_active.columns = ['expt']
    
    cell_id = pd.DataFrame(cell_id)
    cell_id.columns = ['cell_id']
    
    cell_id_active = pd.DataFrame(cell_id_active)
    cell_id_active.columns = ['cell_id_active']
    
    ac = pd.DataFrame(ac)
    ac.columns = ['ac']
    
    all_tcs_filt = pd.concat([expt,cell_id, ac, all_tcs_filt], axis=1)
    active_tcs_filt = pd.concat([expt_active,cell_id_active, active_tcs_filt], axis=1)
    
    all_tcs_filt.sort_values(by=['expt'], inplace=True)
    active_tcs_filt.sort_values(by=['expt'], inplace=True)
    
    print('exporting all_tcs_filt and active_tcs_filt.')
    all_tcs_filt.to_csv(Path(save_dir,f'all_tcs_filt_20220420_{line}.csv'))
    active_tcs_filt.to_csv(Path(save_dir,f'active_tcs_filt_20220420_{line}.csv'))
