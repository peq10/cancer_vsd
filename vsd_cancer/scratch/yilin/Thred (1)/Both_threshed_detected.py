# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:32:34 2021

@author: Yilin
"""

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

tc_Data=np.array(pd.read_csv(r'C:\Users\Firefly\Desktop\20201202_slip1_area2\Filt.csv'))

#print(tc_Data)
T=0.2

#we want to detect what this cell is doing 
#we detect where the cell goes outside a threshold
def soft_upthreshold(arr,thresh,to = 1):
    #Thresholds towards to value
    res = np.copy(arr)
    wh = np.where(arr - to < thresh)
    n_wh = np.where(arr - to >= thresh)
    sgn = np.sign(arr - to)
    
    res[wh] = to
    res[n_wh] -= sgn[n_wh]*thresh
    
    return res

def soft_lowthreshold(arr,thresh,to = 1):
    #Thresholds towards to value
    res = np.copy(arr)
    wh = np.where(arr - to > thresh) #定位不超过阈值的部分，实际上阈值=to+thresh
    n_wh = np.where(arr - to <= thresh) #定位超过阈值的部分
    sgn = np.sign(arr - to) #
    
    res[wh] = to #不超过阈值的部分都赋值为1
    res[n_wh] -= sgn[n_wh]*thresh #超过阈值的部分赋值为与阈值的差值
    
    return res

#see what happens when you change the below value!
hthresh_value = 0.01

lthresh_value = -0.008

df=pd.DataFrame([])
ts=pd.DataFrame([]) #extract the cells that have events

for i in range(0,tc_Data.shape[0]):
    tc=tc_Data[i,252:] #exclude the noise at the initial time points
    hthreshed = soft_upthreshold(tc, hthresh_value)
    threshed = soft_lowthreshold(tc, lthresh_value)
    #now detect the events and calculate their properties
    
    #Detect upper events start and end locations by finding derivative
    hlocs = np.diff((np.abs(hthreshed - 1) != 0).astype(int), prepend=0, append=0)
    #llocs is (num_events,2) array of event start and end indices 
    hllocs = np.array((np.where(hlocs == 1)[0], np.where(hlocs == -1)[0])).T
    
    #now find the properties of the events
    hevent_lengths = np.zeros(hllocs.shape[0])
    hevent_amplitudes = np.zeros_like(hevent_lengths)
    hevent_integrals = np.zeros_like(hevent_lengths)
    
    for idx,l in enumerate(hllocs):
         hevent_lengths[idx] = (l[1] - l[0])*T
         hevent_amplitudes[idx] = tc_Data[i,np.argmax(np.abs(tc_Data[i,l[0]:l[1]]-1))+l[0]] - 1 
         hevent_integrals[idx] = np.sum(tc_Data[i,l[0]:l[1]] - 1)

    
    #Detect lower events start and end locations by finding derivative
    locs = np.diff((np.abs(threshed - 1) != 0).astype(int), prepend=0, append=0) #np.abs(threshed-1)!=0是一个T/F矩阵，不超过阈值的部分为F，超过阈值的为T，用astype函数变换为0/1矩阵，再用diff函数获取超出阈值的起始和结束点
    #llocs is (num_events,2) array of event start and end indices 
    llocs = np.array((np.where(locs == 1)[0], np.where(locs == -1)[0])).T
    
    #now find the properties of the events
    event_lengths = np.zeros(llocs.shape[0])
    event_amplitudes = np.zeros_like(event_lengths)
    event_integrals = np.zeros_like(event_lengths)
    
    for idx,l in enumerate(llocs):
         event_lengths[idx] = (l[1] - l[0])*T
         event_amplitudes[idx] = tc_Data[i,np.argmax(np.abs(tc_Data[i,l[0]:l[1]]-1))+l[0]] - 1 
         event_integrals[idx] = np.sum(tc_Data[i,l[0]:l[1]] - 1)
    
    c=pd.DataFrame([tc_Data[i,0].astype(int),hllocs.shape[0],llocs.shape[0],hevent_lengths.mean(),event_lengths.mean(),np.median(np.abs(hevent_amplitudes))*100,np.median(np.abs(event_amplitudes))*100])
    df=df.append(c.T,ignore_index=True) 
    

   
    print(f'For the {tc_Data[i,0].astype(int)}th cell: Detected {hllocs.shape[0]} upper events which last for {hevent_lengths.mean():.2f}s averagely, median absolute amplitude {np.median(np.abs(hevent_amplitudes))*100:.2f}%')
    print(f'For the {tc_Data[i,0].astype(int)}th cell: Detected {llocs.shape[0]} lower events which last for {event_lengths.mean():.2f}s averagely, median absolute amplitude {np.median(np.abs(event_amplitudes))*100:.2f}%')
    if hllocs.shape[0] or llocs.shape[0] !=0:
        ts=ts.append(pd.DataFrame(tc_Data[i]).T,ignore_index=True)    
df.columns=['Label','Upper Events Number','Lower Events Number','Upper Event Length','Lower Events Time Length','Upper Mean Amplitude','Lower Mean Amplitude']
df.head()
#df.to_csv('Detected.csv')
print(ts)
print(f'In total, {ts.shape[0]} cells have events')