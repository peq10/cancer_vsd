# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:16:14 2021

@author: Firefly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catch22 import catch22_all
import scipy.ndimage as ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import palettable

tc_filt=pd.read_csv(r'C:\Users\Firefly\Desktop\Projection\tc_filt_231.csv')
tc_filt=tc_filt.iloc[:,1:]

#Feature Extraction
features=pd.DataFrame([])
for i in range(tc_filt.shape[0]):
    df=tc_filt.iloc[i,:]
    #Feature Extraction
    catch22_out=catch22_all(df)
    features=features.append(catch22_out,ignore_index=True)
values=features['values'].apply(pd.Series)
values.columns=['DN_HistogramMode_5','DN_HistogramMode_10','CO_f1ecac','CO_FirstMin_ac','CO_HistogramAMI_even_2_5','CO_trev_1_num','MD_hrv_classic_pnn40','SB_BinaryStats_mean_longstretch1','SB_TransitionMatrix_3ac_sumdiagcov','PD_PeriodicityWang_th0_01','CO_Embed2_Dist_tau_d_expfit_meandiff','IN_AutoMutualInfoStats_40_gaussian_fmmi','FC_LocalSimple_mean1_tauresrat','DN_OutlierInclude_p_001_mdrmd','DN_OutlierInclude_n_001_mdrmd','SP_Summaries_welch_rect_area_5_1','SB_BinaryStats_diff_longstretch0','SB_MotifThree_quantile_hh','SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1','SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1','SP_Summaries_welch_rect_centroid','FC_LocalSimple_mean3_stderr']

#Raw data distribution
import seaborn as sns
for i in range(values.shape[1]):
    sns.histplot(values.iloc[:,i],bins=50)
    plt.title(values.columns[i])
    plt.show()

#Remove the least informative one    
f=values
f=f.drop(['IN_AutoMutualInfoStats_40_gaussian_fmmi'],axis=1)

#Normalization: Scale[0.0001,1]+Boxcox+Scale[0,1] and check the normalized distribution
import scipy.stats as spstats
min_max_scaler = MinMaxScaler(feature_range=(0.0001,1))
fb = min_max_scaler.fit_transform(f)
fb=pd.DataFrame(fb)

for i in range(f.shape[1]):
    feature=np.array(fb.iloc[:,i])
    l,opt_lambda=spstats.boxcox(feature)
    #print('Optimal lambda value:', opt_lambda)
    f.iloc[:,i]=spstats.boxcox(fb.iloc[:,i],lmbda=opt_lambda)

fb= MinMaxScaler(feature_range=(0,1)).fit_transform(f)
fb= pd.DataFrame(fb)
fb.columns=['DN_HistogramMode_5','DN_HistogramMode_10','CO_f1ecac','CO_FirstMin_ac','CO_HistogramAMI_even_2_5',
            'CO_trev_1_num','MD_hrv_classic_pnn40','SB_BinaryStats_mean_longstretch1','SB_TransitionMatrix_3ac_sumdiagcov',
            'PD_PeriodicityWang_th0_01','CO_Embed2_Dist_tau_d_expfit_meandiff','FC_LocalSimple_mean1_tauresrat',
            'DN_OutlierInclude_p_001_mdrmd','DN_OutlierInclude_n_001_mdrmd','SP_Summaries_welch_rect_area_5_1',
            'SB_BinaryStats_diff_longstretch0','SB_MotifThree_quantile_hh','SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1','SP_Summaries_welch_rect_centroid','FC_LocalSimple_mean3_stderr']

for i in range(fb.shape[1]):
    sns.histplot(fb.iloc[:,i],bins=50)
    #fig = plt.figure()
    #res = stats.probplot(feature.iloc[:,i], plot=plt)
    plt.title('Scale[0.0001,1]+Boxcox')
    plt.ylabel('Number')
    plt.xlabel(fb.columns[i])
    #plt.savefig(fb.columns[i])
    plt.show()

# Estimate k using Silhouette Coefficient - Intercluster similarity
# Comparing Hierarchical and GMM
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as GMM
Scores = []  
g=[]
for k in range(2,9):
    estimator = AgglomerativeClustering(n_clusters=k,linkage='ward')  # 构造聚类器
    estimator.fit(fb)
    Scores.append(silhouette_score(fb,estimator.labels_,metric='euclidean'))
    estimator_g = GMM(n_components=k, random_state=0)
    estimator_g.fit(fb)
    g.append(silhouette_score(fb,estimator_g.predict(fb),metric='euclidean'))
X = range(2,9)
plt.figure(figsize=(10,7),dpi=600)
plt.xlabel('Clusters',fontsize=20)
plt.ylabel('Silhouette Coefficient',fontsize=20)
plt.plot(X,Scores,'o-',label='Hierarchical')
plt.plot(X,g,'o-',label='GMM')
#plt.title('Silhoutte')
plt.legend(fontsize=18,frameon=False)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
#plt.savefig('silhouette coef.png',dpi=1200)
plt.show()

# Hierarchical clustering
T=0.2
from sklearn.cluster import AgglomerativeClustering
n_clusters=5
clustering = AgglomerativeClustering(linkage='ward',n_clusters=n_clusters).fit(fb)
clustering.labels_

ts=tc_filt
pca = PCA(n_components=0.8)
new_pca = pd.DataFrame(pca.fit_transform(fb))
importance = pca.explained_variance_ratio_
c=pca.components_

output_data = pd.concat([ pd.Series(clustering.labels_, index=fb.index),fb], axis=1)   # Output with group information
output_data.columns = ['Type']+list(fb.columns)  # Rename columns
output_ts = pd.concat([pd.Series(clustering.labels_, index=ts.index),ts], axis=1)   # Output with group information
output_ts.columns = ['Type'] + list(ts.columns)   # Rename columns

#output_data.to_csv('Features with label.csv')
#output_ts.to_csv('Timeseries with label.csv')

fig=plt.figure(figsize=(10,7),dpi=600)
ax=fig.add_subplot(111)
title=['Waving','Noisy','Blinking-S','Blinking-L','Quiet']
#title=['Quiet','Waving','Noisy','Blinking-S','Blinking-L']
color=['cornflowerblue','darkorange','green','indianred','mediumpurple']
for i in range(n_clusters):
    t=new_pca[output_data['Type']==i]
    ax.scatter(t.iloc[:,0],t.iloc[:,1],label=title[i],marker='o',c=color[i])

plt.legend(fontsize=20,frameon=False,loc=[0,1.2],ncol=5,title='Type',title_fontsize=28,columnspacing=0.1)
plt.xticks([])
plt.yticks([])
#plt.title('PCA-231-Hierarchical')
kwargs={'linestyle':'-', 'lw':0.1, 'width':0.02} 
for i in range(c.shape[1]):
    ax.arrow(0,0,c[0,i],c[1,i],alpha=0.5,ec='black',fc='black',**kwargs)
    ax.text(c[0,i],c[1,i],f'{i+1}', ha="center", va="center",fontsize=16)
ax.set_xlabel(f'PC1',fontsize=20)  
ax.set_ylabel(f'PC2',fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#plt.savefig('PCA of 231.png',dpi=1200)
#ax.spines['left'].set_linewidth(5)
#ax.spines['bottom'].set_linewidth(4)

# Plot the traces in each type
for i in range(n_clusters):
    g=output_ts[output_ts['Type']==i].iloc[:,1:]
    
    fig,ax=plt.subplots(figsize=(20,g.shape[0]*0.3))
    ax.plot(np.arange(g.shape[1])*T,g.T + np.arange(g.shape[0])/70)
    ax.set_title(f'Timecourse of Type{i+1}',fontsize=40)
    #ax.set_xlabel('Time(s)',fontsize=30)
    #ax.set_ylabel('Fluorescence df/F',fontsize=30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.axis('off')
    

# Selecting reps
#by=[fb.columns[6],fb.columns[9],fb.columns[18],fb.columns[2],fb.columns[10]]
by=[fb.columns[14],fb.columns[8],fb.columns[19],fb.columns[15],fb.columns[6]]
#a=[False,False,False,True,False]
for i in range(n_clusters):
    f=output_data[output_data['Type']==i].iloc[:,1:]
    f=f.sort_values(by=by[i],axis=0,ascending=False)
    t=output_ts.iloc[f.index[0:10],1:]
    T=0.2
    ts=np.array(t)
    fig,ax=plt.subplots(figsize=(20,3),dpi=600)
    ax.plot(np.arange(ts.shape[1])*T,ts.T+ np.arange(ts.shape[0])/100)
    ax.set_title(title[i],fontsize=20)
    #ax.set_xlabel('Time(s)')
    #ax.set_ylabel('Fluorescence df/F')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.axis('off')
    #plt.savefig('Reps-'+title[i]+'.png',dpi=600)
 
    
 # Using t-sne as feature reduction
from sklearn.manifold import TSNE

tsne=TSNE()
new_tsne = pd.DataFrame(tsne.fit_transform(fb))

fig=plt.figure(figsize=(9,6.3))
ax=fig.add_subplot(111)


for j in range(n_clusters):

    t=new_tsne[(output_data['Type']==j)]
    ax.scatter(t.iloc[:,0],t.iloc[:,1],label=title[j],alpha=0.8,marker='o')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#ax.spines['left'].set_color('none')
#ax.spines['bottom'].set_color('none')
plt.legend(fontsize=20,frameon=False,loc=[1,0.4])
plt.xticks([])
plt.yticks([])
plt.xlabel('t-SNE1',fontsize=20)
plt.ylabel('t-SNE2',fontsize=20)
#plt.title('TSNE-231-Hierarchical')
#plt.savefig('tsne of 231.eps',format='eps',dpi=1200)