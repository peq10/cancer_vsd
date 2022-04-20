# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 08:49:01 2019

@author: Peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import tifffile
import time
import f.general_functions as gf
#a script to do the same as aperture 2 but interpolation inn the views space


def rotate(vec,theta,units = 'degrees'):
    if units == 'degrees':
        theta = theta*np.pi/180
    if vec.shape != (2,1):
        vec = np.matrix(vec).transpose()
    mat = np.matrix([[np.cos(theta),-1*np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.squeeze(np.array(mat*vec))


def get_idx_weights(arr):
    #expects [...,sub aperture dim,sub aperture dim,ij axis]

    offset = np.moveaxis(np.reshape(np.indices((2,2)),(2,-1)),0,-1)

    arr_int,arr_rem = np.divmod(arr,1)
    arr_int = arr_int.astype(int)
    arr_idx = np.array([arr_int + offset[None,i,:] for i in range(len(offset))])
    
    #now do bilinear interpolation - arr_rem is position in unit grid space
    arr_weight = np.array([(1-arr_rem[...,0])*(1-arr_rem[...,1]),(1-arr_rem[...,0])*arr_rem[...,1],arr_rem[...,0]*(1-arr_rem[...,1]),arr_rem[...,0]*arr_rem[...,1]])    
    return arr_idx,arr_weight


def get_view_idx(r,center,rad_spots,n_views):
    d = rotate(r,-90)
    #define the center of each ulens image
    yy,xx = np.meshgrid(np.arange(-rad_spots,rad_spots+1,1),np.arange(-rad_spots,rad_spots+1,1),indexing = 'ij')
    grid = np.array([xx*r[0]+yy*d[0]+center[0],xx*r[1]+yy*d[1]+center[1]])
    
    u = np.arange(-(n_views-1)/2,(n_views+1)/2)
    v = np.copy(u)
    uu,vv = np.meshgrid(u,v,indexing = 'ij')
    
    #get view indices
    views = np.expand_dims(np.expand_dims(np.moveaxis(np.array([uu,vv]),0,-1),-1),-1) + grid
    views = np.moveaxis(views,2,-1)
    
    views_idx,views_weight = get_idx_weights(views)
    
    return views_idx,views_weight,uu,vv,grid,d

def get_shift_idx_cent(views_idx,views_weight,uu,vv,alpha,scale,grid2):
    #now calculate shift between different views
    shift = np.moveaxis(np.array([uu,vv])*scale*(1-1/alpha),0,-1)
    shifted_arr = grid2 + np.expand_dims(np.expand_dims(shift,-1),-1)
    shifted_arr = np.moveaxis(shifted_arr,2,-1)
    
    shifted_idx,shifted_weight = get_idx_weights(shifted_arr)
    #set indices outside of bounds to zero
    size = shifted_idx.shape[-2]
    bad_pointers = np.logical_or(np.logical_or(np.logical_or(shifted_idx[...,0]>=size , shifted_idx[...,0]<0 ), shifted_idx[...,1]>=size), shifted_idx[...,1]<0)
    shifted_weight[bad_pointers] = 0
    shifted_idx[bad_pointers,:] = 0
    #index back in to the views at correct shifts
    u =np.arange(uu.shape[0])
    u = u[:,np.newaxis,np.newaxis]
    v = np.copy(u)[np.newaxis,...]
    u = u[...,np.newaxis]
    s = int((size-1)/2)

    
    all_idx = np.array([views_idx[:,u,v,shifted_idx[i,u,v,s,s,0],shifted_idx[i,u,v,s,s,1],:] for i in range(shifted_idx.shape[0])])
    all_weight = np.array([views_weight[:,u,v,shifted_idx[i,u,v,s,s,0],shifted_idx[i,u,v,s,s,0]]*shifted_weight[i,u,v,s,s] for i in range(shifted_idx.shape[0])])

    all_idx = np.reshape(all_idx,(-1,2))
    all_idx = all_idx[...,0]*2048 + all_idx[...,1]
    all_weight = np.ravel(all_weight)
    
    un = np.unique(all_idx)
    i,j = np.divmod(un,2048)
    un_weights = np.array([np.sum(all_weight[all_idx == un[i]]) for i in range(len(un))])
    un_weights = un_weights/np.sum(un_weights)
    
    nonzer = un_weights.nonzero() 
    
    return np.moveaxis([i[nonzer],j[nonzer]],0,-1),un_weights[nonzer]

def get_all_shift(un_idx,un_weight,grid,r,d):
    appr_idx = np.round(grid[None,...] + un_idx[:,None,None,:]).astype(int)
    bad_idx = np.logical_or(np.logical_or(np.logical_or(appr_idx[...,0]>=2048, appr_idx[...,0]<0 ), appr_idx[...,1]>=2048), appr_idx[...,1]<0)
    appr_idx[bad_idx] = 0
    full_weight = un_weight[:,None,None]*np.ones_like(grid[:,:,0])[None,...]
    full_weight[bad_idx] = 0
    return appr_idx,full_weight

def get_refoc(im,idx,weights):
    return np.sum(im[idx[...,0],idx[...,1]]*weights,0)/np.sum(weights,0)

def get_refoc_stack(stack,idx,weights):
    # deals with memory error by iterating over 10 sets of the indices
    res = np.zeros((stack.shape[0],weights.shape[-2],weights.shape[-1]))
    num_iter = np.ceil(idx.shape[0]/500).astype(int)
    idx_t = np.linspace(0,idx.shape[0]+1,num_iter+1).astype(int)
    print(idx.shape[0])
    for ii,i in enumerate(idx_t[:-1]):
        print(i,idx_t[ii+1])
        res += np.sum(stack[np.arange(stack.shape[0])[:,None,None,None],idx[None,i:idx_t[ii+1],...,0],idx[None,i:idx_t[ii+1],...,1]]*weights[None,i:idx_t[ii+1],...],1)
    return res

def get_refocussed(im,r,center,depths,n_views = 21,rad_spots = 45,hw_dict = {'f':7.2*10**3,'mag':25,'NA':1,'n':1.33}):
    scale = hw_dict['f']*hw_dict['NA']/hw_dict['n']
    scale = scale/hw_dict['mag']
    
    #get view indices
    views_idx,views_weight,uu,vv,grid,d = get_view_idx(r,center,rad_spots,n_views)
    grid = np.moveaxis(grid,0,-1) - center[None,None,:] # for future broadcasting
    
    #for each u,v,s,t there should be a pointer to the 4 meta-pixels that make it up - need array of 4,2,19,19,81,81 indices into corresponding view
    #build the unshifted matrix (essentially just a matrix of indices into views)
    yy1,xx1 = np.meshgrid(np.arange(2*rad_spots+1),np.arange(2*rad_spots+1),indexing = 'ij')
    grid2 = np.array([yy1,xx1])
    
    test = np.zeros((len(depths),2048,2048))
    
    size = 2*rad_spots+1
    result = np.zeros((len(depths),size,size))
    t0 = time.time()
    m = len(depths)
    for idx,depth in enumerate(depths):
        alpha = 1 +depth/hw_dict['f']
        un_idx,un_weight = get_shift_idx_cent(views_idx,views_weight,uu,vv,alpha,scale,grid2)
        test[idx,un_idx[:,0],un_idx[:,1]] = un_weight
        
        full_idx,full_weight = get_all_shift(un_idx,un_weight,grid,r,d)
        result[idx,:,:]= get_refoc(im,full_idx,un_weight[:,None,None])
        print('Time: %d s'%(time.time()-t0))
        print('%d%% complete'%(idx*100/m))
        
    return result

def get_all_idx(depth,r,center,rad_spots = 40,n_views=19):
    hw_dict = {'f':7.2*10**3,'mag':25,'NA':1,'n':1.33}
    scale = hw_dict['f']*hw_dict['NA']/hw_dict['n']
    scale = scale/hw_dict['mag']
    
    #get view indices
    views_idx,views_weight,uu,vv,grid,d = get_view_idx(r,center,rad_spots,n_views)
    grid = np.moveaxis(grid,0,-1) - center[None,None,:] # for future broadcasting
    
    #for each u,v,s,t there should be a pointer to the 4 meta-pixels that make it up - need array of 4,2,19,19,81,81 indices into corresponding view
    #build the unshifted matrix (essentially just a matrix of indices into views)
    yy1,xx1 = np.meshgrid(np.arange(2*rad_spots+1),np.arange(2*rad_spots+1),indexing = 'ij')
    grid2 = np.array([yy1,xx1])
    alpha = 1 +depth/hw_dict['f']
    un_idx,un_weight = get_shift_idx_cent(views_idx,views_weight,uu,vv,alpha,scale,grid2)
    full_idx,full_weight = get_all_shift(un_idx,un_weight,grid,r,d)
    
    return full_idx,full_weight
    

def get_stack_refocussed(stack,r,center,depths,n_views = 21,rad_spots = 45,hw_dict = {'f':7.2*10**3,'mag':25,'NA':1,'n':1.33}):
    scale = hw_dict['f']*hw_dict['NA']/hw_dict['n']
    scale = scale/hw_dict['mag']
    
    #get view indices
    views_idx,views_weight,uu,vv,grid,d = get_view_idx(r,center,rad_spots,n_views)
    grid = np.moveaxis(grid,0,-1) - center[None,None,:] # for future broadcasting
    
    #for each u,v,s,t there should be a pointer to the 4 meta-pixels that make it up - need array of 4,2,19,19,81,81 indices into corresponding view
    #build the unshifted matrix (essentially just a matrix of indices into views)
    yy1,xx1 = np.meshgrid(np.arange(2*rad_spots+1),np.arange(2*rad_spots+1),indexing = 'ij')
    grid2 = np.array([yy1,xx1])
    
    
    
    size = 2*rad_spots+1
    result = np.zeros((len(depths),stack.shape[0],size,size))
    t0 = time.time()
    m = len(depths)
    for idx,depth in enumerate(depths):
        alpha = 1 +depth/hw_dict['f']
        un_idx,un_weight = get_shift_idx_cent(views_idx,views_weight,uu,vv,alpha,scale,grid2)
        full_idx,full_weight = get_all_shift(un_idx,un_weight,grid,r,d)
        result[idx,:,:,:]= get_refoc_stack(stack,full_idx,full_weight)
        print('Time: %d mins'%((time.time()-t0)/60))
        print('%d%% complete'%((idx+1)*100/m))
        
    return result

def get_stack_weights(r,center,depths,n_views = 21,rad_spots = 45,hw_dict = {'f':7.2*10**3,'mag':25,'NA':1,'n':1.33}):
    scale = hw_dict['f']*hw_dict['NA']/hw_dict['n']
    scale = scale/hw_dict['mag']
    
    #get view indices
    views_idx,views_weight,uu,vv,grid,d = get_view_idx(r,center,rad_spots,n_views)
    grid = np.moveaxis(grid,0,-1) - center[None,None,:] # for future broadcasting
    
    #for each u,v,s,t there should be a pointer to the 4 meta-pixels that make it up - need array of 4,2,19,19,81,81 indices into corresponding view
    #build the unshifted matrix (essentially just a matrix of indices into views)
    yy1,xx1 = np.meshgrid(np.arange(2*rad_spots+1),np.arange(2*rad_spots+1),indexing = 'ij')
    grid2 = np.array([yy1,xx1])
    


    result = np.zeros((len(depths),2048,2048))
    for idx,depth in enumerate(depths):
        alpha = 1 +depth/hw_dict['f']
        un_idx,un_weight = get_shift_idx_cent(views_idx,views_weight,uu,vv,alpha,scale,grid2)
        result[idx,un_idx[:,0],un_idx[:,1]] = un_weight
        
    return result

def test():
    r,center = (np.array([0.117,19.525]),np.array([1020.4,1024.1]))
    
    stack = tifffile.imread(r'Z:\Firefly\Lightfield\Phantoms\Non Scattering\lf_refl660nm_stack_1\lf_refl660nm_stack_1_MMStack_Pos0.ome.tif')#np.load(df.loc[0,'mean_stack'])
    im = stack[20,...]
    print('Stack loaded...')    
    result = get_refocussed(im,r,center,np.arange(-20,20,5),n_views = 19,rad_spots = 40)
    print('Refocussed.')
    pg.image(result)
 
    
    
    view_idx,view_weight,_,_,_,_ = get_view_idx(r,center,40,19)
    
    test = np.zeros((19*81,19*81))
    for ii in range(19):
        for jj in range(19):
            test[ii*81:(ii+1)*81,jj*81:(jj+1)*81] = np.sum(im[view_idx[:,ii,jj,:,:,0],view_idx[:,ii,jj,:,:,1]]*view_weight[:,ii,jj,:,:],0)
    
    fig,ax = plt.subplots()
    ax.imshow(gf.bin_array(test,3),cmap = 'Greys_r')
    ax.axis('off')
    fig.savefig(r'C:\Users\peq10\Dropbox\Papers\lightfield\figures\views.png',bbox_inches = 'tight',dpi = 300)
    


    
    fig,ax = plt.subplots()
    ii = 9
    jj = 9
    fig,ax = plt.subplots()
    ax.imshow(np.sum(im[view_idx[:,ii,jj,:,:,0],view_idx[:,ii,jj,:,:,1]]*view_weight[:,ii,jj,:,:],0),cmap = 'Greys_r')
    ax.axis('off')
    fig.savefig(r'C:\Users\peq10\Dropbox\Papers\lightfield\figures\view.png',bbox_inches = 'tight',dpi = 300)
    
    fig,ax = plt.subplots()
    ax.imshow(im,cmap = 'Greys_r')
    ax.set_xlim([800,1600])
    ax.set_ylim([1500,900])
    ax.axis('off')
    fig.savefig(r'C:\Users\peq10\Dropbox\Papers\lightfield\figures\raw_small.png',bbox_inches = 'tight',dpi = 300)
    
    fig,ax = plt.subplots()
    test = np.copy(im)
    test[900:1500,798:802] = np.max(test)
    test[900:1500,1598:1602] = np.max(test)
    test[898:902,800:1600] = np.max(test)
    test[1498:1502,800:1600] = np.max(test)
    test = gf.bin_array(test,4)
    ax.imshow(test,cmap = 'Greys_r')

    ax.axis('off')
    fig.savefig(r'C:\Users\peq10\Dropbox\Papers\lightfield\figures\raw.png',bbox_inches = 'tight',dpi = 300)
    
    res = np.squeeze(get_refocussed(im,r,center,np.arange(0,1,1),n_views = 19,rad_spots = 40))
    
    fig,ax = plt.subplots()
    ax.imshow(res,cmap = 'Greys_r')
    ax.axis('off')
    fig.savefig(r'C:\Users\peq10\Dropbox\Papers\lightfield\figures\refocussed.png',bbox_inches = 'tight',dpi = 300)
    
    res = np.squeeze(get_refocussed(im,r,center,np.arange(-28,-27,1),n_views = 19,rad_spots = 40))
    
    fig,ax = plt.subplots()
    ax.imshow(res,cmap = 'Greys_r')
    ax.axis('off')
    fig.savefig(r'C:\Users\peq10\Dropbox\Papers\lightfield\figures\refocussed_2.png',bbox_inches = 'tight',dpi = 300)
    
    
    r = np.array([0,19])
    center = np.array([1023,1023])
    w = get_stack_weights(r,center,np.arange(-20,20,2),n_views = 21,rad_spots = 45,hw_dict = {'f':7.2*10**3,'mag':25,'NA':1,'n':1.33})
     
         
    fig3,axarr3 = plt.subplots(ncols = 3)
    axarr3[0].imshow(w[0,...])
    axarr3[1].imshow(w[1,...])
    axarr3[2].imshow(w[2,...])
    
    for ax in np.ravel(axarr3):
        ax.axis('off')
        ax.set_xlim([924,1124])
        ax.set_ylim([924,1124])
        
        
    fig3.savefig(r'C:\Users\peq10\Dropbox\Papers\lightfield\figures\scratch_and_thesis\+-10_point_weights_refoc.png',dpi = 300,bbox_inches = 'tight')

