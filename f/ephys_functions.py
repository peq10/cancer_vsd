# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:51:51 2018

@author: peq10
"""

import neo
import sys
import quantities as pq
import numpy as np


def load_ephys(path,lazy = False):
    reader = neo.io.Spike2IO(path)
    bl = reader.read(lazy = lazy)[0]
    
    return bl


def load_ephys_parse(fname, analog_names = ['LED','vcVm'],event_names = ['CamDown','Keyboard']):
    s = load_ephys(fname).segments[0]
    as_names = [s.analogsignals[i].name for i in range(len(s.analogsignals))]
    res_dict = {}
    for name in analog_names:
        try:
            idx = as_names.index(name)
            res_dict[name] = s.analogsignals[idx]
        except ValueError:
            idx = [i for i,x in enumerate(as_names) if name in x][0]
            bundle = as_names[idx][len('Channel bundle ('):as_names[idx].find(')')].strip().split(',')
            idx2 = bundle.index(name)
            res_dict[name] = s.analogsignals[idx][:,idx2]
            
    ev_names = [s.events[i].name for i in range(len(s.events))]
    for name in event_names:
        idx = ev_names.index(name)
        
        res_dict[name+'_times'] = s.events[idx].times.magnitude
        res_dict[name+'_labels'] = s.events[idx].labels
    
        #for backwards compatibility
        if 'cam' in name.lower():
            res_dict['cam'] = s.events[idx].times.magnitude
            
    ephys_start = get_ephys_datetime(fname)
    res_dict['ephys_start']  = ephys_start
     
    return res_dict


def get_ephys_datetime(file_path):
    '''
    See header description below
    Reads the correct bytes from the ephys header for the 'datetime_year' and 'datetime_detail' fields
    '''
    year = np.fromfile(file_path,dtype = np.uint16,count = 30)[29]
    #ucHun,ucSec,ucMin,ucHour,ucDay,ucMon = np.fromfile(file_path,dtype = np.uint8,count = 59)[52:58].astype(str)
    detail = np.fromfile(file_path,dtype = np.uint8,count = 59)[52:58].astype(str)
    time = detail[1:4]
    time2 = []
    for val in time[::-1]:
        if len(val) == 2:
            time2.append(val)
        else:
            time2.append(str(0)+val)
        
    
    day = int(str(year)+str(detail[-1])+str(detail[-2]))
    time = int(time2[0]+time2[1]+time2[2])
    ms = str(int(detail[0])*100)
    return day,time,ms
    
    



def time_to_idx(analogSignal,time):
    try:
        idx = np.round((time - analogSignal.t_start)*analogSignal.sampling_rate).astype(int).magnitude
    except Exception:
        idx = np.round((time*pq.s - analogSignal.t_start)*analogSignal.sampling_rate).astype(int).magnitude
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    return idx


def idx_to_time(analogSignal,idx):
    time = analogSignal.t_start + idx/analogSignal.sampling_rate
    return time

'''
function [Head]=SONFileHeader(fid)
% SONFILEHEADER reads the file header for a SON file
%
% HEADER=SONFILEHEADER(FID)
%
% Used internally by the library. 
% See CED documentation of SON system for details.
%
% 24/6/05 Fix filecomment - now 5x1 not 5x5
%
% Malcolm Lidierth 03/02
% Updated 06/05 ML
% Copyright © The Author & King's College London 2002-2006

try
    frewind(fid);
catch
    warning(['SONFileHeader:' ferror(fid) 'Invalid file handle?' ]);
    Head=[];
    return;
end;

Head.FileIdentifier=fopen(fid);
Head.systemID=fread(fid,1,'int16');
Head.copyright=fscanf(fid,'%c',10);
Head.Creator=fscanf(fid,'%c',8);
Head.usPerTime=fread(fid,1,'int16');
Head.timePerADC=fread(fid,1,'int16');
Head.filestate=fread(fid,1,'int16');
Head.firstdata=fread(fid,1,'int32');
Head.channels=fread(fid,1,'int16');
Head.chansize=fread(fid,1,'int16');
Head.extraData=fread(fid,1,'int16');
Head.buffersize=fread(fid,1,'int16');
Head.osFormat=fread(fid,1,'int16');
Head.maxFTime=fread(fid,1,'int32');
Head.dTimeBase=fread(fid,1,'float64');
if Head.systemID<6
    Head.dTimeBase=1e-6;
end;
Head.timeDate.Detail=fread(fid,6,'uint8');
Head.timeDate.Year=fread(fid,1,'int16');
if Head.systemID<6
    Head.timeDate.Detail=zeros(6,1);
    Head.timeDate.Year=0;
end;
Head.pad=fread(fid,52,'char=>char');
Head.fileComment=cell(5,1);    

pointer=ftell(fid);
for i=1:5
    bytes=fread(fid,1,'uint8');
    Head.fileComment{i}=fread(fid,bytes,'char=>char')';
    pointer=pointer+80;
    fseek(fid,pointer,'bof');
end;

'''
