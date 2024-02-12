import numpy as np
import pandas as pd
import yaml
import os, re, struct
from scipy.signal import find_peaks
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import h5py
#import seaborn as sns
#import matplotlib.pyplot as plt
from tqdm import tqdm

def get_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def select_closest_nodes(root_dir, filename_coor, bins,step):
    ind_2d = np.zeros((bins, bins))
    coor = pd.read_csv(root_dir+filename_coor, sep=' ', names=['x','y','z'])[1:]    
    array= pd.DataFrame(columns = ['x_coor', 'y_coor'])
    array['x_coor'] = coor['x'].values
    array['y_coor'] = coor['y'].values
    
    for x in range(bins):
        for y in range(bins):
            vect_1 = np.array(((2*x+1)/(2*bins), (2*y+1)/(2*bins)))
            part_array = array[(array['x_coor'].between((x/bins), (x+step)/bins, inclusive=False)) \
                  & (array['y_coor'].between((y/bins), (y+step)/bins, inclusive=False))]
            dist=[]
            for i in range(part_array.shape[0]):
                vect_2 = np.array((part_array['x_coor'].iloc[i],part_array['y_coor'].iloc[i]))
                dist.append(np.linalg.norm(vect_1-vect_2)) 
            if np.size(dist) == 0:
                ind_2d[x,y] = float("nan")
                continue
            index =  np.argmin(dist)
            ind_2d[x,y] = part_array.iloc[index].name
    return ind_2d

def select_indexes(biatrial_coor, atrial_coor, atrial_indexes):
    bins = atrial_indexes.shape[0]
    indexes_biatrial = np.zeros((bins,bins))
    for x in range(bins):
        for y in range(bins):
            try:
                x_biatrial = atrial_coor.iloc[int(atrial_indexes[x,y])]['x']*1000   #if LA_only - *1000 remove
                partial_biatrial_coor = biatrial_coor[biatrial_coor['x']==round(x_biatrial,0)]
                if len(partial_biatrial_coor.index.to_list())!=1:
                    y_biatrial = atrial_coor.iloc[int(atrial_indexes[x,y])]['y']*1000 #if LA_only - *1000 remove
                    partial_biatrial_coor = partial_biatrial_coor[partial_biatrial_coor['y']==round(y_biatrial,0)]
                    if len(partial_biatrial_coor.index.to_list())!=1:
                        z_biatrial = atrial_coor.iloc[int(atrial_indexes[x,y])]['z']*1000 #if LA_only - *1000 remove
                        partial_biatrial_coor = partial_biatrial_coor[partial_biatrial_coor['z']==round(z_biatrial,0)]
                index_biatrial=partial_biatrial_coor.index.to_list()[0]
                indexes_biatrial[x,y] = index_biatrial-1 # -1 because biatrial coordinates start from 1
            except ValueError: indexes_biatrial[x,y] = float("nan")
            except IndexError: indexes_biatrial[x,y] = float("nan")
    return indexes_biatrial

def read_array_igb(igbfile):
    """
    Purpose: Function to read a .igb file
    """
    data = []
    file = open(igbfile, mode="rb")
    header = file.read(1024)
    words = header.split()
    word = []
    for i in range(4):
        word.append(int([re.split(r"(\d+)", s.decode("utf-8")) for s in [words[i]]][0][1]))

    nnode = word[0] * word[1] * word[2]

    for _ in range(os.path.getsize(igbfile) // 4 // nnode):
        data.append(struct.unpack("f" * nnode, file.read(4 * nnode)))

    file.close()
    return data

def read_array_elem(elemFile):
    """
    Purpose: Function to read a .elem file from CARP, where the first
    line contains the number of elements and the remaining lines contain
    the element conectivity

    Returns a list
    """
    f = open(elemFile)

    # extract number of elements
    header = f.readline()

    # extract element list
    #    elemList = [(line.strip()).split(' ') for line in f.readlines()]
    elemList = [(line.strip()).split() for line in f.readlines()]

    f.close()

    return elemList

def dom_freq(data, indexes, fs=200, start_time =0, low_band=0, high_band=20):
    df_map = np.zeros((indexes.shape[0],indexes.shape[1]))

    for x in range(indexes.shape[0]):
        for y in range(indexes.shape[1]):
            if not np.isnan(indexes[x,y]):
                signal = data[start_time:,int(indexes[x,y])] 
                N = int(signal.shape[0])
                fft_yf = np.fft.fft(signal)
                fft_xf = np.fft.fftfreq(N, 1/fs)

                fft_20_index = np.argwhere((fft_xf<high_band) & (fft_xf>low_band))        
                fft_yf_20 = fft_yf[fft_20_index] #cutting on 20Hz
                fft_xf_20 = fft_xf[fft_20_index] #cutting on 20Hz

                spectrum = []
                for i in range(len(np.abs(fft_yf_20))):
                    spectrum.append(int(np.abs(fft_yf_20)[i]))

                peaks, properties = find_peaks(spectrum, height=0)
                try:
                    df_index = np.argsort(properties['peak_heights'])[-1]
                    df = float(fft_xf_20[peaks[df_index]])
                except:
                    df = 0
                df_map[x,y] = df
    return df_map

def fibrosis_2d(fibre_1d,indexes):
    fibre_2d = np.zeros((indexes.shape[0],indexes.shape[1]))
    for x in range(indexes.shape[0]):
        for y in range(indexes.shape[1]):
            try:
                fibre_2d[x,y] = float(fibre_1d.iloc[int(indexes[x,y])].values)
            except ValueError: fibre_2d[x,y] =0
    return fibre_2d

def mono(array, indexes):
    array_atrial = np.zeros((indexes.shape[0],indexes.shape[1]))
    for x in range(indexes.shape[0]):
        for y in range(indexes.shape[1]):
            try:
                array_atrial[x,y] = float(array.iloc[int(indexes[x,y])].values)
            except ValueError:
                array_atrial[x,y] = 0 #float("nan")
    return array_atrial

def last_peak_time(data_AF, indexes):
    last_peak = np.zeros((indexes.shape[0],indexes.shape[1]))
    for x in range(indexes.shape[0]):
        for y in range(indexes.shape[1]):
            try:
                signal = data_AF.T[int(indexes[x,y])]
                peaks, properties = find_peaks(signal)
                last_peak[x,y] = peaks[-1]
            except IndexError: last_peak[x,y] = float('nan')
            except ValueError: last_peak[x,y] = float('nan')
    return last_peak