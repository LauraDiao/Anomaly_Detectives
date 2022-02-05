
import numpy as np
import json
import sys
import pandas as pd
import os
from os import listdir
    
from helper import *
def readfilerun_simple(filename, losslog_dir='data/raw/train_r'):
    '''does what readfilerun does but to a single file, no directory iteration, dataframe output.'''
    
    run_labels = filename.split('_')[-1].split('-')[:-1]
    temp_label_str = '-'.join(run_labels) 
    losslog = f'{losslog_dir}/losslog-{temp_label_str}.csv' #losslog filename str
    
    run_df = pd.read_csv(filename)
    losslog_df = pd.read_csv(losslog, header=None).rename(
        columns={0:'event', 1:'drop_unix', 2:'IP1', 3:'Port1', 4:'IP2', 5:'Port2', 6:'Proto'}).ffill()
    losslog_df['Time'] = losslog_df['drop_unix'].astype(int)
    losslog_df = losslog_df.groupby(['Time', 'IP1', 'Port1', 'IP2', 'Port2', 'Proto']).agg(
        lambda x: ';'.join(x.astype(str))).reset_index() # TODO find a way to handle this see: event and drop_unix columns should be semicolon separable values in string format.
    
    df = pd.merge(run_df, losslog_df, on=['Time', 'IP1', 'Port1', 'IP2', 'Port2', 'Proto'], how="left") # merge on fivetuple key
    # df.fillna(inplace=True, value=-1) #TODO plan implementation of dealing with null values
    # df['event'] =  # TODO change to 3 different profiles: no drop, drop, and switch event
    df = df[df['Proto'] == df['Proto'].mode()[0]] # selects relevant non ipv6 int(connection
    
    ## adding labels
    df['latency'] = int(run_labels[0])
    df['loss'] = int(run_labels[1])
    df['later_latency'] = int(run_labels[3])
    df['later_loss'] = int(run_labels[4])
    df['deterministic'] = bool(run_labels[2])
    return df

def readfilerun(run_, output_dir):
    '''reads files in one directory, cleans, labels and then outputs them to other directory'''
    names = listdir(run_) # all filenames in data
    daneruns = [x for x in names if not 'losslog' in x]
    daneruns = [ filename for filename in daneruns if filename.endswith(".csv" ) ]
    # daneruns = ['data/20220116T055105_20-100-true-20-100-iperf.csv', 'data/20220116T055942_20-250-true-20-250-iperf.csv']
    losslogs = [x for x in names if 'losslog' in x]
    
    for run in daneruns: # run = a single csv from danerun inside the run_
        #print(run) #for debug
        
        run_labels = run.split('_')[-1].split('-')[:-1]
        
        temp_label_str = '-'.join(run_labels) 
        losslog = f'{run_}/losslog-{temp_label_str}.csv' #losslog filename str
        
        run_df = pd.read_csv(f'{run_}/{run}')
        losslog_df = pd.read_csv(losslog, header=None).rename(
            columns={0:'event', 1:'drop_unix', 2:'IP1', 3:'Port1', 4:'IP2', 5:'Port2', 6:'Proto'}).ffill()
        losslog_df['Time'] = losslog_df['drop_unix'].astype(int)
        losslog_df = losslog_df.groupby(['Time', 'IP1', 'Port1', 'IP2', 'Port2', 'Proto']).agg(
            lambda x: ';'.join(x.astype(str))).reset_index() # TODO find a way to handle this see: event and drop_unix columns should be semicolon separable values in string format.
        
        df = pd.merge(run_df, losslog_df, on=['Time', 'IP1', 'Port1', 'IP2', 'Port2', 'Proto'], how="left") # merge on fivetuple key
        # df.fillna(inplace=True, value=-1) #TODO plan implementation of dealing with null values
        # df['event'] =  # TODO change to 3 different profiles: no drop, drop, and switch event
        df = df[df['Proto'] == df['Proto'].mode()[0]] # selects relevant non ipv6 int(connection
        
        ## adding labels
        df['latency'] = int(run_labels[0])
        df['loss'] = int(run_labels[1])
        df['later_latency'] = int(run_labels[3])
        df['later_loss'] = int(run_labels[4])
        df['deterministic'] = bool(run_labels[2])
        
        #TODO future implementation of boolean flag for when the switch happens so we know when to use later lat/loss
        
        df.to_csv(f'{output_dir}/labeled_{temp_label_str}.csv') # save to temporary output directory: just merging takes a bit

    return 

def gen(cond , tempdir, subset):
    '''Generates transformed output data aggregated in 3 files.'''
    unseen = ""
    if cond == 'seen': 
        print("transforming seen data")
    if cond == 'unseen': 
        print("transforming un seen data")
        unseen = "unseen"
    
    tempdatafiles = 'data/temp/' + tempdir # temporary data directory for training model
    tempagg = 'data/temp' # temporary data directory for training model
    
    # path = os.path.join(os.getcwd() , "outputs", "gen_temp")
    # path2 = os.path.join(os.getcwd() , "outputs")
    fnames = [ filename for filename in listdir(tempdatafiles) if filename.endswith(".csv" ) ]
    
    data, datasubset, transformed  = [], [], []
    for j in fnames:
        loc = os.path.join(os.getcwd(), 'data', "temp/" + tempdir, j)
        #print(loc)
        df_cols = genfeat(pd.read_csv(loc))
        
        #data
        time_scaled = time__(df_cols)
        #print(time_scaled)
        data.append(time_scaled)
        
        #subset
        df_mid = time_scaled.iloc[60:60+subset]
        datasubset.append(df_mid)

        #transformed
        # f_df = agg10(df_cols)
        f_df = better_agg(df_cols)
        transformed.append(f_df)
        
    # makes paths
    path = os.path.join(os.getcwd() , "outputs", "gen_temp")
    path2 = os.path.join(os.getcwd() , "outputs")
    
    list_to_csv(data, os.path.join(path2, unseen + "combined_all.csv"))
    print('combined_finished', sep=' ')
    list_to_csv(datasubset, os.path.join(path2, unseen + "combined_subset_6068.csv"))
    print('combined_all_finished', sep=' ')
    list_to_csv(transformed, os.path.join(path2, unseen +  "combined_transform.csv"))
    print('transformed_finished', sep=' ')
    
    return None

def list_to_csv(lst, filepth):
    '''takes list of pandas dataframes with similar column structure and outputs them to a single folder'''
    lst[0].to_csv(filepth, index=False)
    for i in range(1, len(lst)):
        lst[i].to_csv(filepth, index=False, header=False, mode='a')
    return