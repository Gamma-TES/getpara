import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import libs.plt_config
import os
import libs.getpara as gp
import sys
import shutil
import pprint
import json
import re
from tkinter import filedialog 

def overlap(df_0,df_1):
    #index_0,index_1 = df_0.index.values,df_1.index.values
    ##ch1 = re.findall(r'\d+', index_1[0])[0]
    #num_0 = [re.findall(r'\d+', i)[2] for i in index_0]
    #num_1 = [re.findall(r'\d+', i)[2] for i in index_1]
    df_comp_0 = df_0[df_0.index.isin(df_1.index)]
    df_comp_1 = df_1[df_1.index.isin(df_0.index)]
    
    
    return df_comp_0,df_comp_1

def df_number(df):
    index= df.index.values
    num = [re.findall(r'\d+', i)[2] for i in index]
    df['number'] = num
    return df
    

def main():
    ax = sys.argv
    
    ch0 = 0
    ch1 = 1
    para = "height"
    set = gp.loadJson()
    os.chdir(set["Config"]["path"])

    output_0 = f'CH{ch0}_pulse/output/{set["Config"]["output"]}'
    output_1 = f'CH{ch1}_pulse/output/{set["Config"]["output"]}'
    
    df_0 =  pd.read_csv((f'{output_0}/output.csv'),index_col=0)
    df_1 =  pd.read_csv((f'{output_1}/output.csv'),index_col=0)



    df_0 = gp.select_condition(df_0,set)
    df_1 = gp.select_condition(df_1,set)

    df_0_over,df_1_over = overlap(df_0,df_1)
    x,y = df_0_over[para],df_1_over[para]

    plt.scatter(x,y,s=0.4)
    plt.xlabel(f'channel {ch0} [V]')
    plt.ylabel(f'channel {ch1} [V]')
    plt.title(para)
    plt.grid()
    plt.savefig(f'{output_0}/pulse_height_CH{ch0}_CH{ch1}.png')
    plt.show()
    



    
    path = filedialog.askopenfilename(filetypes=[('index file','*.txt')])

    selectdata = gp.loadIndex(path)
 
    df_select_0 = df_0_over[df_0_over.index.isin(selectdata)]
    df_select_1 = df_1_over[df_1_over.index.isin(selectdata)]
    
    x_select,y_select = df_select_0[para],df_select_1[para]
    print(x_select)
    plt.scatter(x,y,s=2,alpha=0.5)
    plt.scatter(x_select,y_select,s=2,alpha=1,color="red")
    plt.xlabel(f'channel {ch0}')
    plt.xlabel(f'channel {ch1}')
    plt.title(para)
    plt.grid()
    plt.show()
    plt.cla()
    #a = df.loc['CH0_pulse/rawdata\CH0_47388.dat']
    
    
    
    

#実行
if __name__=='__main__':
    main()