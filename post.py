import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp
import sys
import shutil
import pprint
import json
import re


ax_unit = {
    "base":'base[V]',
    "height":'pulse height[V]',
    "peak_index":'peak index',
    "height_opt":'pulse height opt',
    "height_opt_temp":'pulse height opt temp',
    'rise':'rise[s]',
    'decay':'decay[s]',
    'rise_fit':'rise_fit[s]',
    'tau_rise':'tau_rise[s]',
    'tau_decay':'tau_decay[s]',
    'rSquared':'rSquared'
}


def main():
    ax = sys.argv
    ax.pop(0)

    """
    path =  "E:/tsuruta/20230616/room1-ch2-3_180mK_570uA_100kHz_g10"
    os.chdir(path)
    ch0 =  pd.read_csv((f'CH0_pulse/output/1/output.csv'),index_col=0)
    ch1 =  pd.read_csv((f'CH1_pulse/output/1/output.csv'),index_col=0)
    
    x = ch0['height']
    y = ch1['height']
    print(len(x))
    print(len(y))

    plt.scatter(x,y,s=0.4)
    plt.grid()
    plt.show()
    """

    set = gp.loadJson()
    os.chdir(set["Config"]["path"])

    output_0 = f'CH{0}_pulse/output/{set["Config"]["output"]}'
    output_1 = f'CH{1}_pulse/output/{set["Config"]["output"]}'

    df_0 = pd.read_csv((f'{output_0}/output.csv'),index_col=0)
    df_1 = pd.read_csv((f'{output_1}/output.csv'),index_col=0)
    

    path = "E:/tsuruta/20230616/room1-ch2-3_180mK_570uA_100kHz_g10\CH0_pulse\output/1/1/selected_index.txt"
    ch =1
    selectdata = gp.loadIndex(path)
    numbers = []
    for i in selectdata:
        num = re.findall(r'\d+', i)[2]
        path_data = f"CH{ch}_pulse/rawdata\CH{ch}_{num}.dat"
        numbers.append(path_data)
    df_sel = df_1.loc[numbers]
    
    x,y = gp.extruct(df_0,*ax)
    x_select,y_select = gp.extruct(df_sel.loc[numbers],*ax)


    
    plt.scatter(x,y,s=2,alpha=0.5)
    plt.scatter(x_select,y_select,s=2,alpha=1)
    plt.xlabel(ax_unit[ax[0]])
    plt.ylabel(ax_unit[ax[1]])
    plt.title(f"{ax[0]} vs {ax[1]}")
    plt.grid()
    #plt.savefig(f'{output}/{ax[0]} vs {ax[1]}.png')
    plt.show()
    plt.cla()
    #a = df.loc['CH0_pulse/rawdata\CH0_47388.dat']
    
    

#実行
if __name__=='__main__':
    main()