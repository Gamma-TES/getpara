import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp
import sys
import shutil
import pprint
import json
import glob
import re
from tkinter import filedialog

BUF = 100

def main():
    ax = sys.argv
    set = gp.loadJson()

    os.chdir(set["Config"]["path"])


    output = f'CH{set["Config"]["channel"]}_pulse/output/{set["Config"]["output"]}'
    df = pd.read_csv((f'{output}/output.csv'),index_col=0)
    rate,samples,presamples,threshold,ch = set["Config"]["rate"],set["Config"]["samples"],set["Config"]["presamples"],set["Config"]["threshold"],set["Config"]["channel"]
    time = gp.data_time(rate,samples)

    start = int(input('start: '))
    buf = df.iloc[start:BUF].index.values
    
    df_sel = gp.select_condition(df,set)

    num = 1
    while num != 0:
        os.makedirs(f"{output}/img",exist_ok=True)
        buf = df_sel.iloc[start:start+BUF].index.values
        print("loading...")
        for i in buf:
            data = gp.loadbi(i)
            data = gp.BesselFilter(data,rate,fs = set['main']['cutoff'])
            base,data = gp.baseline(data,presamples,1000,500)
            name = os.path.splitext(os.path.basename(i))[0]
            plt.plot(time,data)
            plt.title(name)
            plt.xlabel("time (s)")
            plt.ylabel("volt (V)")
            plt.savefig(f'{output}/img/{name}.png')
            plt.cla()
        
        fle = filedialog.askopenfilenames(filetypes=[('画像ファイル','*.png')],initialdir=f"{output}/img")
        for f in fle:
            num =  re.findall(r'\d+', os.path.basename(f))[1]
            index = f"CH{ch}_pulse/rawdata\CH{ch}_{num}.dat"
            df.at[index,"quality"] = 0

        shutil.rmtree(f"{output}/img")
        try:
            num = int(input('finish? [0]'))
        except:
            start  += BUF
            continue
        

    df.to_csv(f'{output}/output.csv')
   
    
        
        
    

    """
    num = 1
    while num != 0:
                
                try:
                    picked.remove(f'CH{ch}_pulse/rawdata\\CH{ch}_{num}.dat')
                    os.remove(f'{output_f}/img/CH{ch}_{num}.png')
                    np.savetxt(f'{output_f}/selected_index.txt',picked,fmt="%s")
                    index = f"CH{ch}_pulse/rawdata\CH{ch}_{num}.dat"
                    df.at[index,"quality"] = 0
                except:
                    print("Not exist file")
    """
    

    """
    # delete noise data
    fle = filedialog.askopenfilenames(filetypes=[('画像ファイル','*.png')],initialdir=f"{output_f}/img")
    for f in fle:
        num =  re.findall(r'\d+', os.path.basename(f))[1]
        picked.remove(f'CH{ch}_pulse/rawdata\\CH{ch}_{num}.dat')
        os.remove(f'{output_f}/img/CH{ch}_{num}.png')
        np.savetxt(f'{output_f}/selected_index.txt',picked,fmt="%s")
        index = f"CH{ch}_pulse/rawdata\CH{ch}_{num}.dat"
        df.at[index,"quality"] = 0
    """

if __name__=='__main__':
    main()

    