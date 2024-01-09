
#----- analysis code---------------------
#
# MacOs Montray, Python3.10.8
#
# -----user guide ---------------
# --- option
# ------ "-a": all analysis mode
#           analysis all data in CH*_pulse/rawdata/CH*_****.dat
# ------ "-t": test mode
#           some random data analysis. enter lenght of what you want to analysis sample and random seed
# ------ "-p": post mode
#           analysis some channel data at same time (This mode is not analysis all data of each channe becouse it's faster to run sapalately than with same task.)
# ex) python main.py -p -a

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
from natsort import natsorted
import getpara as gp
import fft_spectrum as sp
import json
import pprint
import sys
import re
import matplotlib.cm as cm
import plt_config
import tqdm



# run
def main():

    # Mode option
    ax = sys.argv
    setting = gp.loadJson()

    #  post mode
    if '-p' in ax:
        post_mode = 1
    else:
        post_mode = 0

    # fitting mode
    if '-f' in ax:
        fitting_mode = 1
    else:
        fitting_mode = 0

    # get setting from setting.json
    config = setting["Config"]
    path = config["path"]
    channel,rate,samples,presamples,threshold = \
        config["channel"],config['rate'],config['samples'],config["presamples"],config["threshold"]
    time = np.arange(0,1/rate*samples,1/rate)
    
     # change directry
    os.chdir(path)

    # analysis output path
    output = f'CH{channel}_pulse/output/{config["output"]}'

    plot = 1

    # -- PoST Mode ------------------------------------------------------

    # post mode use trigger channel data
    if post_mode:
        RED = '\033[33m'
        END = '\033[0m'
        print(RED+'PoST mode'+END)

        try:
            trig_ch = np.loadtxt('channel.txt')[:]
        except:
            trig_ch = np.zeros(len(path_data))

        channels_path = glob.glob('CH*_pulse')
        channels = [re.findall(r'\d+',i)[0] for i in channels_path]
        for i in channels_path:
            ch = int(re.sub(r"\D", "", i))
            print(f"CH{ch} triggered: {np.count_nonzero(trig_ch==ch)} count")
    else:
        channels= [channel]

    # -- test mode ---------------------------------------------------------
            
    # only index datas
    if "index" in config:
        idx = gp.loadIndex(config["index"])
        path_data = [f'CH{channel}_pulse/rawdata/CH{channel}_{i}.dat' for i in idx]
        print(f"{len(path_data)} pulses")

    # some lenght random data analysis mode
    if "-t" in ax:
        print("test mode")
        path_data = glob.glob(f'CH{channel}_pulse/rawdata/CH{channel}_*.dat')
        num_data = [re.findall(r'\d+', os.path.basename(i))[1] for i in path_data]
        arr = np.array(num_data)
        if not 'test_seed' in config:
            config['test_seed'] = int(input("seed: "))
            config['test_lenght'] = int(input('length: ')) # length of array
        
        np.random.seed(config['test_seed'])
        np.random.shuffle(arr) # shuffle array
        arr_select = arr[:config['test_lenght']]
        extruct = []
        for i in arr_select:
            path = f'CH{channel}_pulse/rawdata/CH{channel}_{i}.dat'
            extruct.append(path)
        path_data = natsorted(extruct)
        plot = 0
    # ----------------------------------------------------------------------
    
     # -- all data ----------------------------------------------------------
    if '-a' in ax:
        path_data = natsorted(glob.glob(f'CH{channel}_pulse/rawdata/CH{channel}_*.dat'))
        plot = 0
    
    # -- single ----------------------------------------------
    if plot:
        number= input(f'data number: ')
        path_data = [f'CH{channel}_pulse/rawdata/CH{channel}_{number}.dat']

    num_data = [re.findall(r'\d+', os.path.basename(i))[1] for i in path_data]
    print("\n")


    # --- all Samples -----------------------------------------------------

    data_array = []
    cnt = 0
    for num in tqdm.tqdm(num_data):
        for ch in channels:
            try:
                path = f'CH{ch}_pulse/rawdata/CH{ch}_{num}.dat'
                data = gp.loadbi(path,config["type"])
                # choose triggerd channel

                if post_mode:
                    trig = trig_ch[int(num)-1]
                    if trig == int(ch):
                        analysis = setting["main"]
                    else:
                        analysis = setting["main2"]
                else:
                    analysis = setting["main"]
                    trig = int(ch)

                # base line calibration
                base,data = gp.baseline(data,presamples,analysis['base_x'],analysis['base_w'])

                # fitting
                if fitting_mode:
                    x_fit = np.arange(presamples-5,samples,0.1)
                    popt,rSquared = gp.fitExp(gp.fit_func(analysis['fit_func']),data,presamples+analysis['fit_x'],analysis['fit_w'],p0 = analysis['fit_p0'])
                    if analysis['fit_func'] == "monoExp":
                        tau_rise = 0
                        tau_decay = 1/popt[1]/rate
                    if analysis['fit_func'] == "doubleExp":
                        tau_rise = popt[1]/rate
                        tau_decay = popt[3]/rate
                    fit_func = gp.fit_func(analysis["fit_func"])
                    if plot:
                        fitting = fit_func(x_fit,*popt)


                # low pass filter
                if analysis['cutoff'] > 0:
                    data = gp.BesselFilter(data,rate,analysis['cutoff'])
                
                # analysis
                peak,peak_av,peak_index = gp.peak(data,presamples,analysis['peak_max'],analysis['peak_x'],analysis['peak_w'])
                rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
                decay,decay_10,decay_90 = gp.decaytime(data,peak_av,peak_index,rate)
                area = gp.area(data,peak_index,analysis['area_x'],analysis['area_w'])
                
                
                # if fitting data, array has more parameter
                if fitting_mode:
                    data_column = [samples,base,peak_av,peak_index,rise,decay,trig,tau_rise,tau_decay,rSquared]
                else:
                    data_column = [samples,base,peak_av,peak_index,rise,decay,trig]
                
                if int(ch) == channel:
                    data_array.append(data_column)
                    

                if plot:
                    print('\n---------------------------')
                    print(path)
                    print(f'samples : {len(data)}')
                    print(f'base : {base:.5f}')
                    print(f'hight : {peak_av:.5f}')
                    print(f'peak index : {peak_index:.5f}')
                    print(f'rise : {rise:.5f}')
                    print(f'decay : {decay:.5f}')
                    print(f'area : {area:.5f}')

                    if analysis['cutoff'] > 0:
                        plt.plot(time,data,markersize=1,label=f"rawdata_ch{ch} filt")
                    else:
                        plt.plot(time,data,markersize=1,label=f"rawdata_ch{ch}")
                    if fitting_mode:
                        plt.plot(x_fit/rate,fitting,'-.',label = 'fitting')
                    #plt.plot(time,mv_padding,'o',markersize=1,label=f"mv_ch{ch} mv")
                    #plt.plot(time,dif,'o',markersize=1,label=f"difdif_ch{ch}")
                    plt.plot(time[presamples-analysis['area_x']:presamples-analysis['area_x']+analysis['area_w']],
                                data[presamples-analysis['area_x']:presamples-analysis['area_x']+analysis['area_w']],
                                '-',color='green',label="area",zorder = 2)
                    plt.plot(time[rise_10:rise_10],data[rise_10:rise_10],'-.',color='royalblue',label="rise")
                    plt.scatter(time[rise_10],data[rise_10],color = 'blue',label='rise',zorder = 3)
                    plt.scatter(time[rise_90],data[rise_90],color = 'blue',zorder = 3)
                    plt.scatter(time[peak_index],peak_av, color='cyan', label ='peak',zorder = 3)
                    plt.scatter(time[presamples],data[presamples], color='magenta', label ='risepoint',zorder = 3)

            
            except Exception as e:
                RED = '\033[31m'
                END = '\033[0m'
                print(RED+f'{e}'+END)
                num_data.remove(num)
                continue
    
    if plot:
        plt.xlabel("time(s)")
        plt.ylabel("volt(V)")
        gp.graugh_condition(setting["graugh"])
        plt.title(os.path.basename(path))
        plt.grid()
        #plt.legend(label,fontsize=10,loc='center right',bbox_to_anchor=(1., .5))
        plt.tight_layout()
        plt.show()
        
    # create pandas DataFrame
    if fitting_mode:
        columns=["samples","base","height","peak_index","rise","decay","trig",'tau_rise','tau_decay',"rSquared"]
    else:      
        columns=["samples","base","height","peak_index","rise","decay","trig"]
    
    if not plot:
        print('save')
        df = pd.DataFrame(data_array,columns=columns,index=num_data)
        df.to_csv(os.path.join(output,"output.csv"))
    
    # -------------------------------------------------------------------------------------------------
    gp.saveJson(setting,path=output)

if __name__ == "__main__":
    main()
    print("end")
    print('---------------------------\n')

