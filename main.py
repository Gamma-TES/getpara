
#----- analysis code---------------------
#
# MacOs Montray, Python3.10.8
#
# -----user guide ---------------
# input data number 
# plot data
# --- option
# ------ "-a": all analysis mode
#           analysis all data in CH*_pulse/rawdata/CH*_****.dat
# ------ "-t": test mode
#           some random data analysis. enter lenght of what you want to analysis sample
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
if __name__ == '__main__':

    # Mode option
    ax = sys.argv
    ax.append(0)
    
    # add main option at setting.json
    with open("setting.json") as f:
        set = json.load(f)

    
    # if post mode, add main2 option
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
    set_config = set["Config"]
    set_main = set["main"]
    path = set_config["path"]
    ch,rate,samples,presamples,threshold = \
        set_config["channel"],set_config['rate'],set_config['samples'],set_config["presamples"],set_config["threshold"]
    time = np.arange(0,1/rate*samples,1/rate)
    

    # analysis output path
    output = f'CH{ch}_pulse/output/{set_config["output"]}'

    # change directry
    os.chdir(path)


    # -- PoST Mode setting----------------
    # post mode use trigger channel data
    if '-p' in ax:
        RED = '\033[33m'
        END = '\033[0m'
        print(RED+'PoST mode'+END)
        trig_ch = np.loadtxt('channel.txt')
        post_ch = glob.glob('CH*_pulse')
        for i in post_ch:
            ch = int(re.sub(r"\D", "", i))
            print(f"CH{ch} triggered: {np.count_nonzero(trig_ch==ch)} count")
        
        

    # -- test mode ------------------------
    # some lenght random data analysis mode
    if "-t" in ax:
        print("test mode")
        path_raw = natsorted(glob.glob(f'CH{ch}_pulse/rawdata/CH{ch}_*.dat'))
        arr = np.arange(len(path_raw))
        seed = int(input("seed: "))
        np.random.seed(seed)
        set['main']['seed'] = seed
        np.random.shuffle(arr) # shuffle array
        length = int(input('length: ')) # length of array
        arr_select = arr[:length]
        extruct = []
        for i in arr_select:
            path = f'CH{ch}_pulse/rawdata/CH{ch}_{i}.dat'
            extruct.append(path)
        path_data = natsorted(extruct)
        num_data = [re.findall(r'\d+', os.path.basename(i))[1] for i in path_data]
        ax.append('-a')
    # --------------------------------------

    # -- product mode ----------------------
    else:
        path_data = natsorted(glob.glob(f'CH{ch}_pulse/rawdata/CH{ch}_*.dat'))
        num_data = [re.findall(r'\d+', os.path.basename(i))[1] for i in path_data]
        print(f"{len(path_data)} pulses")


    jsn = json.dumps(set,indent=4)
    with open(f'{output}/setting.json', 'w') as file:
        file.write(jsn)

    # --- all Samples -----------------------------------------------------
    if '-a' in ax:

        data_array = []
        cnt = 0
        for path in tqdm.tqdm(path_data):
            num =  re.findall(r'\d+', os.path.basename(path))[1]
            data = gp.loadbi(path)

            if post_mode:
                trig = trig_ch[int(num)-1]
                if trig == int(ch):
                    set_main = set["main"]
                else:
                    set_main = set["main2"]
            else:
                set_main = set["main"]
                trig = int(ch)

            base,data = gp.baseline(data,presamples,set_main['base_x'],set_main['base_w'])

            # fitting
            if fitting_mode:

                x_fit = np.arange(presamples-5,samples,0.1)
                popt,rSquared = gp.fitExp(gp.fit_func(set_main['fit_func']),data,presamples+set_main['fit_x'],set_main['fit_w'],p0 = set_main['fit_p0'])
                if set_main['fit_func'] == "monoExp":
                    tau_rise = 0
                    tau_decay = 1/popt[1]/rate
                if set_main['fit_func'] == "doubleExp":
                    tau_rise = popt[1]/rate
                    tau_decay = popt[3]/rate
                fit_func = gp.fit_func(set_main["fit_func"])
                #rise_fit = gp.risetime(fitting,np.max(fitting),np.argmax(fitting),rate)


            # low pass filter
            if set_main['cutoff'] > 0:
                data = gp.BesselFilter(data,rate,set_main['cutoff'])
            
            # analysis
            peak,peak_av,peak_index = gp.peak(data,presamples,set_main['peak_max'],set_main['peak_x'],set_main['peak_w'])
            rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
            decay,decay_10,decay_90 = gp.decaytime(data,peak_av,peak_index,rate)
            
            # if fitting data, array has more parameter
            if fitting_mode:
                data_column = [samples,base,peak_av,peak_index,rise,decay,trig,tau_rise,tau_decay,rSquared]
            else:
                data_column = [samples,base,peak_av,peak_index,rise,decay,trig]
            data_array.append(data_column)
            
        
        # create pandas DataFrame
        if fitting_mode:
            df = pd.DataFrame(data_array,\
                    columns=["samples","base","height","peak_index","rise","decay","trig",'tau_rise','tau_decay',"rSquared"],\
                    index=num_data)

        else:
            df = pd.DataFrame(data_array,\
                    columns=["samples","base","height","peak_index","rise","decay","trig"],\
                    index=num_data)

        # output
        df.to_csv(os.path.join(output,"output.csv"))
        
        print('end\n-------------------------------------')
    # -------------------------------------------------------------------------------------------------
        

    # --- one Sample --------------------------------------------------------------------------
    else:
        num = int(input('Enter pulse number: '))

        # post mode plot some datas in one figure


        if post_mode:
            trig = trig_ch[int(num)-1]
            print(f'trigger: {trig}')

            
            for i in post_ch:
                ch = int(re.sub(r"\D", "", i))
                path = f'{i}/rawdata/CH{ch}_{num}.dat'
                data = gp.loadbi(path)

                if trig == ch:
                    set_main = set["main"]
                else:
                    set_main = set["main2"]

                
                if set_main['cutoff'] > 0:
                    data = gp.BesselFilter(data,rate,set_main['cutoff'])
                base,data = gp.baseline(data,presamples,set_main['base_x'],set_main['base_w'])

                mv = gp.moving_average(data,set_main["mv_w"])
                mv_padding = np.pad(mv,(int(set_main["mv_w"]/2-1),int(set_main["mv_w"]/2)),"constant")
                dif = gp.diff(mv_padding)*50000
                dif = gp.diff(dif)

                    
                peak,peak_av,peak_index = gp.peak(data,presamples,set_main['peak_max'],set_main['peak_x'],set_main['peak_w'])
                rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
                decay,decay_10,decay_90 = gp.decaytime(data,peak_av,peak_index,rate)

                print('\n---------------------------')
                print(path)
                print(f'samples : {len(data)}')
                print(f'base : {base:.5f}')
                print(f'hight : {peak_av:.5f}')
                print(f'peak index : {peak_index:.5f}')
                print(f'rise : {rise:.5f}')
                print(f'decay : {decay:.5f}')

                if fitting_mode:
                    x_fit = np.arange(presamples-5,samples,0.1)
                    popt,rSquared = gp.fitExp(gp.fit_func(set_main['fit_func']),data,presamples+set_main['fit_x'],set_main['fit_w'],p0 = set_main['fit_p0'])
                    if set_main['fit_func'] == "monoExp":
                        tau_rise = 0
                        tau_decay = 1/popt[1]/rate
                    if set_main['fit_func'] == "doubleExp":
                        tau_rise = popt[1]/rate
                        tau_decay = popt[3]/rate
                    fit_func = gp.fit_func(set_main["fit_func"])
                    fitting = fit_func(x_fit,*popt)

                    print(f'tau_rise : {tau_rise:.5f}' )
                    print(f'tau_decay : {tau_decay:.5f}' )
                    print(f'rSqared: {rSquared:.5f}')

                
                if set_main['cutoff'] > 0:
                    plt.plot(time,data,'o',markersize=1,label=f"rawdata_ch{ch} filt")
                else:
                    plt.plot(time,data,'o',markersize=1,label=f"rawdata_ch{ch}")
                if fitting_mode:
                    plt.plot(x_fit/rate,fitting,'-.',label = 'fitting')
                #plt.plot(time,mv_padding,'o',markersize=1,label=f"mv_ch{ch} mv")
                #plt.plot(time,dif,'o',markersize=1,label=f"difdif_ch{ch}")
                plt.plot(time[rise_10:rise_10],data[rise_10:rise_10],'-.',color='royalblue',label="rise")
                plt.scatter(time[rise_10],data[rise_10],color = 'blue',label='rise')
                plt.scatter(time[rise_90],data[rise_90],color = 'blue')
                plt.scatter(time[peak_index],peak_av, color='cyan', label ='peak',zorder = 2)
                plt.scatter(time[presamples],data[presamples], color='magenta', label ='risepoint',zorder = 2)
                plt.legend()

            gp.graugh_condition(set)
            plt.xlabel("time(s)")
            plt.ylabel("volt(V)")
            plt.grid()
            plt.title(f'rawdata {num}.')
            plt.show()
            plt.cla()

            
            
            #gp.graugh('diff pulse',dif,time[:samples-set['main']['mv_w']+1])
            #plt.show()

        
        else:
            # Processing
            path = f'CH{ch}_pulse/rawdata/CH{ch}_{num}.dat'
            data = gp.loadbi(path)
            base,data = gp.baseline(data,presamples,set_main['base_x'],set_main['base_w'])
            mv = gp.moving_average(data,set_main['mv_w'])
            dif  = gp.diff(mv)

            if set_main['cutoff'] > 0:
                data = gp.BesselFilter(data,rate,set_main['cutoff'])

            # Analysis
            peak,peak_av,peak_index = gp.peak(data,presamples,set_main['peak_max'],set_main['peak_x'],set_main['peak_w'])
            rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
            decay,decay_10,decay_90 = gp.decaytime(data,peak_av,peak_index,rate)

            # Console output
            print('\n---------------------------')
            print(path)
            print(f'samples : {len(data)}')
            print(f'base : {base:.5f}')
            print(f'hight : {peak_av:.5f}')
            print(f'peak index : {peak_index:.5f}')
            print(f'rise : {rise:.5f}')
            print(f'decay : {decay:.5f}')
            
            # Fitting
            if fitting_mode:
                x_fit = np.arange(presamples-5,samples,0.1)
                popt,rSquared = gp.fitExp(gp.fit_func(set_main['fit_func']),data,presamples+set_main['fit_x'],set_main['fit_w'],p0 = set_main['fit_p0'])
                if set_main['fit_func'] == "monoExp":
                    tau_rise = 0
                    tau_decay = 1/popt[1]/rate
                if set_main['fit_func'] == "doubleExp":
                    tau_rise = popt[1]/rate
                    tau_decay = popt[3]/rate
                fit_func = gp.fit_func(set_main["fit_func"])
                fitting = fit_func(x_fit,*popt)

                print(f'tau_rise : {tau_rise:.5f}' )
                print(f'tau_decay : {tau_decay:.5f}' )
                print(f'rSqared: {rSquared:.5f}')


            # Graugh
            if set_main['cutoff'] > 0:
                plt.plot(time,data,'o',markersize=1,label=f"rawdata_ch{ch} data")
            else:
                plt.plot(time,data,'o',markersize=1,label=f"rawdata_ch{ch}")
            if fitting_mode:
                plt.plot(x_fit/rate,fitting,'-.',color="skyblue",label = 'fitting')
            #plt.plot(time[:samples-set['main']['mv_w']+1],mv,label = "mv")
            plt.plot(time[presamples-set_main['base_x']:presamples-set_main['base_x']+set_main['base_w']],data[presamples-set_main['base_x']:presamples-set_main['base_x']+set_main['base_w']],'-',linewidth=2,color="lightpink",label="base")
            plt.plot(time[rise_10:rise_10],data[rise_10:rise_10],'-',color='royalblue',label="rise")
            plt.scatter(time[rise_10],data[rise_10],color = 'royalblue',label='rise')
            plt.scatter(time[rise_90],data[rise_90],color = 'royalblue')
            plt.scatter(time[peak_index],peak_av, color='orange', label ='peak',zorder = 2)
            plt.scatter(time[presamples],data[presamples], color='violet', label ='risepoint',zorder = 2)
        
            plt.xlabel("time(s)")
            plt.ylabel("volt(V)")
            gp.graugh_condition(set)
            plt.title(os.path.basename(path).replace('.dat',''))
            plt.grid()
            plt.legend()
            plt.show()
            plt.cla()


            gp.graugh('diff pulse',dif,time[:samples-set['main']['mv_w']+1])
            plt.show()
            

        print("end")
        print('---------------------------\n')

