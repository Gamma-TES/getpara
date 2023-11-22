
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
def main():

    # Mode option
    ax = sys.argv
    ax.append(0)
    
    setting = gp.loadJson()

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
    config = setting["Config"]
    analysis = setting["main"]
    path = config["path"]
    ch,rate,samples,presamples,threshold = \
        config["channel"],config['rate'],config['samples'],config["presamples"],config["threshold"]
    time = np.arange(0,1/rate*samples,1/rate)
    

    # analysis output path
    output = f'CH{ch}_pulse/output/{config["output"]}'

    # change directry
    os.chdir(path)

    #all data 
    path_data = natsorted(glob.glob(f'CH{ch}_pulse/rawdata/CH{ch}_*.dat'))

    # -- PoST Mode ---------------------------------

    # post mode use trigger channel data
    if post_mode:
        RED = '\033[33m'
        END = '\033[0m'
        print(RED+'PoST mode'+END)
        try:
            trig_ch = np.loadtxt('channel.txt')[:]
        except:
            trig_ch = np.zeros(len(path_data))

        post_ch = glob.glob('CH*_pulse')
        for i in post_ch:
            ch = int(re.sub(r"\D", "", i))
            print(f"CH{ch} triggered: {np.count_nonzero(trig_ch==ch)} count")
        

    # -- test mode ---------------------------------------------------------
    # only index datas
    if "index" in config:
        idx = gp.loadIndex(config["index"])
        path_data = [f'CH{ch}_pulse/rawdata/CH{ch}_{i}.dat' for i in idx]
    num_data = [re.findall(r'\d+', os.path.basename(i))[1] for i in path_data]
    print(f"{len(path_data)} pulses")

    # some lenght random data analysis mode
    if "-t" in ax:

        print("test mode")
        arr = np.arange(len(path_data))
        if not 'test_seed' in config:
            config['test_seed'] = int(input("seed: "))
            config['test_lenght'] = int(input('length: ')) # length of array
        
        np.random.seed(config['test_seed'])
        np.random.shuffle(arr) # shuffle array
        arr_select = arr[:config['test_lenght']]
        extruct = []
        for i in arr_select:
            path = f'CH{ch}_pulse/rawdata/CH{ch}_{i}.dat'
            extruct.append(path)
        path_data = natsorted(extruct)
        num_data = [re.findall(r'\d+', os.path.basename(i))[1] for i in path_data]
        ax.append('-a')
    # ----------------------------------------------------------------------

    else:
        print("\n")

    # --- all Samples -----------------------------------------------------
    if '-a' in ax:
        
        data_array = []
        cnt = 0
        for path in tqdm.tqdm(path_data):
            try:
                num =  re.findall(r'\d+', os.path.basename(path))[1]
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
                    #rise_fit = gp.risetime(fitting,np.max(fitting),np.argmax(fitting),rate)


                # low pass filter
                if analysis['cutoff'] > 0:
                    data = gp.BesselFilter(data,rate,analysis['cutoff'])
                
                # analysis
                peak,peak_av,peak_index = gp.peak(data,presamples,analysis['peak_max'],analysis['peak_x'],analysis['peak_w'])
                rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
                decay,decay_10,decay_90 = gp.decaytime(data,peak_av,peak_index,rate)
                
                # if fitting data, array has more parameter
                if fitting_mode:
                    data_column = [samples,base,peak_av,peak_index,rise,decay,trig,tau_rise,tau_decay,rSquared]
                else:
                    data_column = [samples,base,peak_av,peak_index,rise,decay,trig]
                data_array.append(data_column)
            except:
                print('error')
                num_data.remove(num)
                continue
            
        
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
    
    # -------------------------------------------------------------------------------------------------
        

    # --- one Sample --------------------------------------------------------------------------
    else:
        num = int(input('Enter pulse number: '))

        # post mode plot some datas in one figure


        if post_mode:
            trig = trig_ch[int(num)-1]
            print(f'trigger: {trig}')

            try:
                for i in post_ch:
                    ch = int(re.sub(r"\D", "", i))
                    path = f'{i}/rawdata/CH{ch}_{num}.dat'
                    data = gp.loadbi(path,config["type"])

                    if trig == ch:
                        analysis = setting["main"]
                    else:
                        analysis = setting["main2"]
                    
                    if analysis['cutoff'] > 0:
                        data = gp.BesselFilter(data,rate,analysis['cutoff'])
                    base,data = gp.baseline(data,presamples,analysis['base_x'],analysis['base_w'])

                    mv = gp.moving_average(data,analysis["mv_w"])
                    mv_padding = np.pad(mv,(int(analysis["mv_w"]/2-1),int(analysis["mv_w"]/2)),"constant")
                    dif = gp.diff(mv_padding)*50000
                    dif = gp.diff(dif)

                        
                    peak,peak_av,peak_index = gp.peak(data,presamples,analysis['peak_max'],analysis['peak_x'],analysis['peak_w'])
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
                        popt,rSquared = gp.fitExp(gp.fit_func(analysis['fit_func']),data,presamples+analysis['fit_x'],analysis['fit_w'],p0 = analysis['fit_p0'])
                        if analysis['fit_func'] == "monoExp":
                            tau_rise = 0
                            tau_decay = 1/popt[1]/rate
                        if analysis['fit_func'] == "doubleExp":
                            tau_rise = popt[1]/rate
                            tau_decay = popt[3]/rate
                        fit_func = gp.fit_func(analysis["fit_func"])
                        fitting = fit_func(x_fit,*popt)

                        print(f'tau_rise : {tau_rise:.5f}' )
                        print(f'tau_decay : {tau_decay:.5f}' )
                        print(f'rSqared: {rSquared:.5f}')

                    
                    if analysis['cutoff'] > 0:
                        plt.plot(time,data,markersize=1,label=f"rawdata_ch{ch} filt")
                    else:
                        plt.plot(time,data,markersize=1,label=f"rawdata_ch{ch}")
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

                gp.graugh_condition(setting["graugh"])
                plt.xlabel("time(s)")
                plt.ylabel("volt(V)")
                plt.grid()
                plt.title(f'rawdata {num}.')
                plt.show()
                plt.cla()
            except Exception as e:
                print(e)

            
            
            #gp.graugh('diff pulse',dif,time[:samples-set['main']['mv_w']+1])
            #plt.show()

        
        else:
            # Processing
            path = f'CH{ch}_pulse/rawdata/CH{ch}_{num}.dat'
            
            data = gp.loadbi(path,config["type"])
            base,data = gp.baseline(data,presamples,analysis['base_x'],analysis['base_w'])
            mv = gp.moving_average(data,analysis['mv_w'])
            dif  = gp.diff(mv)

            if analysis['cutoff'] > 0:
                data = gp.BesselFilter(data,rate,analysis['cutoff'])

            # Analysis
            peak,peak_av,peak_index = gp.peak(data,presamples,analysis['peak_max'],analysis['peak_x'],analysis['peak_w'])
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
                popt,rSquared = gp.fitExp(gp.fit_func(analysis['fit_func']),data,presamples+analysis['fit_x'],analysis['fit_w'],p0 = analysis['fit_p0'])
                if analysis['fit_func'] == "monoExp":
                    tau_rise = 0
                    tau_decay = 1/popt[1]/rate
                if analysis['fit_func'] == "doubleExp":
                    tau_rise = popt[1]/rate
                    tau_decay = popt[3]/rate
                fit_func = gp.fit_func(analysis["fit_func"])
                fitting = fit_func(x_fit,*popt)

                print(f'tau_rise : {tau_rise:.5f}' )
                print(f'tau_decay : {tau_decay:.5f}' )
                print(f'rSqared: {rSquared:.5f}')


            # Graugh
            if analysis['cutoff'] > 0:
                plt.plot(time,data,'o',markersize=1,label=f"rawdata_ch{ch} data")
            else:
                plt.plot(time,data,'o',markersize=1,label=f"rawdata_ch{ch}")
            if fitting_mode:
                plt.plot(x_fit/rate,fitting,'-.',color="skyblue",label = 'fitting')
            #plt.plot(time[:samples-set['main']['mv_w']+1],mv,label = "mv")
            plt.plot(time[presamples-analysis['base_x']:presamples-analysis['base_x']+analysis['base_w']],data[presamples-analysis['base_x']:presamples-analysis['base_x']+analysis['base_w']],'-',linewidth=2,color="lightpink",label="base")
            plt.plot(time[rise_10:rise_10],data[rise_10:rise_10],'-',color='royalblue',label="rise")
            plt.scatter(time[rise_10],data[rise_10],color = 'royalblue',label='rise')
            plt.scatter(time[rise_90],data[rise_90],color = 'royalblue')
            plt.scatter(time[peak_index],peak_av, color='orange', label ='peak',zorder = 2)
            plt.scatter(time[presamples],data[presamples], color='violet', label ='risepoint',zorder = 2)
        
            plt.xlabel("time(s)")
            plt.ylabel("volt(V)")
            gp.graugh_condition(setting["graugh"])
            plt.title(os.path.basename(path).replace('.dat',''))
            plt.grid()
            plt.legend()
            plt.show()
            plt.cla()

            if setting['graugh']['diff']:
                gp.graugh('diff pulse',dif,time[:samples-analysis['mv_w']+1])
                plt.show()
            
    gp.saveJson(setting,path=output)

if __name__ == "__main__":
    main()
    print("end")
    print('---------------------------\n')

