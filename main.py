#　-----解析プログラム---------------------
#　
#  動作環境：MacOs Montray, Python3.10.8
#
#　手順
# 1.パスを変更
#   LabVIEWで作成したファイルパス。（おそらく"CH1_150mK_710uA_..."と長い。できるだけ絶対パスで）
# 2.パラメータ変更
#   ①はLabVIEWでデータ取得した時のパラメータ。新しいものだとsetting.datができているので、それを読み込めば良い。
#   ②はデータ解析用のパラメータ。このパラメータによって解析の結果は大きく変わるため慎重に調整する。詳細は下述。
# 3.実行
#   Chanel:[解析するチャンネル](半角数字)
#   Analysis Mode:[0 or 1](半角数字)
#       全てのパルスを解析したいときは０、任意の1つのパルスを解析したいときは１を入力。
#       初めはパラメータ調整のために１を実行することを推奨。
#   Mode 0 の場合
#       全てのデータを解析。中断するときは[Control+z]
#       解析終了後、outoutフォルダを作成し、中に結果のoutput.csvが作成されている。
#       output.csvを用いて次のanalysis.pyを実行できる。
#   Mode 1 の場合
#       Enter pulse number:[詳細を見たいパルス番号]
#       Enterを押した後、解析後のデータの詳細とグラフが出力
#   


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
from natsort import natsorted
import libs.getpara as gp
import libs.fft_spectrum as sp
import json


#-----Analtsis Parameter------------------------------------------------
cf = 4e4        # Low Pass Filter cut off

x_ba = 1000     # base start: presamples - x_ba
w_ba = 500      # base width: presamples - x_ba + w_ba
w_max = 300     # peak width: presamples + w_max
x_av = 5        # peak_av start: peak_index - x_av
w_av = 20       # peak_av width: peak_index - x_av + w_av

# Mono Fitting
start_fit = 100     # fitting start: peak_index(average) + x_fit
width_fit = 800     # fitting width: peak_index(average) + x_fit + w_fit

# Double Fitting
start = -1      # presamples + start
width = 1000    #  start + width

mv_av = 100     # 移動平均の幅
decay_sillicon = 0.001 # コンイベント判別の為のディケイ
#-----------------------------------------------------------------

def monoExp(x,m,t):
    return m*np.exp(-t*x)

#実行
if __name__ == '__main__':

    # Get Setting
    set = gp.loadJson()
    path = set["Config"]["path"]
    ch,rate,samples,presamples,threshold = \
        set["Config"]["channel"],set["Config"]['rate'],set["Config"]['samples'],set["Config"]["presamples"],set["Config"]["threshold"]
    time = np.arange(0,1/rate*samples,1/rate)
    os.chdir(path) 

    path = natsorted(glob.glob(f'CH{ch}_pulse/rawdata/CH{ch}_*.dat'))
    #path = natsorted(glob.glob(f'CH{ch}_pulse/test/CH{ch}_*.dat'))

    mode = input('Analysis Mode (all -> [0], one -> [1]): ')

    # All Samples
    if mode == '0':
        fit = input('fitting? [1]): ')
        lpf = input('Low Pass Filter? [1]): ')
        data_array = []
        for num in path:
            print(os.path.basename(num))
            data = gp.loadbi(num)
            base,data = gp.baseline(data,presamples,x_ba,w_ba)
            if lpf == "1":
                filt = gp.BesselFilter(data,rate,cf)
            peak,peak_av,peak_index = gp.peak(data,presamples,w_max,x_av,w_av)
            rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
            
            decay = gp.decaytime(data,peak_av,peak_index,rate)

            # not average for silicon event
            if decay < decay_sillicon:
                peak_av = peak
                peak_mv_av = peak

            if fit == '1':

                p0 = [-1.6,12,presamples,5570,presamples]
                x_fit = np.arange(presamples-5,samples,0.1)
                popt,rSquared = gp.fitting_double(data,presamples,start,width,p0)
                tau_rise = popt[1]
                tau_decay = popt[3]
                fit_double = gp.doubleExp(x_fit,*popt)

                rise_fit = gp.risetime(fit_double,np.max(fit_double),np.argmax(fit_double),rate)
                data_column = [samples,base,peak_av,rise,decay,rise_fit[0],tau_rise,tau_decay,rSquared]
                
            else:
                data_column = [samples,base,peak_av,rise,decay]
            
            data_array.append(data_column)

        # create pandas DataFrame
        if fit == "1":
            df = pd.DataFrame(data_array,\
            columns=["samples","base","height","rise","decay","rise_fit",'tau_rise','tau_decay',"rSquared"],\
            index=path)
        else:
            df = pd.DataFrame(data_array,\
            columns=["samples","base","height","rise","decay"],\
            index=path)

        # output
        output = (f'CH{ch}_pulse/output')
        if not os.path.exists(output):
            os.makedirs(output,exist_ok=True)
            df.to_csv(os.path.join(output,"output.csv"))
        else:
            replace = input('Replace output folder? (Yes -> [0], No (not save) -> [1]): ')
            if replace =='0':
                shutil.rmtree(output)
                os.makedirs(output,exist_ok=True)
                df.to_csv(os.path.join(output,"output.csv"))
        print('end\n-------------------------------------') 
        

        # One Sample
    elif mode == '1':
        num = input('Enter pulse number: ')
        path = f'CH{ch}_pulse/rawdata/CH{ch}_{num}.dat'
        data = gp.loadbi(path)

        # Processing
        base,data = gp.baseline(data,presamples,x_ba,w_ba)
        mv = gp.moving_average(data,mv_av)
        dif  = gp.diff(mv)
        filt = gp.BesselFilter(data,rate,cf)

        # Analysis
        peak,peak_av,peak_index = gp.peak(filt,presamples,w_max,x_av,w_av)
        rise,rise_10,rise_90 = gp.risetime(filt,peak_av,peak_index,rate)
        decay = gp.decaytime(filt,peak_av,peak_index,rate)

        # Fitting
        m,t,rSquared,max_div = gp.fitting(data,peak_index,time,start_fit,width_fit,mode,p0 = [-14,70])
        p0 = [-1.6,12,presamples,5570,presamples]
        x_fit = np.arange(presamples-5,samples,0.1)
        popt,rSquared_double = gp.fitting_double(data,presamples,start,width,p0)
        fit_double = gp.doubleExp(x_fit,*popt)
        peak_fit_index = np.argmax(fit_double)
        
        rise_fit = gp.risetime(fit_double,np.max(fit_double),np.argmax(fit_double),rate)
        

        # not average for Silicon Event
        if decay < decay_sillicon:
            peak_av = peak
            print("Seilicon Event")
        else:
            print('Absorver Event')

        # Console output
        print('\n---------------------------')
        print(path)
        print(f'samples : {len(data)}')
        print(f'base : {base:.5f}')
        print(f'hight : {peak_av:.5f}')
        print(f'peak index : {peak_index:.5f}')
        print(f'rise : {rise}')
        print(f'rise fit : {rise_fit[0]}')
        print(f'decay : {decay}')
        print(f'tauSec : {1/t}')
        print('y = {0:.2f}[e^((x-{2:.2f})/{1:.2f}) + e^((x-{4:.2f})/{3:.2f})]'.format(*popt))
        print(f"rSquared = {rSquared:.5f}")
        print(f"max_div = {max_div:.5f}")

        # Graugh
        plt.plot(time,data,'o',markersize=1,label="rawdata")
        plt.plot(time,filt,'o',markersize=1,label = "filt")
        plt.plot(time[presamples-x_ba:presamples-x_ba+w_ba],filt[presamples-x_ba:presamples-x_ba+w_ba],'--',label="base")
        plt.plot(x_fit/rate,fit_double,'-.',label = 'fitting')

        plt.vlines(time[rise_10],ymin=0,ymax=filt[rise_10],color = 'black',linestyle='-.')
        plt.vlines(time[rise_90],ymin=0,ymax=filt[rise_90],color = 'black',linestyle='-.')
        plt.scatter(time[rise_10],filt[rise_10],color = 'black',label='rise')
        plt.scatter(time[rise_90],filt[rise_90],color = 'black')
        plt.scatter(time[peak_index],peak_av, color='red', label ='peak',zorder = 2)
        plt.scatter(time[presamples],filt[presamples], color='magenta', label ='risepoint',zorder = 2)
    
        plt.xlabel("time(s)")
        plt.ylabel("volt(V)")
        plt.title(os.path.basename(path).replace('.dat',''))
        plt.grid()
        plt.legend()
        plt.show()
        plt.cla()
        # show diff pulse
        #gp.graugh('diff pulse',dif,time[:samples-mv_av+1])
        #plt.show()

        print("end")
        print('---------------------------\n')

    else:
        print("Exit")
