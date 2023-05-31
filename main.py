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
import pprint


#-----Analtsis Parameter------------------------------------------------

para = {"main":{
    'cutoff':1e4,
    'base_x':1000,
    'base_w':500,
    'peak_max':300,
    'peak_x':5,
    'peak_w':20,
    'fit_x':-1,
    'fit_w':1000,
    'fit_p0': [-1.6,12,10000,5570,10000],
    'mv_w':100,
    }
}


# run
if __name__ == '__main__':

    # Get Setting
    set = gp.loadJson()
    if not 'main' in set:
        set.update(para)
        jsn = json.dumps(set,indent=4)
        with open("setting.json", 'w') as file:
            file.write(jsn)
    pprint.pprint(set)
    path = set["Config"]["path"]
    ch,rate,samples,presamples,threshold = \
        set["Config"]["channel"],set["Config"]['rate'],set["Config"]['samples'],set["Config"]["presamples"],set["Config"]["threshold"]
    time = np.arange(0,1/rate*samples,1/rate)
    os.chdir(path) 

    path_data = natsorted(glob.glob(f'CH{ch}_pulse/rawdata/CH{ch}_*.dat'))
    #path = natsorted(glob.glob(f'CH{ch}_pulse/test/CH{ch}_*.dat'))

    mode = input('Analysis Mode (all -> [0], one -> [1]): ')

    # All Samples
    if mode == '0':
        data_array = []
        for num in path_data:
            print(os.path.basename(num))
            data = gp.loadbi(num)
            base,data = gp.baseline(data,presamples,para['main']['base_x'],para['main']['base_w'])

            
            p0 = [-1.6,12,presamples,5570,presamples]
            x_fit = np.arange(presamples-5,samples,0.1)
            popt,rSquared = gp.fitExp(gp.doubleExp,data,presamples+para['main']['fit_x'],para['main']['fit_w'],p0 = para['main']['fit_p0'])
            tau_rise = popt[1]/rate
            tau_decay = popt[3]/rate
            fitting = gp.doubleExp(x_fit,*popt)
            rise_fit = gp.risetime(fitting,np.max(fitting),np.argmax(fitting),rate)


            # Low pass filter
            data = gp.BesselFilter(data,rate,para['main']['cutoff'])
            
            # analysis
            peak,peak_av,peak_index = gp.peak(data,presamples,para['main']['peak_max'],para['main']['peak_x'],para['main']['peak_w'])
            rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
            decay,decay_10,decay_90 = gp.decaytime(data,peak_av,peak_index,rate)
            
            data_column = [samples,base,peak_av,rise,decay,rise_fit[0],tau_rise,tau_decay,rSquared]
            data_array.append(data_column)
        
        # create pandas DataFrame
        df = pd.DataFrame(data_array,\
        columns=["samples","base","height","rise","decay","rise_fit",'tau_rise','tau_decay',"rSquared"],\
        index=path_data)


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
        base,data = gp.baseline(data,presamples,para['main']['base_x'],para['main']['base_w'])
        mv = gp.moving_average(data,para['main']['mv_w'])
        dif  = gp.diff(mv)
        filt = gp.BesselFilter(data,rate,para['main']['cutoff'])

        # Analysis
        peak,peak_av,peak_index = gp.peak(data,presamples,para['main']['peak_max'],para['main']['peak_x'],para['main']['peak_w'])
        rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
        decay,decay_10,decay_90 = gp.decaytime(data,peak_av,peak_index,rate)

        # Fitting
        p0 = [-1.6,12,presamples,5570,presamples]
        x_fit = np.arange(presamples-5,samples,0.1)
        popt,rSquared = gp.fitExp(gp.doubleExp,data,presamples+para['main']['fit_x'],para['main']['fit_w'],p0 = para['main']['fit_p0'])
        tau_rise = popt[1]/rate
        tau_decay = popt[3]/rate
        fitting = gp.doubleExp(x_fit,*popt)
        rise_fit = gp.risetime(fitting,np.max(fitting),np.argmax(fitting),rate)


        # Console output
        print('\n---------------------------')
        print(path)
        print(f'samples : {len(data)}')
        print(f'base : {base:.5f}')
        print(f'hight : {peak_av:.5f}')
        print(f'peak index : {peak_index:.5f}')
        print(f'rise : {rise:.5f}')
        print(f'rise_fit : {rise_fit[0]:.5f}')
        print(f'tau_rise : {tau_rise:.5f}' )
        print(f'decay : {decay:.5f}')
        print(f'tau_decay : {tau_decay:.5f}' )
        print(f'rSqared: {rSquared:.5f}')


        # Graugh
        plt.plot(time,data,'o',markersize=1,label="rawdata")
        plt.plot(time,filt,'o',markersize=1,label = "filt")
        plt.plot(time[presamples-para['main']['base_x']:presamples-para['main']['base_x']+para['main']['base_w']],filt[presamples-para['main']['base_x']:presamples-para['main']['base_x']+para['main']['base_w']],'--',label="base")
        plt.plot(x_fit/rate,fitting,'-.',label = 'fitting')

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
        gp.graugh('diff pulse',dif,time[:samples-para['main']['mv_w']+1])
        plt.show()

        print("end")
        print('---------------------------\n')

    else:
        print("Exit")
