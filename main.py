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


#-----②解析パラメータ------------------------------------------------
x_ba = 1000     #baseを取るときのスタート点: presamples - x_ba
w_ba = 500      #baseを取る幅: presamples - x_ba + w_ba
w_max = 500     #peakを探す幅: presamples + w_max
x_av = 10       #peak_avを探す時のスタート点: peak_index - x_av
w_av = 30       #peak_avを探す幅: peak_index - x_av + w_av
x_fit = 5       #fittingのスタート点: peak_index(average) + x_fit
w_fit = 800     #fittingの幅: peak_index(average) + x_fit + w_fit
mv_av = 100     #移動平均の幅
decay_sillicon = 0.001 #シリコンイベント判別の為のディケイ
#-----------------------------------------------------------------



#実行
if __name__ == '__main__':

    # Settingを取得
    set = gp.loadJson()
    path = set["Config"]["path"]
    ch,rate,samples,presamples,threshold = \
        set["Config"]["channel"],set["Config"]['rate'],set["Config"]['samples'],set["Config"]["presamples"],set["Config"]["threshold"]
    time = np.arange(0,1/rate*samples,1/rate)
    os.chdir(path) 


    #パルス解析モード
    path = natsorted(glob.glob(f'CH{ch}_pulse/rawdata/CH{ch}_*.dat'))
    #path = natsorted(glob.glob(f'test/rawdata/CH{ch}_*.dat'))
    mode = input('Analysis Mode (all -> [0], one -> [1]): ')
    #全てのパルスを解析
    if mode == '0':
        fit = int(input('fitting? (no -> [0], yes -> [1]): '))
        data_array = []
        for num in path:
            print(os.path.basename(num))
            data = gp.loadbi(num)
            base,data = gp.baseline(data,presamples,x_ba,w_ba)
            mv = gp.moving_average(data,mv_av)
            dif  = gp.diff(mv)
            peak,peak_av,peak_index = gp.peak(data,presamples,w_max,x_av,w_av)
            peak_mv,peak_mv_av,peak_mv_index = gp.peak(mv,presamples,w_max,x_av,w_av)
            rise,rise_10,rise_90 = gp.risetime(data,peak_mv,peak_mv_index,rate)
            decay = gp.decaytime(data,peak_mv,peak_mv_index,rate)

            if fit:
                m,t,tauSec,rSquared,max_div,max_index = gp.fitting(data,peak_index,time,x_fit,w_fit,mode)
                data_column = [samples,base,peak_av,peak_mv_av,rise,decay,tauSec,max_div,rSquared]
            else:
                data_column = [samples,base,peak_av,peak_mv_av,rise,decay]
            # シリコンイベントはピークの平均値をとらない
            if decay < decay_sillicon:
                peak_av = peak
                peak_mv_av = peak
            data_array.append(data_column)
            #DataFrameの作成

        if fit:
            df = pd.DataFrame(data_array,\
            columns=["samples","base","height","height_mv","rise","decay","tauSec","max_div","rSquared"],\
            index=path)
            print('fitting')
        else:
            df = pd.DataFrame(data_array,\
            columns=["samples","base","height","height_mv","rise","decay"],\
            index=path)

        output_path = (f'CH{ch}_pulse/output')
        gp.output(output_path,df)
        print('end')    
        
        #1つのパルスを解析
    elif mode == '1':
        num = input('Enter pulse number: ')
        path = os.path.join('CH'+ch+'_pulse'+'/rawdata/CH'+ch+'_'+str(num)+'.dat')
        data = gp.loadbi(path)
        base,data = gp.baseline(data,presamples,x_ba,w_ba)
        mv = gp.moving_average(data,mv_av)
        dif  = gp.diff(mv)
        peak,peak_av,peak_index = gp.peak(data,presamples,w_max,x_av,w_av)
        peak_mv,peak_mv_av,peak_mv_index = gp.peak(mv,presamples,w_max,x_av,w_av)
        rise,rise_10,rise_90 = gp.risetime(data,peak_mv,peak_mv_index,rate)
        decay = gp.decaytime(data,peak_mv,peak_mv_index,rate)
        
        
        # シリコンイベントはピークの平均値をとらない
        if decay < decay_sillicon:
            peak_av = peak
            peak_mv_av = peak
            print("Seilicon Event")
        else:
            print('Absorver Event')
        
        
        gp.graugh(path,data,time)
        plt.plot(time[presamples-x_ba:presamples-x_ba+w_ba],data[presamples-x_ba:presamples-x_ba+w_ba],color = "green",label="base")
        plt.plot(time[rise_10:rise_90],data[rise_10:rise_90],color = "black",label="rise")
        plt.scatter(time[peak_index],peak_av, color='red', label ='peak',zorder = 2)
        plt.scatter(time[presamples],data[presamples], color='magenta', label ='risepoint',zorder = 2)
        plt.plot(time[peak_index + x_fit:peak_index + x_fit+ w_fit],data[peak_index + x_fit:peak_index + x_fit+ w_fit], label ='fit_range',zorder = 2)
        m,t,tauSec,rSquared,max_div,max_index = gp.fitting(data,peak_index,time,x_fit,w_fit,mode)
        #m1,t1,m2,t2,tauSec,rSquared,max_div,max_index = gp.fitting_double(data,peak_index,time,x_fit,w_fit,mode)
        
        
        
        
        #gp.graugh_save(path,data,time)
        print('\n---------------------------')
        print(path)
        print(f'samples : {len(data)}')
        print(f'hight : {peak_av:.5f}')
        print(f'peak index : {peak_index:.5f}')
        print(f'hight_mv : {peak_mv_av:.5f}')
        print(f'peak_mv index : {peak_mv_index:.5f}')
        print(f'rise : {rise}')
        print(f'decay : {decay}')
        print(f"Y = {m:.3f} * e^(-{t:.3f} * x)")
        #print(f"Y = {m1:.3f} * e^(-{t1:.3f} * x) + {m2:.3f} * e^(-{t2:.3f} * x)")
        print(f"tauSec = {tauSec:.5f}")
        print(f"rSquared = {rSquared:.5f}")
        print(f"max_div = {max_div:.5f}")
        print(f"max_index = {max_index}")

        

        #plt.yscale("log")
        plt.legend()
        
        
        gp.graugh(path,mv,time[:samples-mv_av+1])
        plt.scatter(time[peak_mv_index],mv[peak_mv_index], label ='mv_peak',color = 'yellow',zorder = 2)
        plt.show()
        plt.cla()

        gp.graugh('diff pulse',dif,time[:samples-mv_av+1])
        plt.show()
        print("end")
        print('---------------------------\n')

    else:
        print("Exit")