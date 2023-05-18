import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
from natsort import natsorted
from scipy.optimize import curve_fit
import libs.getpara as gp
import libs.fft_spectrum as sp
import json

#-----②解析パラメータ------------------------------------------------
cf = 1e5        # Low Pass Filter cut off
x_ba = 1000     # baseを取るときのスタート点: presamples - x_ba
w_ba = 500      # baseを取る幅: presamples - x_ba + w_ba
w_max = 300     # peakを探す幅: presamples + w_max
x_av = 5       # peak_avを探す時のスタート点: peak_index - x_av
w_av = 20       # peak_avを探す幅: peak_index - x_av + w_av
x_fit = 5       # fittingのスタート点: peak_index(average) + x_fit
w_fit = 10     # fittingの幅: peak_index(average) + x_fit + w_fit
mv_av = 100     # 移動平均の幅
decay_sillicon = 0.001 # コンイベント判別の為のディケイ
#-----------------------------------------------------------------

def monoExp(x,m,t,a,b):
    return m * np.exp(-(x-b)/t) + a

def doubleExp(x,m1,t1,b1,m2,t2,b2,a):
    return m1 * np.exp(-(x-b1)/t1) + m2 * np.exp(-(x-b2)/t2) + a

def doubleExp2(x,m1,t1,b1,m2,t2,b2,a):
    return (1-t1/t2)*(1-t1/t2)   * np.exp(-(x-b1)/t1) + np.exp(-(x-b2)/t2) + a

def main():
    # Settingを取得
    set = gp.loadJson()
    path = 'H:/Matsumi/data/20230510/room2-2_140mK_870uA_gain10_trig0.1_10kHz/compare'
    ch,rate,samples,presamples,threshold = \
        set["Config"]["channel"],set["Config"]['rate'],set["Config"]['samples'],set["Config"]["presamples"],set["Config"]["threshold"]
    time = np.arange(0,1/rate*samples,1/rate)
    os.chdir(path) 

    path = f'average_pulse_10kHz.txt'
    data = np.loadtxt(path)
        
    base,data = gp.baseline(data,presamples,x_ba,w_ba)
    mv = gp.moving_average(data,mv_av)
    dif  = gp.diff(mv)
    filt = gp.BesselFilter(data,rate,cf)

    base,data = gp.baseline(data,presamples,x_ba,w_ba)

    peak,peak_av,peak_index = gp.peak(filt,presamples,w_max,x_av,w_av)
    rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)

    x = np.arange(0,samples,1)
    x_fit = np.arange(presamples-5,samples,0.1)
    
    
    plt.plot(x,data)

    # rise
    x_rise = np.arange(presamples,presamples+50,1)

    data_rise = data[presamples:presamples+50]
    params,cov = curve_fit(monoExp,x_rise,data_rise,p0=[-2,12,2.5,presamples],maxfev=50000)
    print(f"rise:{params}")

    
    fit_rise = monoExp(x_fit,*params)
    plt.plot(x_fit,fit_rise,'-.',label = "rise fit")

    # decay

    x_decay = np.arange(peak_index+100,peak_index + 20000,1)
    data_decay = data[peak_index+100:peak_index + 20000]
    params_decay,cov = curve_fit(monoExp,x_decay,data_decay,p0=[4,5000,0,presamples],maxfev=500000)
    print(f"decay:{params_decay}")

    fit_decay = monoExp(x_fit,*params_decay)
    plt.plot(x_fit,fit_decay,'-.',label = "decay fit")

    fit_sum = fit_rise + fit_decay -peak_av
    plt.plot(x_fit,fit_sum,'--',label='fit sum')

    # y = -((- rise + peak ) + (-decay)) = rise decay -peak

    params_double,cov = curve_fit(doubleExp,x[presamples-5:peak_index + 20000],data[presamples-5:peak_index + 20000],p0=[-2,12,2.5,4,5000,0,presamples],maxfev=500000)
    print(f"double:{params_double}")
    fit_double = doubleExp(x_fit,*params_double)
    plt.plot(x_fit,fit_double,'-.',label = "fit double",c='yellow')


    plt.xlabel("number")
    plt.ylabel('Volt [V]')
    plt.legend()
    plt.grid()
    #plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()
