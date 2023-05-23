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

path = 'E:/matsumi/data/20230512/room2-2_140mK_870uA_gain10_trig0.4_500kHz/CH0_pulse/output/select'
rawdata = 'average_pulse.txt'
rate = int(1e6)
samples = int(1e5)
presamples = int(10000)
time = np.arange(0,1/rate*samples,1/rate)


#-----②解析パラメータ------------------------------------------------
cf = 4e4        # Low Pass Filter cut off
x_ba = 1000     # baseを取るときのスタート点: presamples - x_ba
w_ba = 500      # baseを取る幅: presamples - x_ba + w_ba
w_max = 300     # peakを探す幅: presamples + w_max
x_av = 5        # peak_avを探す時のスタート点: peak_index - x_av
w_av = 20       # peak_avを探す幅: peak_index - x_av + w_av


# ----Fitting parameter----------------------------

# rise
start_rise = 0 # presamples + start_rise
width_rise = 60 # start_rise + width_rise

# decay
start_decay = 2000 # peak_index + start_decay
width_decay = 10000 # start_decay + width_decay

# double
start = -1      # presamples + start
width = 1000  #  start + width
p0 = [-1.6,12,presamples,5570,presamples]
#-----------------------------------------------------------------

# Fitting only rise or decay
def monoExp(x,m,t,b):
    return m * np.exp(-(x-b)/t)

def monoExp_rise(x,m,t,b,a):
    return m * np.exp(-(x-b)/t) + a

# Fitting entire pulse
def doubleExp(x,m,t1,b1,t2,b2):
    return m * (np.exp(-(x-b1)/t1) - np.exp(-(x-b2)/t2))



def main():
    os.chdir(path) 
    #data = gp.loadbi(rawdata)
    data = np.loadtxt(rawdata)
        
    base,data = gp.baseline(data,presamples,x_ba,w_ba)
    #data_txt = np.savetxt('CH0_242.txt',data)
    filt = gp.BesselFilter(data,rate,cf)
    peak,peak_av,peak_index = gp.peak(data,presamples,w_max,x_av,w_av)

    x_fit = np.arange(presamples-5,samples,1)
    x_fit_time = time[presamples-5:samples]
    

    # rise
    x_rise = np.arange(presamples+start_rise,presamples+start_rise+width_rise)
    data_rise = data[presamples+start_rise:presamples+start_rise+width_rise]
    params,cov = curve_fit(monoExp_rise,x_rise,data_rise,p0=[-2,12,presamples,peak_av],maxfev=500000)
    fit_rise = monoExp_rise(x_fit,*params)
    print(f"rise:{params}")
    
    
    # decay
    x_decay = np.arange(peak_index+start_rise,peak_index+start_rise+width_decay)
    data_decay = data[peak_index+start_rise:peak_index+start_rise+width_decay]
    params_decay,cov = curve_fit(monoExp,x_decay,data_decay,p0=[4,5000,presamples],maxfev=500000)
    fit_decay = monoExp(x_fit,*params_decay)
    print(f"decay:{params_decay}")

    # sum      y = -((- rise + peak ) + (-decay)) = rise decay -peak
    fit_sum = fit_rise + fit_decay - peak_av


    x = np.arange(0,samples)
    params_double,cov = curve_fit(doubleExp,\
                                  x[presamples+start: presamples+start+width],data[presamples+start: presamples+start+width],p0=p0,maxfev=1000000)
    print(f"double:{params_double}")
    fit_double = doubleExp(x_fit,*params_double)

    # rise time
    rise,rise_10,rise_90 = gp.risetime(data,peak_av,peak_index,rate)
    rise_fit = gp.risetime(fit_double,np.max(fit_double),np.argmax(fit_double),rate)
    print(f'rise time (rawdata): {rise}')
    print(f'rise time (fitting): {rise_fit[0]}')

    


    plt.plot(time,data,'o',markersize=1,label = rawdata)
    #plt.plot(x_fit_time,fit_rise,'-.',label = "rise fit")
    #plt.plot(x_fit_time,fit_decay,'-.',label = "decay fit")
    #plt.plot(x_fit_time,fit_sum,'--',label='sum fit ',c = 'purple')
    plt.plot(x_fit_time,fit_double,label = "double fit")
    plt.title('{0:.2f}($e^{{(x-{2:.2f})/{1:.2f}}}$ - $e^{{(x-{4:.2f})/{3:.2f}}})$'.format(*params_double))
    plt.xlabel("number")
    plt.ylabel('Volt [V]')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()
