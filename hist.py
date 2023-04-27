import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import libs.getpara as gp


# ---Parameter--------------------------------------------------
path = 'H:/Matsumi/data/20230413/room2-2_140mK_1000uA_gain10_trig0.05_10kHz'
ch = 0
eta = 100
cf = 10*1000
#------------------------------------------------

os.chdir(path)
setting = np.loadtxt("Setting.txt",skiprows = 10)
rate = int(setting[2])
samples = int(setting[4])
time = np.arange(0,1/rate*samples,1/rate)
fq = np.arange(0,rate,rate/samples)


data = np.loadtxt("CH0_pulse/output/pulseheight_opt_tem.txt")

x = np.arange(0,len(data),1)
plt.hist(data,bins = 4096,range = (0,2.0),color = 'blue')
plt.xlabel('pulseheight[V]',fontsize = 16)
plt.ylabel('count[-]',fontsize = 16)
#plt.yscale(('log'))
plt.savefig('spectral_hist_opt.png',format = 'png')
plt.show()