# -*- coding: utf-8 -*-
#Calibration
# 11-6D1

import math
import glob
import numpy as np
import scipy.optimize as opt
import scipy.fftpack as fft
from array import array
import os
import os.path
import time
import matplotlib.pyplot as plt
import libs.getpara as gp
import pandas as pd
from scipy.optimize import curve_fit

#---Parameter----
# Co60: 1332keV,1172keV  Kx:-77
bins = 4098
hist_range = 1.2

#----------- パルスハイトの読み込み ----------------
set = gp.loadJson()
path = "/Volumes/Extreme 1TB/Matsumi/data/20230418/room2-2_140mK_870uA_gain10_trig0.1_10kHz"
os.chdir(path)
df = pd.read_csv((f'CH{set["channel"]}_pulse/output/output.csv'),index_col=0)
rate,samples,presamples,threshold,ch = set["rate"],set["samples"],set["presamples"],set["threshold"],set["channel"]



pulseheight = df["height_opt_temp"]

def gausse(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.0*sigma**2))

def FWHW(sigma):
    return 2*sigma*(2*np.log(2))**(1/2)

ap = len(pulseheight)


x = np.arange(0,bins,1)
hist = np.histogram(pulseheight,bins=bins,range=(0,hist_range))[0]

plt.bar(x,hist)
plt.show()


peaks = gp.search_peak(hist)

print(peaks)
print(f"{len(peaks)} peaks are detected!\n")

x_fit = np.arange(0,bins,0.1)
plt.bar(x,hist[0],width=1)
calibs = []
for i in range(len(peaks)):
	popt, cov = curve_fit(gausse,x,hist, p0=peaks[i])
	energy = int(input("input energy (keV): "))
	calibs.append(popt[1])
	fwhw = FWHW(np.abs(popt[2])) 
	dE = energy*fwhw/popt[1]
	print (f'半値幅: {fwhw}\nエネルギー分解能: {dE:.3f} keV\n')
	print(peaks[i])
	print("\n")
        
print(calibs)



        
	

