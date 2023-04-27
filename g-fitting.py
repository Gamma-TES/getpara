
import glob
import numpy as np
import scipy.optimize as opt
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
import pandas as pd
import libs.getpara as gp

# ---Parameter--------------------------------------------------
x_ax = "height"
energy = 1332
pulse_fmin = 0.95
pulse_fmax = 1.05
bins = 128
#------------------------------------------------

def histgrum(df,min,max):
    df = df[(df[x_ax]>min)&(df[x_ax]<max)]
    hist, bin_edges = np.histogram(df[x_ax], bins=bins)
    print((hist))
    
    

def gausse(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.0*sigma**2))

def FWHW(sigma):
    return 2*sigma*(2*np.log(2))**(1/2)
    

def main():
    set = gp.loadJson()
    ch,path = set["channel"],set["path"]
    os.chdir(path)
    df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
    data = gp.extruct(df,x_ax)
    histgrum(df,pulse_fmin,pulse_fmax)

    x = len(df)
    #popt,cov = curve_fit(gausse,df)
    
if __name__ == "__main__":
    main()
    

    
