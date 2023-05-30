import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
import pandas as pd
import libs.getpara as gp
import json


# ---Parameter--------------------------------------------------
y_ax = 'height_opt_temp'
bin = 8192
#------------------------------------------------

cmap = cm.get_cmap("Set2")



def gausse(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.0*sigma**2))

def FWHW(sigma):
    return 2*sigma*(2*np.log(2))**(1/2)


def main():

    # Load Setting 
    set = gp.loadJson()
    path,ch = set["Config"]["path"],set["Config"]["channel"]
    samples,threshold = set['Config']["samples"],set['Config']['threshold']
    os.chdir(path)

    # Load data and transform histgrum
    df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
    #df = df[(df['samples']==samples)&(df['height']>threshold)&(df['decay']>0.01)&(df['rise_fit']!=0)&(df['rise_fit'] < 100)&(df['base']>0.0)\
    #        &(df['rise_fit']<0.001)&(df['tau_decay']<10000)]
    #data_index = gp.loadIndex(f'CH{ch}_pulse/output/select/selected_base_index.txt')
    data = df["height_opt_temp"]
    hist,bins = np.histogram(data,bins=bin,range=[0,np.max(data)*1.05])

    
    #plt.hist(data,bins=bins, range=[0,np.max(data)*1.05])
    plt.bar(bins[:-1],hist,width = bins[1]-bins[0])
    plt.xlabel('Pulse Height [ch]')
    plt.ylabel('Counts')
    plt.tick_params(axis='both', which='both', direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.2)
    plt.grid(True, which='minor', color='black', linestyle=':', linewidth=0.1)
    plt.savefig(f'CH{ch}_pulse/output/select/histgrum.png')
    plt.show()
    

    # Serch peaks
    """
    if input("Serch peaks[0]: ") == "0":
        peaks = gp.search_peak(hist)
        print(f"{len(peaks)} peaks are detected!\n")
        for i in peaks:
            print(i)
    """

    """
    # Fiiting every peaks and plot
    x_fit = np.arange(0,bins,0.1)
    plt.bar(x,hist,width=1)
    for i in range(len(peaks)):
        popt, cov = curve_fit(gausse,x,hist, p0=peaks[i])
        energy = int(input("input energy (keV): "))
        fwhw = FWHW(popt[2]) 
        dE = energy*fwhw/popt[1]
        print (f'半値幅: {fwhw}\nエネルギー分解能: {dE:.3f} keV\n')
        print(peaks[i])
        print("\n")
        

        gfit = gausse(x_fit,popt[0],popt[1],popt[2])
        plt.plot(x_fit,gfit,color = cmap((float(i))/float(len(peaks))),\
                 label = f"{energy} keV",linestyle='-')

    plt.xlabel('Pulse Height [ch]')
    plt.ylabel('Counts')
    plt.legend(edgecolor='white', framealpha=1, fontsize=6)
    plt.tick_params(axis='both', which='both', direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.2)
    plt.grid(True, which='minor', color='black', linestyle=':', linewidth=0.1)
   
    plt.show()
    plt.cla()
    """


if __name__ == "__main__":
    main()
    print('end')