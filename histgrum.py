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

hist_set = {
    "bins" :8192,
}
#------------------------------------------------

         
y_ax = 'height_opt_temp'
bins = hist_set['bins']


cmap = cm.get_cmap("Set2")

def search_peak(hist):
    diff = gp.diff(hist)
    diff2 = gp.diff(diff)

    plt.plot(diff)
    plt.show()
    min= int(input("range min: "))
    max= int(input("range max: "))
    threshold = float(input("Threshold: "))
    print("\n")

    trigger = False
    trigger2 = False 
    peak_list = []

    for i in reversed(np.arange(min,max,1)):
        if trigger == False and diff[i] < threshold * -1:
            trigger = True
        if trigger == True and diff[i] > 0:
            if trigger2 == False:
                peak = []
                peak.append(np.max(hist[i-1:i+1]))
                peak.append(np.argmax(hist[i-1:i+1]))
                trigger2 = True
            if trigger2 == True and diff2[i] > 0:
                peak.append(2 * (peak[1]-(i-1)))
                peak_list.append(peak)
                trigger = False
                trigger2 = False

    return peak_list


def gausse(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.0*sigma**2))

def FWHW(sigma):
    return 2*sigma*(2*np.log(2))**(1/2)


def main():

    # Load Setting 
    set = gp.loadJson()
    path,ch = set["Config"]["path"],set["Config"]["channel"]
    os.chdir(path)

    # Load data and transform histgrum
    df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
    data = gp.extruct(df,y_ax)
    hist = np.histogram(data,bins=bins,range={0,50})[0]

    output = f"CH{ch}_pulse/output"
    set = json.dumps(hist_set,indent=4)
    with open(os.path.join(output,"histgrum_set.json"), 'w') as file:
        file.write(set)
    np.savetxt(os.path.join(output,"histgrum.txt"),hist)

    # Show histgrum
    x = np.arange(0,hist_set['bins'],1)
    plt.bar(x,hist,width=1)
    plt.xlabel('Pulse Height [ch]')
    plt.ylabel('Counts')
    plt.legend(edgecolor='white', framealpha=1, fontsize=6)
    plt.tick_params(axis='both', which='both', direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.2)
    plt.grid(True, which='minor', color='black', linestyle=':', linewidth=0.1)
    plt.show()
    

    # Serch peaks
    peaks = gp.search_peak(hist)
    print(f"{len(peaks)} peaks are detected!\n")
    for i in peaks:
        print(i)
    



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