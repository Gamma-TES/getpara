
#2022/09/12
#それぞれのデータの周波数スペクトルの平均（モデルノイズ）を出力


import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import os
from natsort import natsorted
import glob
import shutil
import libs.getpara as gp
import pandas as pd

cf = 1e4

#実行
def main():
    set = gp.loadJson()
    os.chdir(set["Config"]["path"])
    ch = set['Config']['channel']
    rate,samples= set["Config"]["rate"],set["Config"]["samples"]
    time = gp.data_time(rate,samples)
    fq = np.arange(0,rate,rate/samples)
    eta = input("eta: ")
    noise = []
    model = np.array(0)*samples
    
    if os.path.exists(f"CH{ch}_noize/random_noise.txt"):
        mode = input('All -> [0], random -> [1]: ')
        if mode == "1":
            with open(f"CH{ch}_noize/random_noise.txt",'r',encoding='latin-1') as f:
                for row in f.read().splitlines():
                    noise.append(row)
    
    noise = natsorted(glob.glob(f"CH{ch}_noize/rawdata/CH{ch}_*.dat"))

    lpf = input('Low Pass Filter? [1]): ')
    for i in noise:
        try :
            data = gp.loadbi(i)
            if lpf == '1':
                data = gp.BesselFilter(data,rate,cf)
            base,data_ba = gp.baseline(data,set['Config']['presamples'],1000,500)
            peak = np.max(data_ba)
            if (base <= -3 and base >= 3) or peak >= float(set['Config']['threshold']):
                print("Not noise")
                continue
            amp = np.abs(fft.fft(data_ba))**2
            model = model + amp

            print(i)
        except FileNotFoundError:
            continue
    model = model/len(noise)

    amp_spe = np.sqrt(model[:int(samples/2)+1])*int(eta)*1e+6*np.sqrt(1/rate/samples)

    os.mkdir(f'CH{ch}_noize/output')
    np.savetxt(f'CH{ch}_noize/output/modelnoise.txt',model)
    
    

    #スペクトルをグラフ化
    plt.plot(fq[:int(samples/2)+1],amp_spe,linestyle = '-',linewidth = 0.7)
    plt.loglog()
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Intensity[pA/kHz$^{1/2}$]')
    plt.grid()
    plt.savefig(f'CH{ch}_noize/output/modelnoise.png')
    plt.show()

if __name__ == "__main__":
    main()

