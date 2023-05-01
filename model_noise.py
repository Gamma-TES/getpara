
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

# ---Parameter--------------------------------------------------



#------------------------------------------------

    


#実行
if __name__ == "__main__":
    set = gp.loadJson()
    os.chdir(set["Config"]["path"])
    ch = set['Config']['channel']
    rate,samples= set["Config"]["rate"],set["Config"]["samples"],set["Config"]
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

    for i in noise:
        print(i)
        try :
            data = gp.loadbi(i)
            amp = np.abs(fft.fft(data))**2
            model = model + amp
        except FileNotFoundError:
            continue
    model = model/len(noise)
    amp_spe = np.sqrt(model[:int(samples/2)+1])*int(eta)*1e+6*np.sqrt(1/rate/samples)

    os.mkdir(f'CH{ch}_noize/output')
    np.savetxt(f'CH{ch}_noize/output/modelnoise.txt',model)
    
    

    #スペクトルをグラフ化
    plt.plot(fq[:int(samples/2)+1],amp_spe,linestyle = '-',linewidth = 0.7)
    plt.loglog()
    plt.xlabel('Frequency[kHz]')
    plt.ylabel('Intensity[pA/kHz$^{1/2}$]')
    plt.grid()
    plt.savefig(f'CH{ch}_noize/{output}/modelnoise.png')
    plt.show()

