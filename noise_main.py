
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
import json


#実行
def main():
    set = gp.loadJson()
    if not 'eta' in set['Config']:
        eta = input("eta: ")
        set['Config']['eta'] = float(eta)
        jsn = json.dumps(set,indent=4)
        with open("setting.json", 'w') as file:
            file.write(jsn) 
    os.chdir(set["Config"]["path"])
    ch = set['Config']['channel']
    rate,samples= set["Config"]["rate"],set["Config"]["samples"]
    eta = set["Config"]["eta"]*1e-6
    time = gp.data_time(rate,samples)
    fq = np.arange(0,rate,rate/samples)
    output = f'CH{set["Config"]["channel"]}_noise/output/{set["Config"]["output"]}'

    

    model = np.array(0)*samples   
    noise = natsorted(glob.glob(f"CH{ch}_noise/rawdata/CH{ch}_*.dat"))


    for i in noise:
        try :
            data = gp.loadbi(i)      
            base,data_ba = gp.baseline(data,set['Config']['presamples'],1000,500)
            data = gp.BesselFilter(data_ba,rate,set['main']['cutoff'])
            peak = np.max(data_ba)
            if (base <= -3 and base >= 3) or peak >= float(set['Config']['threshold']):
                print("Not noise")
                continue
            amp = np.abs(data/(samples/2))
            df = rate/samples
            power = amp**2/df
            amp = np.sqrt(power)*eta
            model = model + amp

            print(i)
        except FileNotFoundError:
            continue
    model = model/len(noise)

    amp_spe = model[:int(samples/2)+1]*eta

    np.savetxt(f'{output}/modelnoise.txt',model) 
    
    

    #スペクトルをグラフ化
    plt.plot(fq[:int(samples/2)+1],amp_spe,linestyle = '-',linewidth = 0.7)
    plt.loglog()
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Intensity[pA/Hz$^{1/2}$]')
    plt.grid()
    plt.savefig(f'{output}/modelnoise.png')
    plt.show()

if __name__ == "__main__":
    main()

