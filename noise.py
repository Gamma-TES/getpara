
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
ch = 0
eta = 98
output = 'output1'
#------------------------------------------------

def output(path_output):
    if not os.path.exists(path_output):
        os.mkdir(path_output)
        np.savetxt(f'CH{ch}_noize/output/modelnoise.txt',model)
    else:
        shutil.rmtree(path_output)
        os.mkdir(path_output)
        np.savetxt(f'CH{ch}_noize/output/modelnoise.txt',model)


#実行
if __name__ == "__main__":
    set = gp.loadJson()
    os.chdir(set["path"])
    df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
    rate,samples,presamples,threshold = set["rate"],set["samples"],set["presamples"],set["threshold"]
    time = gp.data_time(rate,samples)
    fq = np.arange(0,rate,rate/samples)


    model = np.array(0)*samples
    noise = natsorted(glob.glob(f"CH{ch}_noize/rawdata/CH{ch}_*.dat"))
    for i in noise:
        print(i) 
        data = gp.loadbi(i)
        amp = np.abs(fft.fft(data))**2
        model = model + amp
    model = model/len(noise)
    amp_spe = np.sqrt(model[:int(samples/2)+1])*int(eta)*1e+6*np.sqrt(1/rate/samples)

    #outputを出力
    output(f'CH{ch}_noize/output')
    

    #スペクトルをグラフ化
    plt.plot(fq[:int(samples/2)+1],amp_spe,linestyle = '-',linewidth = 0.7)
    plt.loglog()
    plt.xlabel('Frequency[kHz]')
    plt.ylabel('Intensity[pA/kHz$^{1/2}$]')
    plt.grid()
    plt.savefig(f'CH{ch}_noize/output/modelnoise.png')
    plt.show()

