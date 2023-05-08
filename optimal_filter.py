import numpy as np
import pandas as pd
import scipy.fftpack as fft
import libs.getpara as gp
import libs.fft_spectrum as sp
import matplotlib.pyplot as plt
import shutil
import os

# output.csvは必ず閉じておくこと（ファイルを保存できなくなる）
# ---Parameter--------------------------------------------------

eta = 100
cf = 10*1000
#------------------------------------------------



def lowpass(F,fq,cf):
    F[(fq>cf)] = 0
    return F

def main():
    #read setting
    set = gp.loadJson()
    path = set["Config"]["path"]
    print(path)
    os.chdir(path)
    ch,rate,samples,presamples,threshold = set["Config"]["channel"],set["Config"]["rate"],set["Config"]["samples"],set["Config"]["presamples"],set["Config"]["threshold"]
    time = gp.data_time(rate,samples)
    fq = np.arange(0,rate,rate/samples)
    path_output = (f'CH{ch}_pulse/output')
    
    pulse_av = np.loadtxt(f"CH{ch}_pulse/output/average_pulse.txt")
    noise_spe = np.loadtxt(f"CH{ch}_noize/output/modelnoise.txt")


    #outputファイルを読み込み
    df  = pd.read_csv(f"CH{ch}_pulse/output/output.csv",index_col=0)
    print(f'all data : {len(df)}')

    #Fourier transform
    F = fft.fft(pulse_av)
    amp = np.abs(F)
    amp_spe = np.sqrt(amp)*int(eta)*1e+6*np.sqrt(1/rate/samples)
    sp.graugh_spe(fq[:int(samples/2)+1],amp_spe[:int(samples/2)+1])
    plt.savefig(os.path.join(path_output,'averagepulse_spectrum.png'))
    plt.show()

    #LowPassFilter
    F2 = lowpass(F,fq,cf)
    amp_filt = np.abs(F2)
    amp_spe_filt = np.sqrt(amp_filt)*int(eta)*1e+6*np.sqrt(1/rate/samples)
    sp.graugh_spe(fq[:int(samples/2)+1],amp_spe_filt[:int(samples/2)+1])
    plt.show()

    filt = fft.ifft(F2/noise_spe)
    filt = filt.real
    np.savetxt(os.path.join(path_output,'opt_template.txt'),filt)
    plt.plot(time,filt)
    plt.savefig(os.path.join(path_output,'opt_template.png'))
    plt.show()

    pulsehight_array = []
    #最適フィルタ適用
    for i in df.index:
        print(i)
        data = gp.loadbi(i)
        base = df.at[i,"base"]
        data = data-base
        pulsehight = np.sum(data*filt)
        pulsehight_array.append(pulsehight)

    
    np.savetxt(os.path.join(path_output,'pulseheight_opt.txt'),pulsehight_array)
    df["height_opt"] = pulsehight_array
    df.to_csv(os.path.join(path_output,"output.csv"))




if __name__ == "__main__":
    main()

    