import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp
import sys

# main.py　の後　output.dat　を用いる
# 見たい関係性のパラメータをプロット（risetime vs pulseheight など）
# プロットから平均パルスを作成


# Parameter

x_ax = 'rise'          
y_ax = 'height_opt_temp'
output = 'output'

def main():
    ax = sys.argv
    ax.pop(0)
    set = gp.loadJson()

    os.chdir(set["Config"]["path"])
    df = pd.read_csv((f'CH{set["Config"]["channel"]}_pulse/output/output.csv'),index_col=0)
    rate,samples,presamples,threshold,ch = set["Config"]["rate"],set["Config"]["samples"],set["Config"]["presamples"],set["Config"]["threshold"],set["Config"]["channel"]
    time = gp.data_time(rate,samples)
    df = df[(df['samples']==samples)&(df['height']>threshold)]
    #&(df['decay']>0.001)&(df['rise']<0.0001)&(df['max_div']<0.01)&(df['decay']>0.01)
    print(f'Pulse : {len(df)} samples')

    #平均パルスを取得
    if len(ax) == 1:
        y = gp.extruct(df,ax)
        x = np.arange(len(y[0]))
        plt.scatter(x,y[0],s=0.4)
        plt.xlabel("number")
        plt.ylabel(ax[0])
        plt.savefig(f'CH{ch}_pulse/output/{ax[0]}.png')
        plt.show()


    elif len(ax) == 2:
        picked = gp.pickSamples(df,*ax) # pick samples from graugh
        print(f"Selected {len((picked))} samples.")

        if len(picked) == 1:
            path_name = picked[0]
            picked_data = gp.loadbi(path_name)
            gp.graugh(path_name,picked_data,time)
            plt.show()
        else:
            print('Creating Average Pulse...')
            av = gp.average_pulse(picked,presamples)
            plt.plot(time,av)
            plt.xlabel("time(s)")
            plt.ylabel("volt(V)")
            plt.title("average pulse")
            plt.savefig(f'CH{ch}_pulse/output/average_pulse.png')
            plt.show()
            np.savetxt(f'CH{ch}_pulse/output/selected_index.txt',picked,fmt="%s")
            np.savetxt(f'CH{ch}_pulse/output/average_pulse.txt',av)
    
    print('end')


#実行
if __name__=='__main__':
    main()

    

    
