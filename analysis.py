import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp

# main.py　の後　output.dat　を用いる
# 見たい関係性のパラメータをプロット（risetime vs pulseheight など）
# プロットから平均パルスを作成


# Parameter

x_ax = 'base'          
y_ax = 'height_opt_temp'
output = 'output'

def main():
    set = gp.loadJson()

    os.chdir(set["path"])
    df = pd.read_csv((f'CH{set["channel"]}_pulse/output/output.csv'),index_col=0)
    rate,samples,presamples,threshold,ch = set["rate"],set["samples"],set["presamples"],set["threshold"],set["channel"]
    time = gp.data_time(rate,samples)
    print(len((df)))
    df = df[(df['samples']==samples)&(df['height']>threshold)&(df['max_div']<0.1)&(df['rise']>4e-05)]
    #&(df['decay']>0.001)&(df['rise']<0.0001)&(df['max_div']<0.01)&(df['decay']>0.01)
    print(f'pulse : {len(df)}')
     
    x,y = gp.extruct(df,x_ax,y_ax)
    gp.graugh_para(x,y,x_ax,y_ax,color = "tab:blue")
    plt.show()
    plt.cla()

    timeline = np.arange(0,len(x),1)
    plt.scatter(timeline,x,s=0.5)
    plt.show()

    #平均パルスを取得
    picked = gp.pickSamples(x_ax,y_ax,df) # pick samples from graugh
    print(f"Selected {len((picked))} Samples.")

    if len(picked) == 1:
        path_name = picked[0]
        picked_data = gp.loadbi(path_name)
        gp.graugh(path_name,picked_data,time)
        plt.show()
    else:    
        av = gp.average_pulse(picked,presamples)
        plt.savefig(f'CH0_pulse/output/average_pulse.png')
        plt.show()
        np.savetxt(f'CH0_pulse/output/average_pulse_index.txt',picked,fmt="%s")
        np.savetxt(f'CH0_pulse/output/average_pulse.txt',av)


#実行
if __name__=='__main__':
    main()

    

    
