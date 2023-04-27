import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp

# main.py　の後　output.dat　を用いる
# 見たい関係性のパラメータをプロット（risetime vs pulseheight など）
# プロットから平均パルスを作成


# Parameter
path = 'H:/Matsumi/data/20230413/room2-2_140mK_1000uA_gain10_trig0.05_10kHz'
ch = 0 
x_ax = 'base'          
y_ax = 'pulseheight'



#平均パルスのグラフ化
def graugh_av(data,time):
    x = time
    y = data
    plt.plot(x,y,label="data")
    #plt.yscale('log')
    plt.xlabel("time")
    plt.ylabel("volt")
    plt.title('Average Pulse')
    plt.legend()
    plt.savefig("CH0_pulse/output/average_pulse.png")
    plt.show()
    plt.cla()




def main():
    os.chdir(path) 
    df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
    rate,samples,presamples = gp.setting("Setting.txt")
    time = gp.data_time(rate,samples)

    df = df[(df['samples']==samples)&(df['pulseheight']>0)&(df['rSquared']!=0)]
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

    av = gp.average_pulse(picked,presamples)
    graugh_av(av,time)
    np.savetxt('CH0_pulse/output/average_pulse_index.txt',picked,fmt="%s")
    np.savetxt('CH0_pulse/output/average_pulse.txt',av)


#実行
if __name__=='__main__':
    main()

    

    
