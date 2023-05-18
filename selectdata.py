import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp
import sys
import shutil
import pprint

# main.py　の後　output.csv　を用いる
# 見たい関係性のパラメータをプロット（risetime vs pulseheight など）
# プロットから平均パルスを作成


#平均パルスを作成
def average_pulse(index,presamples):
    array = []
    for i in index:
        data = gp.loadbi(i)
        base,data = gp.baseline(data,presamples,1000,500)
        array.append(data)
    av = np.mean(array,axis=0)
    return av

ax_unit = {
    "base":'base[V]',
    "height":'pulse height[V]',
    "height_opt":'pulse height opt',
    'rise':'time[s]',
    'decay':'time[s]',

}

def main():
    ax = sys.argv
    ax.pop(0)
    set = gp.loadJson()
    pprint.pprint(set)
    os.chdir(set["Config"]["path"])

    output = f'CH{set["Config"]["channel"]}_pulse/output/select/img'
    if not os.path.exists(output):
        os.makedirs(output,exist_ok=True)
    else:
        shutil.rmtree(output)
        os.mkdir(output)
    

    df = pd.read_csv((f'CH{set["Config"]["channel"]}_pulse/output/output.csv'),index_col=0)
    rate,samples,presamples,threshold,ch = set["Config"]["rate"],set["Config"]["samples"],set["Config"]["presamples"],set["Config"]["threshold"],set["Config"]["channel"]
    time = gp.data_time(rate,samples)


    df = df[(df['samples']==samples)&(df['height']>threshold)]
    print(f'Pulse : {len(df)} samples')

    #&(df['decay']>0.001)&(df['rise']<0.0001)&(df['max_div']<0.01)&(df['decay']>0.01)
    #a = df.loc['CH0_pulse/rawdata\CH0_47388.dat']
    

    #平均パルスを取得
    if len(ax) == 1:
        y = gp.extruct(df,ax)
        x = np.arange(len(y[0]))
        plt.scatter(x,y[0],s=0.4)
        plt.xlabel("data number")
        plt.ylabel(ax_unit[ax[0]])
        plt.savefig(f'CH{ch}_pulse/output/{ax[0]}.png')
        plt.show()


    elif len(ax) == 2:
        x,y = gp.extruct(df,*ax)
        plt.scatter(x,y,s=2)
        plt.xlabel(ax_unit[ax[0]])
        plt.ylabel(ax_unit[ax[1]])
        plt.title(f"{ax[0]} vs {ax[1]}")
        plt.grid()
        plt.show()
        plt.cla()
        
        if input('exit[1]: ') == "1":
            exit()

        picked = gp.pickSamples(df,*ax) # pick samples from graugh
        print(f"Selected {len((picked))} samples.")

        if len(picked) == 1:
            path_name = picked[0]
            picked_data = gp.loadbi(path_name)
            print(df.loc[path_name]) 
            gp.graugh(path_name,picked_data,time)
            plt.show()
        else:
            print('Creating Average Pulse...')
            for i in picked:
                data = gp.loadbi(i)
                base,data = gp.baseline(data,presamples,1000,500)
                name = os.path.splitext(os.path.basename(i))[0]
                plt.plot(time,data)
                plt.xlim(0.009,0.0130)
                plt.title(name)
                #plt.yscale('log')
                plt.xlabel("time(s)")
                plt.ylabel("volt(V)")
                plt.savefig(f'CH{ch}_pulse/output/select/img/{name}.png')
                plt.cla()

            av = gp.average_pulse(picked,presamples)
            plt.plot(time,av)
            plt.xlabel("time(s)")
            plt.ylabel("volt(V)")
            plt.title("average pulse")
            plt.yscale('log')
            plt.savefig(f'CH{ch}_pulse/output/select/average_pulse.png')
            plt.show()
            np.savetxt(f'CH{ch}_pulse/output/select/selected_index.txt',picked,fmt="%s")
            np.savetxt(f'CH{ch}_pulse/output/select/average_pulse.txt',av)
    
    print('end')


#実行
if __name__=='__main__':
    main()

    

    
