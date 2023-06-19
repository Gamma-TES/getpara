import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp
import sys
import shutil
import pprint
import json

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
    "height_opt_temp":'pulse height opt temp',
    'rise':'rise[s]',
    'decay':'decay[s]',
    'rise_fit':'rise_fit[s]',
    'tau_rise':'tau_rise[s]',
    'tau_decay':'tau_decay[s]',
    'rSquared':'rSquared'
}



def main():

    ax = sys.argv
    ax.pop(0)

    if '-p' in ax:
        ax.remove('-p')

    set = gp.loadJson()

    # default condition
    para = {'select':{
    'samples-=':set['Config']['samples'],
    'height->':set['Config']['threshold'],
    'rSquared->':0,
    }}

    if not 'select' in set:
        set.update(para)
        jsn = json.dumps(set,indent=4)
        with open("setting.json", 'w') as file:
            file.write(jsn)

    os.chdir(set["Config"]["path"])

    output = f'CH{set["Config"]["channel"]}_pulse/output/{set["Config"]["output"]}'
    df = pd.read_csv((f'{output}/output.csv'),index_col=0)
    rate,samples,presamples,threshold,ch = set["Config"]["rate"],set["Config"]["samples"],set["Config"]["presamples"],set["Config"]["threshold"],set["Config"]["channel"]
    time = gp.data_time(rate,samples)
    

    # manual select
    df = gp.select_condition(df,set)
    
    print(f'Pulse : {len(df)} samples')
    #&(df['decay']>0.01)&(df['rise_fit']!=0)&(df['rise_fit'] < 100)&(df['base']>0.0)\&(df['rise_fit']<0.001)&(df['tau_decay']<10000)
    #&(df['decay']>0.001)&(df['rise']<0.0001)&(df['max_div']<0.01)&(df['decay']>0.01)
    #a = df.loc['CH0_pulse/rawdata\CH0_47388.dat']
    
    

    # time vs ax
    if len(ax) == 1:
        y = gp.extruct(df,ax)
        x = np.arange(len(y[0]))
        plt.scatter(x,y[0],s=0.4)
        plt.xlabel("data number")
        plt.ylabel(ax_unit[ax[0]])
        plt.savefig(f'{output}/{ax[0]}.png')
        plt.show()


    elif len(ax) == 2:
        x,y = gp.extruct(df,*ax)
        plt.scatter(x,y,s=2,alpha=0.7)
        plt.xlabel(ax_unit[ax[0]])
        plt.ylabel(ax_unit[ax[1]])
        plt.title(f"{ax[0]} vs {ax[1]}")
        plt.grid()
        plt.savefig(f'{output}/{ax[0]} vs {ax[1]}.png')
        plt.show()
        plt.cla()
        
        if input('exit[1]: ') == "1":
            exit()
        out_select = input('output name:')
        if not 'output' in set['select']:
            set['select']['output'] = out_select
            jsn = json.dumps(set,indent=4)
            with open(f"{__file__}/../setting.json", 'w') as file:
                file.write(jsn)


        output_f = f'{output}/{out_select}'
        if not os.path.exists(f"{output_f}/img"):
            os.makedirs(f"{output_f}/img",exist_ok=True)
        else:
            shutil.rmtree(f"{output_f}/img")
            os.mkdir(f"{output_f}/img")

        picked = gp.pickSamples(df,*ax).tolist() # pick samples from graugh
        print(f"Selected {len((picked))} samples.")


        if len(picked) == 1:
            path_name = picked[0]
            picked_data = gp.loadbi(path_name)
            print(df.loc[path_name]) 
            gp.graugh(path_name,picked_data,time)
            plt.show()
        else:
            # average pulse
            for i in picked:
                data = gp.loadbi(i)
                data = gp.BesselFilter(data,rate,fs = set['main']['cutoff'])
                base,data = gp.baseline(data,presamples,1000,500)
                name = os.path.splitext(os.path.basename(i))[0]
                plt.plot(time,data)
                #plt.xlim(0.009,0.0130)
                plt.title(name)
                #plt.yscale('log')
                plt.xlabel("time(s)")
                plt.ylabel("volt(V)")
                plt.savefig(f'{output_f}/img/{name}.png')
                plt.cla()
                print(name)

            np.savetxt(f'{output_f}/selected_index.txt',picked,fmt="%s")

            
            num = 1
            while num != 0:
                num = int(input("delete pulse number (finish [0]): "))
                try:
                    picked.remove(f'CH{ch}_pulse/rawdata\\CH{ch}_{num}.dat')
                    os.remove(f'{output_f}/img/CH{ch}_{num}.png')
                    np.savetxt(f'{output_f}/selected_index.txt',picked,fmt="%s")
                except:
                    print("Not exist file")

            

            # create average pulse
            print('Creating Average Pulse...')
            array = []
            for i in picked:
                data = gp.loadbi(i)
                filt = gp.BesselFilter(data,rate,set['main']['cutoff'])
                base,data = gp.baseline(filt,presamples,1000,500)
                array.append(data)
            av = np.mean(array,axis=0)

            plt.plot(time,av)
            plt.xlabel("time(s)")
            plt.ylabel("volt(V)")
            plt.title("average pulse")
            plt.savefig(f'{output_f}/average_pulse.png')
            plt.show()

            plt.cla()
            plt.plot(time,av)
            plt.xlabel("time(s)")
            plt.ylabel("volt(V)")
            plt.title("average pulse")
            plt.yscale('log')
            plt.savefig(f'{output_f}/average_pulse_log.png')
            plt.show()

            np.savetxt(f'{output_f}/average_pulse.txt',av)
    
    print('end')


#実行
if __name__=='__main__':
    main()

    

    
