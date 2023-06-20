import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp
import sys
import shutil
import pprint
import json


ax_unit = {
    "base":'base[V]',
    "height":'pulse height[V]',
    "peak_index":'peak index',
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



    path =  "E:/tsuruta/20230616/room1-ch2-3_180mK_570uA_100kHz_g10"
    os.chdir(path)
    ch0 =  pd.read_csv((f'CH0_pulse/output/1/output.csv'),index_col=0)
    ch1 =  pd.read_csv((f'CH1_pulse/output/1/output.csv'),index_col=0)
    
    x = ch0['height']
    y = ch1['height']
    print(len(x))
    print(len(y))

    plt.scatter(x,y,s=0.4)
    plt.grid()
    plt.show()
    """
    df = pd.read_csv((f'{output}/output.csv'),index_col=0)
    rate,samples,presamples,threshold,ch = set["Config"]["rate"],set["Config"]["samples"],set["Config"]["presamples"],set["Config"]["threshold"],set["Config"]["channel"]
    time = gp.data_time(rate,samples)
    

    # manual select
    df = gp.select_condition(df,set)
    
    print(f'Pulse : {len(df)} samples')
 
    

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
        """

#実行
if __name__=='__main__':
    main()