import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp
import sys
import shutil
import pprint
import json
import glob
import re
from tkinter import filedialog

# main.py　の後　output.csv　を用いてパラメータの相関をみる
# 
# 実行例--------------
# ex) python selectdata.py rise height
# exit[1]: 1 (終了)
# exit[1]: Enter (continue)
# 囲んだ所のパルスを抽出、平均パルスを作成
#
# setting.json　を用いてプロットするデータの条件を絞ることができる
# ex) "base->":0
# ex) "rise-<":0.01
# ex) "samples-=":100000



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

    ax = sys.argv
    ax.pop(0)

    

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

    if '-p' in ax:
        mode = "post"
        ax.remove('-p')
        if ch == '0':
            ch_p = '1'
        elif ch == '1':
            ch_p = '0'
        output2 = f'CH{ch_p}_pulse/output/{set["Config"]["output"]}'
        df2 = pd.read_csv((f'{output2}/output.csv'),index_col=0)
    else:
        mode = "single"
    

    # manual select
    df_clear = gp.select_condition(df,set)
    
    print(f'Pulse : {len(df_clear)} samples')
       
    

    # time vs ax
    if len(ax) == 1:
        y = gp.extruct(df_clear,ax)
        x = np.arange(len(y[0]))
        plt.scatter(x,y[0],s=0.4)
        plt.xlabel("data number")
        plt.ylabel(ax_unit[ax[0]])
        plt.savefig(f'{output}/{ax[0]}.png')
        plt.show()


    elif len(ax) == 2:
        x,y = gp.extruct(df_clear,*ax)
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
        set['select']['output'] = out_select
        jsn = json.dumps(set,indent=4)
        print(jsn)
        with open(f"{__file__}/../setting.json", 'w') as file:
            file.write(jsn)


        output_f = f'{output}/{out_select}'
        if not os.path.exists(f"{output_f}/img"):
            os.makedirs(f"{output_f}/img",exist_ok=True)
        else:
            shutil.rmtree(f"{output_f}/img")
            os.mkdir(f"{output_f}/img")

        picked = gp.pickSamples(df_clear,*ax).tolist() # pick samples from graugh
        print(f"Selected {len((picked))} samples.")

        
        x_sel,y_sel =gp.extruct(df_clear.loc[picked],*ax)
        plt.scatter(x,y,s=2,alpha=0.7)
        plt.scatter(x_sel,y_sel,s=4)
        plt.xlabel(ax_unit[ax[0]])
        plt.ylabel(ax_unit[ax[1]])
        plt.title(f"{ax[0]} vs {ax[1]}")
        plt.grid()
        plt.savefig(f'{output_f}/{ax[0]} vs {ax[1]}_sel_ch{ch}.png')
        plt.show()
        plt.cla()

        if len(picked) == 1:
            path_name = picked[0]
            picked_data = gp.loadbi(path_name)
            print(df_clear.loc[path_name]) 
            gp.graugh(path_name,picked_data,time)
            plt.show()
        else:
            # average pulse
            for i in picked:
                path = f'CH{ch}_pulse/rawdata\\CH{ch}_{i}.dat'
                data = gp.loadbi(path)
                data = gp.BesselFilter(data,rate,fs = set['main']['cutoff'])
                base,data = gp.baseline(data,presamples,1000,500)
                name = os.path.splitext(os.path.basename(path))[0]
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
            
           
            if mode == "post":
                sel_num = []
                """
                for i in picked:
                    path_data = f"CH{ch_p}_pulse/rawdata\CH{ch_p}_{i}.dat"
                    sel_num.append(path_data)
                """
                df_sel2 = df2.loc[picked]
                x2,y2 = gp.extruct(df2,*ax)
                x2_sel,y2_sel = gp.extruct(df_sel2,*ax)
                plt.scatter(x2,y2,s=2,alpha=0.7)
                plt.scatter(x2_sel,y2_sel,s=4)
                plt.xlabel(ax_unit[ax[0]])
                plt.ylabel(ax_unit[ax[1]])
                plt.title(f"{ax[0]} vs {ax[1]}")
                plt.grid()
                plt.savefig(f'{output_f}/{ax[0]} vs {ax[1]}_sel_ch{ch_p}.png')
                plt.show()
                plt.cla()


            # delete noise data
            fle = filedialog.askopenfilenames(filetypes=[('画像ファイル','*.png')],initialdir=f"{output_f}/img")
            for f in fle:
                num =  int(re.findall(r'\d+', os.path.basename(f))[1])              
                picked.remove(num)
                os.remove(f'{output_f}/img/CH{ch}_{num}.png')
                np.savetxt(f'{output_f}/selected_index.txt',picked,fmt="%s")
                df.at[num,"quality"] = 0

            

            # create average pulse
            print('Creating Average Pulse...')
            array = []
            for i in picked:
                path = f'CH{ch}_pulse/rawdata\\CH{ch}_{i}.dat'
                data = gp.loadbi(path)
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
    
    
    jsn = json.dumps(set,indent=4)
    with open(f'{output_f}/setting.json', 'w') as file:
        file.write(jsn)
    df.to_csv(f'{output}/output.csv')


#実行
if __name__=='__main__':
    main()
    print('end')

    

    
