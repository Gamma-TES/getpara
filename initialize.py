import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
from natsort import natsorted
import libs.getpara as gp
import libs.fft_spectrum as sp
import json
from tkinter import filedialog
import sys
import re

# Json形式へ変更

def loadJson():
    with open("setting.json") as f:
        jsn = json.load(f)
    return jsn

def main():

    ax = sys.argv

    if len(ax) > 1:
        if ax[1] == "-p":
            print("Post mode")
        ch = ax[2:]
        print(ch)
    

    path = filedialog.askdirectory()
    ch = input('ch: ')
    output = input("output name:")
    post_ch = glob.glob(f'{path}/CH*_pulse')

    setting = np.loadtxt(f"{path}/Setting.txt",skiprows = 10)
    setting_json = {
        "Config":{
            "path" : path,
            "channel":ch,
            "rate" : int(setting[2]),
            "samples" : int(setting[4]),
            "presamples" : int(setting[5]),
            "threshold" : setting[6],
            "output" : output
        }
        
    }
    set = json.dumps(setting_json,indent=4)
    
    with open("setting.json", 'w') as file:
        file.write(set)

    # output
    
    for i in post_ch:
        ch = int(re.sub(r"\D", "", i))
        pulse_output = f'{path}/CH{ch}_pulse/output/{output}'
        if not os.path.exists(pulse_output):
            os.makedirs(pulse_output,exist_ok=True)
        else:
            replace = input('Replace pulse output folder? (Yes -> [0], No (not save) -> [1]): ')
            if replace =='0':
                shutil.rmtree(pulse_output)
                os.makedirs(pulse_output,exist_ok=True)

    output_noise = f'{path}/CH{ch}_noise/output/{output}'
    if not os.path.exists(output_noise):
        os.makedirs(output_noise,exist_ok=True)
    else:
        replace = input('Replace noise output folder? (Yes -> [0], No (not save) -> [1]): ')
        if replace =='0':
            shutil.rmtree(output_noise)
            os.makedirs(output_noise,exist_ok=True)

    set = json.loads(set)
    print(set)
    RED = '\033[33m'
    END = '\033[0m'
    print(RED + "Hello GetPara!" + END)


if __name__ == "__main__":
    main()