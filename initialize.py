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

# Json形式へ変更
def setting_json(path,ch):
    setting = np.loadtxt("Setting.txt",skiprows = 10)
    setting_json = {
        "Config":{
            "path" : path,
            "channel":ch,
            "rate" : int(setting[2]),
            "samples" : int(setting[4]),
            "presamples" : int(setting[5]),
            "threshold" : setting[6],
        }
        
    }
    set = json.dumps(setting_json,indent=4)
    print("Hello")
    with open("C:/Users/gamma/matsumi/scripts/setting.json", 'w') as file:
        file.write(set)

    return set

def loadJson():
    with open("setting.json") as f:
        jsn = json.load(f)
    return jsn

def main():
    path = input('Path: ')
    ch = input('Ch: ')
    os.chdir(path)
    set_json = setting_json(path,ch)
    set = json.loads(set_json)
    print(set_json)

if __name__ == "__main__":
    main()