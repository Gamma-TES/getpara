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

def loadJson():
    with open("setting.json") as f:
        jsn = json.load(f)
    return jsn

def main():

    path = input('path: ')
    ch = input('ch: ')
    output = input("output name:")
    

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
    
    with open("C:/Users/gamma/matsumi/getpara/setting.json", 'w') as file:
        file.write(set)

    set = json.loads(set)
    print(set)
    RED = '\033[33m'
    END = '\033[0m'
    print(RED + "Hello GetPara!" + END)


if __name__ == "__main__":
    main()