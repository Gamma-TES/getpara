import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
import pandas as pd
import libs.getpara as gp
import json

BIN = 4096

def main():

    # Load Setting 
    set = gp.loadJson()
    path,ch = set["Config"]["path"],set["Config"]["channel"]
    os.chdir(path)

    jsn = json.dumps(set,indent=4)
    with open(f'CH{ch}_pulse/output/setting.json', 'w') as file:
        file.write(jsn)

    # Load data and transform histgrum
    df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
    df = gp.select_condition(df,set)
    data = df["height_opt_temp"]
    hist,bins = np.histogram(data,bins=BIN,range=[0,np.max(data)*1.05])

    plt.bar(bins[:-1],hist,width = bins[1]-bins[0])
    plt.xlabel('Pulse Height [ch]')
    plt.ylabel('Counts')
    plt.tick_params(axis='both', which='both', direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.2)
    plt.grid(True, which='minor', color='black', linestyle=':', linewidth=0.1)
    plt.savefig(f'CH{ch}_pulse/output/select/histgrum.png')
    plt.show()


if __name__ == "__main__":
    main()
    print('end')