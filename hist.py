import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import libs.getpara as gp
import sys
import os


# histgrum for selected datas
# Give some index files as arguments 
# ex) python .\hist.py selected_index_fast.txt selected_index.txt


def main():
    files = sys.argv
    files.pop(0)
    print(files)
    set = gp.loadJson()
    ch = set["Config"]["channel"]
    path = set["Config"]["path"]
    os.chdir(path)
    df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)

    for i in files:
        index = gp.loadIndex(f"CH{ch}_pulse/output/{i}")
        data = df.loc[index,'height_opt_temp']
        plt.hist(data,bins = 40,label=i)
        plt.xlabel('pulseheight[V]',fontsize = 16)
        plt.ylabel('count[-]',fontsize = 16)
        plt.legend()
        plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

