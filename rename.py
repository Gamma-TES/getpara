import os
import glob
import numpy as np
from natsort import natsorted

ch = 0
start = 144344 #　変更後のファイルの始めの番号
data = "noize"
path = 'H:/Matsumi/data/20230424/room2-2_140mK_870uA_gain10_trig0.1_10kHz'


def main():
    os.chdir(f'{path}/CH{ch}_{data}/rawdata')
    files = natsorted(glob.glob("*.dat")) 
    number = np.arange(start,start + len(files),1)

    # ファイル名を変更時、ディレクトリ内に同名のファイルを作らないように仮名に変更
    num = 0
    for i in files:
        os.rename(i,f"{number[num]}.dat")
        num += 1
        print(i)
    
    # 正しいファイルをに変更
    files = natsorted(glob.glob("*.dat")) 
    num = 0
    for i in files:
        os.rename(i,f"CH{ch}_{number[num]}.dat")
        num += 1
        print(i)

    print(files)

if __name__ == "__main__":
    main()





