import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import getpara as gp
import sys
import shutil
import pprint
import json
import glob
import re
import tkinter as tk
from tkinter import filedialog
import tqdm
import platform
import plt_config
import natsort
import glob
import matplotlib.cm as cm


def main():
	setting = gp.loadJson()
	config = setting['Config']
	path = config['path']
	output = config['output']
	ch = config['channel']
	para = setting['main']
	time = gp.data_time(rate=config['rate'],samples=config['samples'])*1e3
	os.chdir(path)

	'''
	df_0 = pd.read_csv(f'CH0_pulse/output/{output}/output.csv')
	df_1 = pd.read_csv(f'CH1_pulse/output/{output}/output.csv')

	df_0_clear = gp.select_condition(df_0, setting["select"])
	df_1_clear = gp.select_condition(df_1, setting["select"])

	df_0_over, df_1_over = gp.overlap(df_0_clear, df_1_clear)
	x, y = df_0_over['height'].values, df_1_over['height'].values
	'''
	
	blocks_path_0 = natsort.natsorted(glob.glob(f'CH0_pulse/output/{output}/blocks/average_pulse_*.txt'))
	blocks_path_1 = natsort.natsorted(glob.glob(f'CH1_pulse/output/{output}/blocks/average_pulse_*.txt'))
	

	for i,j in zip(blocks_path_0,blocks_path_1):
		block_0 = np.loadtxt(i)
		block_1 = np.loadtxt(j)
		height_0 = gp.peak(block_0,config["presamples"],para['peak_max'],para['peak_x'],para['peak_w'])
		height_1 = gp.peak(block_1,config["presamples"],para['peak_max'],para['peak_x'],para['peak_w'])
		print(height_1[1])
		ratio = height_0[1]/height_1[1]
		plt.plot(time,block_0,label = 'CH0')
		plt.plot(time,block_1,label = 'CH1')
		plt.xlabel("time [ms]")
		plt.ylabel("volt [V]")
		gp.graugh_condition(setting["graugh"])
		plt.grid()
		plt.legend(fontsize=12)
		plt.tight_layout()
		plt.show()


	b = 0
	for i,j in zip(blocks_path_0,blocks_path_1):
		block_0 = gp.loadbi(i,"text")
		block_1 = gp.loadbi(j,"text")
		height_0 = gp.peak(block_0,config["presamples"],para['peak_max'],para['peak_x'],para['peak_w'])
		height_1 = gp.peak(block_1,config["presamples"],para['peak_max'],para['peak_x'],para['peak_w'])
		print(height_1[1])
		ratio = height_0[1]/height_1[1]
		plt.scatter(b+1,ratio,c=cm.hsv(float(b)/float(len(blocks_path_0))))
		b += 1
		
	block = np.arange(1,12,1)
	#plt.plot(x_fit, y_fit, "--")
	# plt.title("pixel vs pulseheight1/ pulseheight2")
	plt.ylabel("pulseheight0/ pulseheight1")
	plt.xlabel("block")
	#plt.ylim(0-1,setting['length']+1)
	plt.xticks(block)
	plt.grid()
	plt.tight_layout()
	#plt.savefig(f"{out_path}/height_ratio.png", dpi=350)
	plt.show()
	


if __name__ == "__main__":
	main()