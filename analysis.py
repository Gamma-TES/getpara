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

arrival_threshold = 0.01

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

	block = np.arange(1,12,1)
	
	k = 1
	arrival_diffs = []

	for i,j in zip(blocks_path_0,blocks_path_1):
		ch_0 = np.loadtxt(i)
		ch_1 = np.loadtxt(j)
		height_0 = gp.peak(ch_0,config["presamples"],para['peak_max'],para['peak_x'],para['peak_w'])
		height_1 = gp.peak(ch_1,config["presamples"],para['peak_max'],para['peak_x'],para['peak_w'])

		ch_1 = ch_1 * 1.6345
		
		arrival_0 = gp.arrival_time(ch_0[6000:],arrival_threshold)+6000
		arrival_1 = gp.arrival_time(ch_1[6000:],arrival_threshold)+6000
		arrival_diff = (arrival_0 - arrival_1)
		arrival_diffs.append(arrival_diff/config['rate'])
		print(k)
		print(height_0[1]/height_1[1])
		print(arrival_diff/config['rate'])
		ratio = height_0[1]/height_1[1]
		plt.plot(time,ch_0,label = 'CH0')
		plt.plot(time,ch_1,label = 'CH1')
		plt.scatter(time[arrival_0],ch_0[arrival_0])
		plt.scatter(time[arrival_1],ch_1[arrival_1])
		plt.title(f"block{k}")
		plt.xlabel("time [ms]")
		plt.ylabel("volt [V]")
		gp.graugh_condition(setting["graugh"])
		plt.grid()
		plt.legend(fontsize=12)
		plt.tight_layout()
		plt.savefig(f'CH0_pulse/output/{output}/blocks/block{k}.png')
		plt.show()
		plt.cla()
		k+=1
	
	arrival_diffs = np.array(arrival_diffs)*1e3
	plt.scatter(block,arrival_diffs)
	plt.ylabel("arrival time [ms]")
	plt.xlabel("block")
	#plt.ylim(0-1,setting['length']+1)
	plt.xticks(block)
	plt.grid()
	plt.tight_layout()
	plt.savefig(f'CH0_pulse/output/{output}/blocks/arrival_time.png')
	plt.show()
	


	b = 0
	for i,j in zip(blocks_path_0,blocks_path_1):
		block_0 = gp.loadbi(i,"text")
		block_1 = gp.loadbi(j,"text")
		height_0 = gp.peak(block_0,config["presamples"],para['peak_max'],para['peak_x'],para['peak_w'])
		height_1 = gp.peak(block_1,config["presamples"],para['peak_max'],para['peak_x'],para['peak_w'])
		ratio = height_0[1]/height_1[1]
		plt.scatter(b+1,ratio,c=cm.hsv(float(b)/float(len(blocks_path_0))))
		b += 1
		
	
	#plt.plot(x_fit, y_fit, "--")
	# plt.title("pixel vs pulseheight1/ pulseheight2")
	plt.ylabel("pulseheight0/ pulseheight1")
	plt.xlabel("block")
	#plt.ylim(0-1,setting['length']+1)
	plt.xticks(block)
	plt.grid()
	plt.tight_layout()
	plt.savefig(f'CH0_pulse/output/{output}/blocks/PH_PH_ratio.png')
	plt.show()
	


if __name__ == "__main__":
	main()