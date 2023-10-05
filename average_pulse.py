import getpara as gp 
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def main():
	ax = sys.argv
	ax.pop(0)

	setting = gp.loadJson()
	config = setting["Config"]
	_main = setting["main"]
	presamples = config["presamples"]
	time = gp.data_time(config['rate'],config['samples'])
	os.chdir(config["path"])

	_path = filedialog.askopenfilename(filetypes=[('index file','*.txt')])
	index = gp.loadIndex(_path)
	output = os.path.dirname(_path)

	print(index)
	ch = input("cahnnel: ")
	output_path = f'CH{ch}_pulse/output/{config["output"]}/output.csv'
	df = pd.read_csv(output_path)
	df = gp.select_condition(df,set)

	x,y = gp.extruct(df,*ax)
	x_sel,y_sel =gp.extruct(df.loc[index],*ax)
	plt.scatter(x,y,s=2,alpha=0.7)
	plt.scatter(x_sel,y_sel,s=4)
	plt.xlabel(gp.ax_unit[ax[0]])
	plt.ylabel(gp.ax_unit[ax[1]])
	plt.title(f"{ax[0]} vs {ax[1]}")
	plt.grid()
	plt.savefig(f'{output}/{ax[0]} vs {ax[1]}_sel_ch{ch}.png')
	plt.show()
	plt.cla()

	array = []
	for num in index:
		path = f'CH{ch}_pulse/rawdata/CH{ch}_{num}.dat'
		data = gp.loadbi(path)
		base,data = gp.baseline(data,presamples,_main["base_x"],_main["base_w"])
		array.append(data)
	av = np.mean(array,axis=0)

	
	np.savetxt(f'{output}/selected_average_pulse.txt',av)
	plt.plot(time,av)
	gp.graugh_condition(setting)
	plt.xlabel("time(s)")
	plt.ylabel("volt(V)")
	plt.title("average pulse")
	plt.savefig(f'{output}/selected_average_pulse.png')
	plt.show()
	plt.cla()

if __name__ == "__main__":
	main()