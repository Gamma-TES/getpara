import matplotlib.pyplot as plt
from tkinter import filedialog
import getpara as gp
import numpy as np
import os


def main():
	setting = gp.loadJson()
	config = setting["Config"]
	_main = setting["main"]
	presamples = config["presamples"]
	
	path1 = filedialog.askopenfilename()
	path2 = filedialog.askopenfilename()
	data1 = np.loadtxt(path1)
	data2 = np.loadtxt(path2)

	time = gp.data_time(config["rate"],config["samples"])
	plt.plot(time*1000,data1)
	plt.plot(time*1000,data2)
	plt.xlabel("time [ms]")
	plt.ylabel("volt [V]")
	plt.savefig(f"{os.path.dirname(path1)}/double_plot.png")
	plt.show()

main()