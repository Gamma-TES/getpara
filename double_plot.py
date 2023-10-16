import matplotlib.pyplot as plt
from tkinter import filedialog
import getpara as gp
import numpy as np
import os
import natsort
import matplotlib.cm as cm


def main():
	setting = gp.loadJson()
	config = setting["Config"]
	_main = setting["main"]
	presamples = config["presamples"]
	
	
	paths = natsort.natsorted(filedialog.askopenfilenames())
	
	cnt=0
	for i in paths:
		data1 = np.loadtxt(i)

		time = gp.data_time(config["rate"],config["samples"])
		plt.plot(time*1000,data1,color = cm.hsv((float(cnt))/float(len(paths))))
		cnt+=1
	plt.xlabel("time [ms]")
	plt.ylabel("volt [V]")

	#plt.savefig(f"{os.path.dirname(paths)}/double_plot.png")
	plt.show()

main()