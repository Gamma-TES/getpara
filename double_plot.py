import matplotlib.pyplot as plt
from tkinter import filedialog
import getpara as gp
import numpy as np
def main():
	setting = gp.loadJson()
	config = setting["Config"]
	_main = setting["main"]
	presamples = config["presamples"]
	
	path1 = filedialog.askopenfilename()
	path2 = filedialog.askopenfilename()
	data1 = gp.loadtxt(path1)
	data2 = gp.loadtxt(path2)
	time = np.linspace(0,0.2,len(data1))
	plt.plot(time,data1)
	plt.plot(time,data2)
	plt.show()

main()