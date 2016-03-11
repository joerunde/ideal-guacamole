import os, sys


for c in range(2,6):
    os.system("python plot_difficulties.py " + str(c) + " css center shape spread")
    os.system("python plot_difficulties.py " + str(c) + " xy x_axis y_axis")
    os.system("python plot_difficulties.py " + str(c) + " descrip h_to_d d_to_h")
    os.system("python plot_difficulties.py " + str(c) + " circleall circlediam circlecir circlearea")