import os


for student in [10,25,50,75,100,500]:
    os.system('python fit_KT_IDEAL.py 100 1000 2 n simulated_trans_' + str(student) + ' 0 1')

