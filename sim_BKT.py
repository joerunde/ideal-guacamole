import os


for student in [10,25,50,75,100,500]:
    os.system('python fit_MLFKT_model.py 100 1000 2 y simulated_trans_' + str(student) + ' 0 1 L1')
    os.system('python fit_MLFKT_model.py 100 1000 2 y simulated_sep0_KT2_' + str(student) + ' 0 1 L1')

