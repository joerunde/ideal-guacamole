import thread, os, time

#thread.start_new_thread

def syscall(cmd, crap=None):
    os.system(cmd)

for student in [10,25,50,75,100,500]:
    thread.start_new_thread(syscall,('python fit_MLFKT_usefulness_penalty_model.py 100 1000 2 n simulated_KT2_' + str(student) + ' 0 1 L1 0.5', 0))
    #os.system('python fit_MLFKT_transition_model.py 100 1000 2 n simulated_KT2_' + str(student) + ' 0 1 L1')

time.sleep(5)
print thread._count()