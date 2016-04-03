import thread, os, time

#thread.start_new_thread

def syscall(cmd, crap=None):
    os.system(cmd)

#run transition models
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py x_axis y_axis h_to_d d_to_h center shape spread histogram',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss-first_L1.py x_axis y_axis h_to_d d_to_h center shape spread histogram',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py xy css descrip whole_tutor',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss-first_L1.py xy css descrip whole_tutor',0) )

#run knowledge transfer models
thread.start_new_thread(syscall,('python run_LFKT-useful_L1.py xy css descrip',0) )
thread.start_new_thread(syscall,('python run_LFKT-useful-diff_L1.py xy css descrip',0) )
thread.start_new_thread(syscall,('python run_LFKT-useful_L1.py whole_tutor',0) )
thread.start_new_thread(syscall,('python run_LFKT-useful-diff_L1.py whole_tutor',0) )



#do the KT stuff twice, to get some more chainz
time.sleep(1)
thread.start_new_thread(syscall,('python run_LFKT-useful_L1.py xy css descrip',0) )
thread.start_new_thread(syscall,('python run_LFKT-useful-diff_L1.py xy css descrip',0) )
thread.start_new_thread(syscall,('python run_LFKT-useful_L1.py whole_tutor',0) )
thread.start_new_thread(syscall,('python run_LFKT-useful-diff_L1.py whole_tutor',0) )


print thread._count()
