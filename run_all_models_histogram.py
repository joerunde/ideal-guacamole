import thread, os

#thread.start_new_thread

def syscall(cmd):
    os.system(cmd)



#run BKT models
thread.start_new_thread(syscall,('python run_BKT.py x_axis y_axis h_to_d d_to_h center shape spread histogram xy css descrip whole_tutor') )
thread.start_new_thread(syscall,('python run_BKT-4.py x_axis y_axis h_to_d d_to_h center shape spread histogram xy css descrip whole_tutor') )


#run basic LFKT models
thread.start_new_thread(syscall,('python run_LFKT.py x_axis y_axis h_to_d d_to_h center shape spread histogram xy css descrip whole_tutor') )
thread.start_new_thread(syscall,('python run_LFKT-4.py x_axis y_axis h_to_d d_to_h center shape spread histogram xy css descrip whole_tutor') )
thread.start_new_thread(syscall,('python run_LFKT_L1.py x_axis y_axis h_to_d d_to_h center shape spread histogram xy css descrip whole_tutor') )


#run skill difficulty model
thread.start_new_thread(syscall,('python run_LFKT-skills_L1.py xy css descrip whole_tutor') )


#run transition models
thread.start_new_thread(syscall,('python run_LFKT-trans_L1.py x_axis y_axis h_to_d d_to_h center shape spread histogram xy css descrip whole_tutor') )
thread.start_new_thread(syscall,('python run_LFKT-trans-first_L1.py x_axis y_axis h_to_d d_to_h center shape spread histogram xy css descrip whole_tutor') )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py x_axis y_axis h_to_d d_to_h center shape spread histogram') )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-first_L1.py x_axis y_axis h_to_d d_to_h center shape spread histogram') )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py xy css descrip whole_tutor') )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-first_L1.py xy css descrip whole_tutor') )

#run knowledge transfer models
thread.start_new_thread(syscall,('python run_LFKT-useful_L1.py xy css descrip') )
thread.start_new_thread(syscall,('python run_LFKT-useful-diff_L1.py xy css descrip') )
thread.start_new_thread(syscall,('python run_LFKT-useful_L1.py whole_tutor') )
thread.start_new_thread(syscall,('python run_LFKT-useful-diff_L1.py whole_tutor') )
