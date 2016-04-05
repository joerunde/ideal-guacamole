import thread, os, time

#thread.start_new_thread

def syscall(cmd, crap=None):
    os.system(cmd)

#run adaptive transition models
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py x_axis',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py y_axis',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py center',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py shape',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py spread',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py h_to_d',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py d_to_h',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py histogram',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py xy',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py css',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py descrip',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans_L1.py whole_tutor',0) )


print thread._count()
