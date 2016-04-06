import thread, os, time

#thread.start_new_thread

def syscall(cmd, crap=None):
    os.system(cmd)

#run transition models
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py x_axis',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py y_axis',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py center',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py shape',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py spread',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py h_to_d',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py d_to_h',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff-gauss_L1.py histogram',0) )

thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py x_axis',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py y_axis',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py center',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py shape',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py spread',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py h_to_d',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py d_to_h',0) )
thread.start_new_thread(syscall,('python run_LFKT-trans-diff_L1.py histogram',0) )


time.sleep(10)
print "ASDFASDGASDFGASDFG"
print thread._count()
