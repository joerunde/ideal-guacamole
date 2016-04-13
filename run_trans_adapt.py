import thread, os, time

#thread.start_new_thread

def syscall(cmd, crap=None):
    os.system(cmd)

#run transition models

thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans.py x_axis',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans.py y_axis',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans.py center',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans.py shape',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans.py spread',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans.py h_to_d',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans.py d_to_h',0) )
thread.start_new_thread(syscall,('python run_LFKT-adaptive-trans.py histogram',0) )


time.sleep(10)
print "ASDFASDGASDFGASDFG"
print thread._count()
