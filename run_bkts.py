import thread, os

#thread.start_new_thread

def syscall(cmd, crap=None):
    os.system(cmd)



#run BKT models
thread.start_new_thread(syscall, ('python run_BKT.py x_axis',0) )
thread.start_new_thread(syscall, ('python run_BKT.py y_axis',0) )
thread.start_new_thread(syscall, ('python run_BKT.py h_to_d',0) )
thread.start_new_thread(syscall, ('python run_BKT.py d_to_h',0) )
thread.start_new_thread(syscall, ('python run_BKT.py center',0) )
thread.start_new_thread(syscall, ('python run_BKT.py shape',0) )
thread.start_new_thread(syscall, ('python run_BKT.py spread',0) )
thread.start_new_thread(syscall, ('python run_BKT.py histogram ',0) )


import time
time.sleep(10)

print thread._count()
