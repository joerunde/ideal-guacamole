import thread, os, time

#thread.start_new_thread

def syscall(cmd, crap=None):
    os.system(cmd)

#run over the skills
thread.start_new_thread(syscall,('python run_KT_IDEAL_FIRST.py x_axis y_axis h_to_d',0) )
thread.start_new_thread(syscall,('python run_KT_IDEAL_FIRST.py d_to_h center spread',0) )
thread.start_new_thread(syscall,('python run_KT_IDEAL_FIRST.py shape histogram',0) )

time.sleep(10)

print thread._count()
