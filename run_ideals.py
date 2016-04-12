import thread, os, time

#thread.start_new_thread

def syscall(cmd, crap=None):
    os.system(cmd)

#run over the skills
thread.start_new_thread(syscall,('python run_KT_IDEAL.py x_axis y_axis h_to_d',0) )
thread.start_new_thread(syscall,('python run_KT_IDEAL.py d_to_h center spread',0) )
thread.start_new_thread(syscall,('python run_KT_IDEAL.py shape histogram',0) )

print thread._count()
