from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint
import math

def f(v,w,x,y,z):
        return (x+y)**2 + 4*x -2*y + math.sin(v) + 0.3 * math.cos(z) + (w-3)**2

exp = Experiment([[-10,10],[-10,10],[-10,10],[-10,10],[-10,10]])

for c in range(100):
    try:
        x = gp_next_points(exp)
    except:
        print "500"
        continue
    y = f(x[0][0], x[0][1], x[0][2], x[0][3], x[0][4])
    print x, y
    exp.historical_data.append_sample_points([SamplePoint(x,y,0.001)])
