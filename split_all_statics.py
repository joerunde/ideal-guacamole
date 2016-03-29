import os
"""
os.system('python statics_data_split.py compz.csv COMPO-Z-AXIS')
os.system('python statics_data_split.py drawrest.csv DRAW-ANG-VELOCITY-AT-REST')
os.system('python statics_data_split.py compzero.csv COMPO-ZERO-VECTOR')
os.system('python statics_data_split.py drawcurve.csv DRAW-VELOCITY-CURVED')
os.system('python statics_data_split.py defdur.csv DEFINE-DURATION')
os.system('python statics_data_split.py drawup.csv DRAW-ANG-ACCEL-SPEED-UP')
os.system('python statics_data_split.py drawaccel.csv DRAW-ANG-ACCEL')
os.system('python statics_data_split.py drawdown.csv DRAW-ANG-ACCEL-SLOW-DOWN')
os.system('python statics_data_split.py writeeq.csv WRITE-KNOWN-VALUE-EQN')
os.system('python statics_data_split.py writerk.csv WRITE-RK-NO-VF')
os.system('python statics_data_split.py writecvel.csv WRITE-CONNECTED-VELOCITIES')
os.system('python statics_data_split.py drawrel.csv DRAW-RELATIVE-POSITION')
os.system('python statics_data_split.py drawdisp.csv DRAW-ANG-DISPLACEMENT-ROTATING')
os.system('python statics_data_split.py writevel.csv WRITE-LINEAR-VEL')
os.system('python statics_data_split.py writeimp.csv WRITE-IMPLICIT-EQN')
os.system('python statics_data_split.py drawdispu.csv DRAW-ANG-DISPLACEMENT-UNKNOWN')
os.system('python statics_data_split.py writeang.csv WRITE-ANG-SDD')
os.system('python statics_data_split.py drawbod.csv DRAW-BODY')
os.system('python statics_data_split.py drawunr.csv DRAW-UNROTATED-AXES')
os.system('python statics_data_split.py drawrot.csv DRAW-ANG-VELOCITY-ROTATING')
os.system('python statics_data_split.py writenos.csv WRITE-RK-NO-S')
os.system('python statics_data_split.py writenot.csv WRITE-RK-NO-T')
"""

skills = ['DRAW-VELOCITY-CURVED-UNKNOWN', 'DRAW-VELOCITY-STRAIGHT', 'WRITE-SDD', 'DRAW-COMPO-FORM-AXES', 'WRITE-AVG-ACCEL-COMPO', 'WRITE-SUM-DISTANCES', 'DRAW-DISPLACEMENT-STRAIGHT', 'WRITE-FREE-FALL-ACCEL', 'COMPO-ZERO-VECTOR', 'DRAW-VELOCITY-CURVED', 'DEFINE-DURATION', 'DRAW-VECTOR-ALIGNED-AXES', 'COMPO-GENERAL-CASE', 'SDD-CONSTVEL-COMPO', 'DRAW-ACCEL-SLOW-DOWN', 'DRAW-ACCEL-SPEED-UP', 'DEFINE-DISTANCE', 'WRITE-KNOWN-VALUE-EQN', 'DRAW-ACCEL-FREE-FALL', 'DRAW-DISPLACEMENT-ZERO', 'WRITE-PYTH-THM', 'WRITE-G-ON-EARTH', 'DRAW-RELATIVE-POSITION', 'WRITE-SUM-TIMES', 'WRITE-LK-NO-VF-COMPO', 'WRITE-LK-NO-S-COMPO', 'COMPO-PERPENDICULAR', 'WRITE-IMPLICIT-EQN', 'DEFINE-SPEED', 'DRAW-AVG-VEL-FROM-DISPLACEMENT', 'DRAW-RELATIVE-POSITION-UNKNOWN', 'WRITE-AVG-VEL-COMPO', 'DRAW-UNROTATED-AXES', 'DRAW-BODY', 'WRITE-VECTOR-MAGNITUDE', 'COMPO-PARALLEL-AXIS', 'DRAW-DISPLACEMENT-PROJECTILE', 'DRAW-DISPLACEMENT-GIVEN-DIR', 'WRITE-EQUALITY', 'USE-CONST-VX', 'WRITE-LK-NO-T-COMPO', 'DRAW-VELOCITY-MOMENTARILY-AT-REST', 'DRAW-PROJECTION-AXES']

names = [a.lower().replace("-","") + ".csv" for a in skills]

for c in range(len(skills)):
    os.system('python statics_data_split.py ' + names[c] + ' ' + skills[c])

os.system('python statics_data_split.py all_translations ' + ' '.join(skills))

print names
