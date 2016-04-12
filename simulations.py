#simulate different student situations
import numpy as np
import random

def print_obs(obs, skill):
    f = open('dump/observations_simulated_' + skill + '.csv', 'w')
    for obsl in obs:
        obsl = map(str, obsl)
        f.write(','.join(obsl))
        f.write(',-1,-1\n')
    f.close()

def print_states(obs, skill):
    f = open('dump/states_simulated_' + skill + '.csv', 'w')
    for obsl in obs:
        obsl = map(str, obsl)
        f.write(','.join(obsl))
        f.write(',-1,-1\n')
    f.close()

def print_probs(obs, skill):
    f = open('dump/problems_simulated_' + skill + '.csv', 'w')
    for obsl in obs:
        obsl = map(str, obsl)
        f.write(','.join(obsl))
        f.write(',-1,-1\n')
    f.close()

def print_skills(obs, skill):
    f = open('dump/skills_simulated_' + skill + '.csv', 'w')
    for obsl in obs:
        obsl = map(str, obsl)
        f.write(','.join(obsl))
        f.write(',-1,-1\n')
    f.close()




#simulate different size data sets

for students in [10, 25, 50, 75, 100, 500]:

    #do modified transition stuff
    transobs = []
    transprobs = []
    transstates = []
    #twice the students, we'll split half for train/test
    for c in range(students * 2):
        transobs.append([])
        transprobs.append([])
        transstates.append([])
        a = np.random.random()
        if a < 0.1:
            z = 1
        else:
            z = 0

        #approximately expit( laplace(-2, 0.5, 10) )
        tps = [ 0.06,  0.01,  0.085,  0.30,  0.16, 0.09,  0.20,  0.11,  0.11,  0.11]

        probs = [0,1,2,3,4,5,6,7,8,9]
        random.shuffle(probs)
        for prob in probs:
            a = np.random.random()
            if (z and a < .1 ) or (z == 0 and a > .1):
                transobs[-1].append(0)
            else:
                transobs[-1].append(1)

            transprobs[-1].append(prob)

            transstates[-1].append(z)

            tp = tps[prob]

            t = np.random.random()
            if t < tp:
                z = 1

    print_obs(transobs, 'trans_'+str(students))
    print_probs(transprobs, 'trans_'+str(students))
    print_states(transstates, 'trans_'+str(students))




    # shared knowledge time
    skillobs = []
    skillskills = []
    skillprobs = []

    sepobs = {}
    sepprobs = {}
    sepstates = {}
    for j in range(2):
        sepobs[j] = []
        sepprobs[j] = []
        sepstates[j] = []


    for c in range(students*2):
        skillobs.append([])
        skillskills.append([])
        skillprobs.append([])
        for j in range(2):
            sepobs[j].append([])
            sepprobs[j].append([])
            sepstates[j].append([])

        z = [0,0]

        """for j in [0,1]:
            a = np.random.random()
            if a < 0.1:
                z[j] = 1
            else:
                z[j] = 0"""

        #print z
        skills = [0,1] * 15
        random.shuffle(skills)

        for sk in skills:
            #print z, sk

            guess = .05
            slip = .33

            st = z[sk]

            if sk == 0 and z[1] or sk == 1 and z[0]:
                guess = 0.33
                slip = 0.05

            a = np.random.random()
            if (st == 1 and a < slip) or (st == 0 and a > guess):
                skillobs[-1].append(0)
                sepobs[sk][-1].append(0)
            else:
                skillobs[-1].append(1)
                sepobs[sk][-1].append(1)

            prob = np.random.randint(15)

            skillskills[-1].append(sk)
            skillprobs[-1].append(prob)
            sepprobs[sk][-1].append(prob)
            sepstates[sk][-1].append(st)

            tp = 0.1

            t = np.random.random()
            if t < tp:
                z[sk] = 1

        #print skillobs[-1]

    print_obs(skillobs, 'KT2_'+str(students))
    print_skills(skillskills, 'KT2_'+str(students))
    print_probs(skillprobs, 'KT2_'+str(students))

    for j in range(2):
        print_obs(sepobs[j], 'sep'+str(j)+'_KT2_'+str(students))
        print_probs(sepprobs[j], 'sep'+str(j)+'_KT2_'+str(students))
        print_states(sepstates[j], 'sep'+str(j)+'_KT2_'+str(students))



















"""

# UNSHARED knowledge time
skillobs = []
skillskills = []
skillprobs = []

for c in range(120):
    skillobs.append([])
    skillskills.append([])
    skillprobs.append([])

    z = [0,0]

    for j in [0,1]:
        a = np.random.random()
        if a < 0.1:
            z[j] = 1
        else:
            z[j] = 0

    #print z

    for i in range(15):
        sk = np.random.randint(0,len(z))

        #print z, sk

        guess = .15
        slip = .15

        st = z[sk]

        a = np.random.random()
        if (st == 1 and a < slip) or (st == 0 and a > guess):
            skillobs[-1].append(0)
        else:
            skillobs[-1].append(1)

        skillskills[-1].append(sk)
        skillprobs[-1].append(sk)

        tp = 0.25

        t = np.random.random()
        if t < tp:
            z[sk] = 1

    #print skillobs[-1]

print_obs(skillobs, 'KT3')
print_skills(skillskills, 'KT3')
print_probs(skillprobs, 'KT3')



#multiple shared knowledge skills
# shared knowledge time
skillobs = []
skillskills = []
skillprobs = []

for c in range(150):
    skillobs.append([])
    skillskills.append([])
    skillprobs.append([])

    z = [0,0,0,0]
    #print z

    skills = [1,2,3,0] * 9
    random.shuffle(skills)

    for sk in skills:
        #sk = np.random.randint(0,len(z))

        #print z, sk

        guess = .05
        slip = .33

        st = z[sk]

        if sk == 0 and z[1] or sk == 1 and z[0]:
            guess = 0.33
            slip = 0.05

        if sk == 2 and z[3] or sk == 3 and z[2]:
            guess = 0.33
            slip = 0.05

        a = np.random.random()
        if (st == 1 and a < slip) or (st == 0 and a > guess):
            skillobs[-1].append(0)
        else:
            skillobs[-1].append(1)

        skillskills[-1].append(sk)
        skillprobs[-1].append(sk)

        tp = 0.1

        t = np.random.random()
        if t < tp:
            z[sk] = 1

    #print skillobs[-1]

print_obs(skillobs, 'KT4')
print_skills(skillskills, 'KT4')
print_probs(skillprobs, 'KT4')














#simulate bkt student
bktobs = []
bktprobs = []
for c in range(40):
    bktobs.append([])
    bktprobs.append([])
    a = np.random.random()
    if a < 0.1:
        z = 1
    else:
        z = 0

    for i in range(10):
        a = np.random.random()
        if (z and a < .2 ) or (z == 0 and a > .2):
            bktobs[-1].append(0)
        else:
            bktobs[-1].append(1)
        bktprobs[-1].append(i)

        t = np.random.random()
        if t < 0.1:
            z = 1


#simulate students w/ continually increasing probability of correctness
contobs = []
contprobs = []

for c in range(1000):
    contobs.append([])
    contprobs.append([])
    z = 0

    for i in range(10):
        a = np.random.random()
        if a < z:
            contobs[-1].append(1)
        else:
            contobs[-1].append(0)
        contprobs[-1].append(i)
        z += 0.1
    #print contobs[-1]

print_obs(contobs, 'cont')
print_probs(contprobs, 'cont')













#do weird skill stuff
#prereq time

skillobs = []
skillskills = []
skillprobs = []

for c in range(120):
    skillobs.append([])
    skillskills.append([])
    skillprobs.append([])

    z = [0,0]

    a = np.random.random()
    if a < 0.1:
        z[0] = 1
    else:
        z[0] = 0

    if z[0]:
        a = np.random.random()
        if a < 0.1:
            z[1] = 1
        else:
            z[1] = 0

    #print z

    for i in range(15):
        sk = np.random.randint(0,2)

        #print z, sk

        guess = .25
        slip = .25

        st = z[sk]


        a = np.random.random()
        if (st == 1 and a < slip) or (st == 0 and a > guess):
            skillobs[-1].append(0)
        else:
            skillobs[-1].append(1)

        skillskills[-1].append(sk)
        skillprobs[-1].append(sk)

        tp = 0.25
        if sk == 0 or (sk == 1 and z[0] == 1):
            t = np.random.random()
            if t < tp:
                z[sk] = 1
    #print skillobs[-1]

print_obs(skillobs, 'KT')
print_skills(skillskills, 'KT')
print_probs(skillprobs, 'KT')

print
print

"""


