#simulate different student situations
import numpy as np

def print_obs(obs, skill):
    f = open('dump/observations_simulated_' + skill + '.csv', 'w')
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

for c in range(40):
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



#do modified transition stuff
transobs = []
transprobs = []
for c in range(40):
    transobs.append([])
    transprobs.append([])
    a = np.random.random()
    if a < 0.1:
        z = 1
    else:
        z = 0

    for i in range(10):
        a = np.random.random()
        if (z and a < .2 ) or (z == 0 and a > .2):
            transobs[-1].append(0)
        else:
            transobs[-1].append(1)

        prob = np.random.randint(0, 15)

        transprobs[-1].append(prob)

        tp = 0.1
        if prob < 5:
            tp += 0.05
        if prob >= 10:
            tp -= 0.05

        t = np.random.random()
        if t < tp:
            z = 1

print_obs(transobs, 'trans')
print_probs(transprobs, 'trans')



#do weird skill stuff
skillobs = []
skillskills = []
skillprobs = []

for c in range(60):
    skillobs.append([])
    skillskills.append([])
    skillprobs.append([])

    z = [0,0,0,0]

    for j in [0,2,3]:
        a = np.random.random()
        if a < 0.1:
            z[j] = 1
        else:
            z[j] = 0
    if z[0]:
        a = np.random.random()
        if a < 0.1:
            z[1] = 1
        else:
            z[1] = 0

    print z

    for i in range(30):
        sk = np.random.randint(0,4)

        if sk == 1 and z[0] == 0 and np.random.random() < .5:
            sk = np.random.randint(0, 4)

        while z[sk] and np.random.random() < .5:
            sk = np.random.randint(0, 4)

        print z, sk

        guess = .2
        slip = .2

        st = z[sk]

        if sk == 2 and z[3]:
            guess = 0.35
            slip = 0.1
        if sk == 3 and z[2]:
            guess = 0.35
            slip = 0.1

        a = np.random.random()
        if (z[sk] == 1 and a < slip ) or (z[sk] == 0 and a > guess):
            skillobs[-1].append(0)
        else:
            skillobs[-1].append(1)

        skillskills[-1].append(sk)
        skillprobs[-1].append(sk)

        tp = 0.25
        if sk == 0 or sk == 2 or sk == 3 or (sk == 1 and z[0] == 1):
            t = np.random.random()
            if t < tp:
                z[sk] = 1
    print skillobs[-1]

print_obs(skillobs, 'KT')
print_skills(skillskills, 'KT')
print_probs(skillprobs, 'KT')










