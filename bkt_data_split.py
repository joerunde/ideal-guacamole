##split up a .tsv file into trajectories
import sys, json, os
print("Usage: filename.csv [skills]")

f = open("bktdata.txt","r")

userdata = {}
problems = []

for line in f.readlines():
    vals = line.strip().split('\t')
    correct = 2 - int(vals[0])
    user = vals[1]
    problem = vals[2]
    problem = problem.replace("Post","Pre")
    skill = vals[3]

    if skill not in sys.argv:
        continue

    skillid = sys.argv.index(skill) - 2
    
    if user not in userdata:
        userdata[user] = {}
        userdata[user]['all'] = []
    if skill not in userdata[user]:
        userdata[user][skill] = []
    
    if len(problems) > 0 and problems[-1] == problem and 'assess' not in problem:
        continue
    problems.append(problem)
    tup = (correct, problem, skillid)
    userdata[user][skill].append(tup)
    userdata[user]['all'].append(tup)

problems = list(set(problems))

pf = open("dump/problems_idx_" + sys.argv[1], "w")
pf.write(json.dumps(problems))
pf.close()

#print userdata
#print problems

f.close()

f = open("dump/observations_" + sys.argv[1], "w")
g = open("dump/problems_" + sys.argv[1], "w")
h = open("dump/assessment_" + sys.argv[1], "w")
s = open("dump/skills_" + sys.argv[1], "w")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#skills all in one traj now....

for user, data in userdata.iteritems():
    #for skill, seq in data.iteritems():
    #print seq
    seq = data['all']
    fline = ''
    gline = ''
    hline = ''
    sline = ''
    if len(seq) > 0:
        for tup in seq:
            #print str(tup[0]) + '\t' + str(problems.index(tup[1]))
            sline += str(tup[2]) + ','
            fline += str(tup[0]) + ','
            gline += str(problems.index(tup[1])) + ','
            if 'assess' in tup[1]:
                hline += ('1,')
            else:
                hline += ('0,')
        for i in range(80 - len(seq)):
            fline += "-1,"
            gline += "-1,"
            hline += "-1,"
            sline += "-1,"
        f.write(fline[0:-1] + '\n')
        g.write(gline[0:-1] + '\n')
        h.write(hline[0:-1] + '\n')
        s.write(sline[0:-1] + '\n')
    else:
        print "skipped user: " + user

dirname = "plots_" + sys.argv[1][:-4]
os.makedirs(dirname)
os.makedirs(dirname + '/D')
os.makedirs(dirname + '/T')
os.makedirs(dirname + '/L')
os.makedirs(dirname + '/G')

