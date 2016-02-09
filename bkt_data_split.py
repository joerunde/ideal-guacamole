##split up a .tsv file into trajectories
import sys, json
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
    
    if user not in userdata:
        userdata[user] = {}
        userdata[user]['all'] = []
    if skill not in userdata[user]:
        userdata[user][skill] = []
    
    if len(problems) > 0 and problems[-1] == problem:
        continue
    problems.append(problem)
    tup = (correct, problem)
    userdata[user][skill].append(tup)
    userdata[user]['all'].append(tup)

problems = list(set(problems))

pf = open("problems_idx_" + sys.argv[1], "w")
pf.write(json.dumps(problems))
pf.close()

#print userdata
#print problems

f.close()

f = open("observations_" + sys.argv[1], "w")
g = open("problems_" + sys.argv[1], "w")
h = open("assessment_" + sys.argv[1], "w")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#skills all in one traj now....

for user, data in userdata.iteritems():
    #for skill, seq in data.iteritems():
    #print seq
    seq = data['all']
    fline = ''
    gline = ''
    hline = ''
    for tup in seq:
        #print str(tup[0]) + '\t' + str(problems.index(tup[1]))
        fline += str(tup[0]) + ','
        gline += str(problems.index(tup[1])) + ','
        if 'assess' in tup[1]:
            hline += ('1,')
        else:
            hline += ('0,')
    for i in range(70 - len(seq)):
        fline += "-1,"
        gline += "-1,"
        hline += "-1"
    f.write(fline[0:-1] + '\n')
    g.write(gline[0:-1] + '\n')
    h.write(hline[0:-1] + '\n')

