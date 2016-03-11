import numpy as np



def get_length(row):
    length = 0
    for i in range(len(row)):
        if row[i] > -1:
            length+= 1
    return length

def make_traj(x, t):
    xs = []

    for c in range(x.shape[0]):
        newx = []

        pre = 0.0
        prenum = 0

        post = 0.0
        postnum = 0

        length = get_length(x[c,:])

        print t[c,0:length]

        for i in range(length):

            if t[c,i] > 0 and i < 2:
                prenum += 1
                pre += x[c,i]

            if t[c,i] < 1:
                newx.append(x[c,i])

            if t[c,i] > 0 and i >= length - 2:
                post += x[c,i]
                postnum += 1

        print
        print prenum
        print postnum

        if postnum == 0:
            continue
        if prenum == 0:
            continue

        pre = pre / prenum
        post = post / postnum
        gg = [pre]
        gg.extend(newx)
        gg.append(post)
        xs.append(gg)

    return xs


f = open("matches.tsv", "w")
f.write("skill\tposttest variance\ttotal matches\tsquared error\tmean error\ttotal ignoring zero-tutor-problem students\tsquared error\tmean_error\n")

for skill in ['center','shape','spread', 'x_axis', 'y_axis', 'histogram', 'd_to_h', 'h_to_d']:

    X = np.loadtxt(open("observations_" + skill + ".csv","rb"),delimiter=",")
    #load problem IDs for these observations
    T = np.loadtxt(open("assessment_" + skill + ".csv","rb"),delimiter=",")

    xs = make_traj(X,T)

    print xs

    print skill


    errs_all = []
    errs_g2 = []

    for c in range(len(xs)):

        for i in range(c,len(xs)):

            if len(xs[i]) > len(xs[c]):
                #t' longer than t. Check for a match
                match = True
                for j in range(len(xs[c]) - 1):
                    if xs[c][j] != xs[i][j]:
                        match = False

                if match:
                    index = len(xs[c]) - 1
                    errs_all.append(xs[i][index] - xs[c][index])
                    if index > 1:
                        errs_g2.append(xs[i][index] - xs[c][index])

    skill_squared_error_all = np.mean(np.array(errs_all)**2)
    skill_squared_error_g2 = np.mean(np.array(errs_g2)**2)

    mean_error_all = np.mean(errs_all)
    mean_error_g2 = np.mean(errs_g2)

    post_var = np.var(np.array( [x[-1] for x in xs] ))

    """
    print num_match_all
    print skill_squared_error_all
    print num_match_g2
    print skill_squared_error_g2
    """


    f.write(skill + "\t" + str(post_var) + "\t" + str(len(errs_all)) + "\t" + str(skill_squared_error_all) + "\t"
            + str(mean_error_all) + "\t" + str(len(errs_g2)) + "\t" + str(skill_squared_error_g2)
            + "\t" + str(mean_error_g2) + "\n")


