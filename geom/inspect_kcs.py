


f = open('ds76_tx_All_Data_74_2015_0928_045712.txt', 'r')
data = []
heads = f.readline().strip().split('\t')

for line in f.readlines():
    vals = line.strip().split('\t')
    data.append(vals)
f.close()

cols = []
for c in range(len(data[0])):
    cols.append([])

for row in data:
    for i in range(len(row)):
        cols[i].append(row[i])

for c in range(len(cols)):
    cols[c] = list(set(cols[c]))
    cols[c].append(heads[c])
    if len(cols[c]) < 100 and len(cols[c]) > 2:
        print cols[c]
    print

print data
#print cols