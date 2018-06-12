type_file = open("node_type.txt",'w')

res = dict()

with open("node_type_back.txt") as f:
    for line in f:
        splits = line.split()
        res[int(splits[0])] = int(splits[1])

for key, value in res.iteritems():
    type_file.write(str(key) + ' '+ str(value) + '\n')
