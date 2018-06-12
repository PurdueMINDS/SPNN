type_file = open("node_type_new.txt",'w')
record = set()

with open("graph_1.txt") as f:
    for line in f:
        splits = line.split()
        record.add(int(splits[0]))
        record.add(int(splits[1]))

with open("graph_2.txt") as f:
    for line in f:
        splits = line.split()
        record.add(int(splits[0]))
        record.add(int(splits[1]))


res = dict()

with open("node_type.txt") as f:
    for line in f:
        splits = line.split()
        if(int(splits[0]) in record):
            type_file.write(line)

