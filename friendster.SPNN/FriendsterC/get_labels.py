import networkx as nx
import itertools
import sys
import networkx.algorithms.isomorphism as iso

exs_file = "train_x.txt"
graph_file = "graph_2.txt"
node_type_file = "node_type.txt"
k = 4
#build graph
print "hey"
with open(graph_file) as f:
    graph_lines = f.readlines()
graph_lines = [x.strip() for x in graph_lines]
G=nx.Graph()
for x in graph_lines:
    x = x.split()
    edge = (int(x[0]),int(x[1]))
    G.add_edge(*edge)
with open(node_type_file) as f:
    node_types = f.readlines()
node_types = [x.strip() for x in node_types]
node_type_table = {}
for x in node_types:
    x = x.split()
    node_type_table[str(int(x[0]))] = int(x[1])
print "hey"

G=nx.Graph()
for x in graph_lines:
    x = x.split()
    edge = (int(x[0]),int(x[1]))
    G.add_edge(*edge)

for x in G.nodes():
    G.add_node(x, type=node_type_table[str(x)])
G.nodes(data=True)

with open(exs_file) as f:
    exs = f.readlines()
exs = [x.strip() for x in exs]
for ex in exs:
    ex = ex.split()
    ex = map(int, ex)
    #del ex[-2]
    sub = nx.Graph(G.subgraph(ex))
    if len(sub.edges()) < 1:
        print "1"
    else:
        print "0"
