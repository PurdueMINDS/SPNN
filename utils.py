# Copyright 2017 Changping Meng, Leonardo Cotta, S Chandra Mouli, Bruno Ribeiro, Jennifer Neville
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx
import itertools
import sys
import networkx.algorithms.isomorphism as iso
import sys
import random
import pdb
sys.path.append('./PyBliss-0.50beta')
sys.path.append('./PyBliss-0.50beta/lib/python')
import PyBliss

def build_bliss_graph(G):
    H = PyBliss.Graph()
    pdb.set_trace()
    for v in G.nodes():
        H.add_vertex(str(v), int(nx.get_node_attributes(G,'type')[v]))
    for e in G.edges():
        H.add_edge(str(e[0]),str(e[1]))
    return H

def bliss_subgraph(H,vertices):
    S = PyBliss.Graph()
    for v in vertices:
        S.add_vertex(str(int(v)),H._vertices[str(int(v))].color)
    set_v = set(S.get_vertices())
    for v in vertices:
        v_edges = H._vertices[str(int(v))].edges
        for u in v_edges:
            if u.name in set_v:
                S.add_edge(str(int(v)), u.name)
    return S

def graph_hash(H):
    # H = PyBliss.Graph()
    # for v in G.nodes():
    #     H.add_vertex(str(v), int(nx.get_node_attributes(G,'type')[v]))
    # for e in G.edges():
    #     H.add_edge(str(e[0]),str(e[1]))
    canlab = H.canonical_labeling()
    return str(hash(H.relabel(canlab)))

def build_graph(graph_file, node_type_file):
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
    G=nx.Graph()
    for x in graph_lines:
        x = x.split()
        edge = (int(x[0]),int(x[1]))
        G.add_edge(*edge)
    for x in G.nodes():
        G.add_node(x, type=node_type_table[str(x)])
    G.nodes(data=True)
    return G

def get_neighbors(G,ex, GN, prob):
    all_neighs = set()
    e = set(ex)
    for n in ex:
        for neigh in G[n]:
            if (GN > 0 and ((len(G[n]) < 100) or (len(G[n]) > GN and random.random()< prob ))) or GN==0:
                all_neighs.add(neigh)
    all_neighs = all_neighs - e
    return list(all_neighs)

def sample_neighbors(N):
    return N
