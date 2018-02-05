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

from datetime import datetime
startTime = datetime.now()
import networkx as nx
import itertools
import collections
import sys
import numpy as np
import networkx.algorithms.isomorphism as iso
from utils import build_graph, get_neighbors, graph_hash, build_bliss_graph, bliss_subgraph
import sets
import sys
sys.path.append('./PyBliss-0.50beta')
sys.path.append('./PyBliss-0.50beta/lib/python')
import PyBliss
import os

def get_features(dataset,k,n_exs,n_test,b,mode,N_SAMPLES,Neigh_SAMPLE=0, Neigh_PROB=0):
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    log = open("./logs/log-"+dataset+"-features.out","w",0)
    opt = "train"
    x_file = "./"+dataset+"/"+opt+"_x.txt"
    graph_file = "./"+dataset+"/graph_1.txt"
    node_type_file = "./"+dataset+"/node_type.txt"
    #build graph
    G = build_graph(graph_file, node_type_file)
    if dataset == "DBLP":
        to_remove = []
        for n in G.nodes():
            if nx.get_node_attributes(G,'type')[n] == 4:
                to_remove.append(n)
        G.remove_nodes_from(to_remove)
    H = build_bliss_graph(G)
    #load x
    X = np.loadtxt(x_file)
    #gather all patterns in data
    patterns = {}
    patterns[0] = {}
    ex_index = 0
    for x in X:
    	#print(ex_index)
        x = map(int,x)
        neighbors = get_neighbors(G,x, Neigh_SAMPLE, Neigh_PROB)
        if len(neighbors) > N_SAMPLES and N_SAMPLES > 0:
            n_hash = []
            xl = list(x)
            for n in neighbors:
                xl.append(n)
                xl = map(int, xl)
                n_hash.append(graph_hash(bliss_subgraph(H,xl)))
                xl.pop()
            prob = 1/float(len(set(n_hash)))
            counter=collections.Counter(n_hash)
            for h in range(0,len(n_hash)):
                n_hash[h] = prob*(1/float(counter[n_hash[h]]))
            neighbors = np.random.choice(neighbors,N_SAMPLES,replace=False,p=n_hash)
        for n in neighbors:
            
            xl = list(x)
            xl.append(n)
            xl = map(int, xl)
            #print xl
            gh = graph_hash(bliss_subgraph(H,xl))
            conn = 1
            if gh not in patterns[0]:
                patterns[0][gh] = []
                patterns[0][gh].append(ex_index)
            else:
                patterns[0][gh].append(ex_index)

            if gh not in patterns:
                g = nx.Graph(G.subgraph(xl))
                if nx.is_connected(g):
                    #print g.nodes(data=True)
                    #print g.edges()
                    patterns[gh] = {}
                else:
                    conn = 0
            if conn == 1:
                #print >>log, datetime.now() - startTime
                second_neighbors = get_neighbors(G,xl,Neigh_SAMPLE, Neigh_PROB)
                # second_neighbors = G[n]
                # second_neighbors = list((set(neighbors) | set(second_neighbors)) - set(xl))
                for sn in second_neighbors:
                    to_del = list(xl)
                    for d in to_del:
                        input_pattern = list(xl)
                        input_pattern.remove(d)
                        input_pattern.append(int(sn))
                        ghi = graph_hash(bliss_subgraph(H,input_pattern))
                        if ghi not in patterns[gh] and ghi not in patterns:
                            #if nx.is_connected(nx.Graph(G.subgraph(input_pattern))):
                            patterns[gh][ghi] = []
                            patterns[gh][ghi].append(ex_index)
                        elif ghi not in patterns[gh] and ghi in patterns:
                            patterns[gh][ghi] = []
                            patterns[gh][ghi].append(ex_index)
                        else:
                            patterns[gh][ghi].append(ex_index)
                        #print >>log, datetime.now() - startTime
            xl.pop()
        #print datetime.now() - startTime
        ex_index = ex_index + 1
        if ex_index == n_exs:
            break
        print >>log, datetime.now() - startTime
        print >>log, ex_index/float(n_exs)

    opt = "test"
    x_file = "./"+dataset+"/"+opt+"_x.txt"
    graph_file = "./"+dataset+"/graph_2.txt"
    node_type_file = "./"+dataset+"/node_type.txt"
    #build graph
    G = build_graph(graph_file, node_type_file)
    if dataset == "DBLP":
        to_remove = []
        for n in G.nodes():
            if nx.get_node_attributes(G,'type')[n] == 4:
                to_remove.append(n)
        G.remove_nodes_from(to_remove)
    H = build_bliss_graph(G)
    #load x
    X = np.loadtxt(x_file)
    #test_IDS = range(0,len(X[:,]))
    test_IDS = False
    print >>log, "CONSTRUCTING TEST FEATURES"
    ex_index = 0
    for x in X:
        #print(ex_index + n_exs)
        x = x.tolist()
        x = map(int,x)
        neighbors = get_neighbors(G,x,Neigh_SAMPLE, Neigh_PROB)
        if len(neighbors) > N_SAMPLES and N_SAMPLES > 0:
            n_hash = []
            xl = list(x)
            for n in neighbors:
                xl.append(n)
                xl = map(int, xl)
                n_hash.append(graph_hash(bliss_subgraph(H,xl)))
                xl.pop()
            prob = 1/float(len(set(n_hash)))
            counter=collections.Counter(n_hash)
            for h in range(0,len(n_hash)):
                n_hash[h] = prob*(1/float(counter[n_hash[h]]))
            neighbors = np.random.choice(neighbors,N_SAMPLES,replace=False,p=n_hash,)
        for n in neighbors:
            xl = list(x)
            xl.append(n)
            xl = map(int, xl)
            gh = graph_hash(bliss_subgraph(H,xl))
            if gh in patterns[0]:
                
                patterns[0][gh].append(ex_index + n_exs)

            if gh in patterns:
                #print >>log, datetime.now() - startTime
                second_neighbors = get_neighbors(G,xl,Neigh_SAMPLE, Neigh_PROB)
                # second_neighbors = G[n]
                # second_neighbors = list((set(neighbors) | set(second_neighbors)) - set(xl))
                for sn in second_neighbors:
                    to_del = list(xl)
                    for d in to_del:
                        input_pattern = list(xl)
                        #print input_pattern
                        input_pattern.remove(d)
                        input_pattern.append(int(sn))
                        ghi = graph_hash(bliss_subgraph(H,input_pattern))
                        if ghi in patterns[gh]:
                            patterns[gh][ghi].append(ex_index + n_exs)
                        #print >>log, datetime.now() - startTime
            xl.pop()
        print >>log, datetime.now() - startTime
        ex_index = ex_index + 1
        if ex_index == n_test:
            break

    print >>log, datetime.now() - startTime

    num_dim = 0
    print >>log, len(patterns)
    PL = []
    for p in patterns:
        num_dim = num_dim + len(patterns[p])
        PL.append(len(patterns[p]))
    print >>log, PL
    print >>log, num_dim
    new_X = np.zeros((n_exs + n_test, num_dim))
    norm = np.zeros((n_exs + n_test, len(PL)))
    j = 0
    k = 0
    for p in patterns:
        for pp in patterns[p]:
            for i in patterns[p][pp]:
                new_X[i,j] = new_X[i,j] + 1
                norm[i,k] = norm[i,k] + 1
            j = j + 1
        k = k + 1
    s = 0
    f = 0
    pi = 0
    for p in PL:
        f = s + p
        if mode == 'avg':
            new_X[:,s:f] = new_X[:,s:f]/(np.transpose(np.tile(norm[:,pi],(p,1))))
        s = f
        pi = pi + 1
    new_X = np.nan_to_num(new_X)
    np.savetxt(dataset+'train_x.txt',new_X[0:n_exs,:], '%5.0f')
    np.savetxt(dataset+'test_x.txt',new_X[n_exs:(n_exs+n_test),:], '%5.0f')
    np.savetxt(dataset+'PL.txt',np.asarray(PL), '%5.0f')
    np.savetxt(dataset+'test_IDS.txt',np.atleast_1d(test_IDS), '%5.0f')

    return new_X[0:n_exs,:].astype(float), new_X[n_exs:(n_exs+n_test),:].astype(float),PL,test_IDS
