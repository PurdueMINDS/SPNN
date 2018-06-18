# SPNN
### Authors: Changping Meng, S Chandra Mouli, [Bruno Ribeiro](https://www.cs.purdue.edu/homes/ribeirob/), [Jennifer Neville](https://www.cs.purdue.edu/homes/neville/) and [Leonardo Cotta](https://cottascience.github.io/)
## Overview:
This is the code for AAAI2018 paper
"Subgraph Pattern Neural Networks for High-Order Graph Evolution Prediction"
Changping Meng, S Chandra Mouli, Bruno Ribeiro, Jennifer Neville

It mainly has two parts, subgraph-based feature generation(input_feature.py) and model training(SPNN.py).
We have a C++ version of feature generation code which is fast but hard to use.
(We will clean and provide the C++ version if people are interested in it.)
Thanks Leonardo Cotta for the python version of subgraph-based feature generation and model.

# Installation

You will need PyTorch installed. The code was tested with PyTorch 0.27, which can be downloaded [here](https://pytorch.org/)

[PyBliss](http://www.tcs.hut.fi/Software/bliss/) library is also needed. A version ready to use in MacOS and Linux systems is already included in this project, you just need to compile it by doing the following from the project's root directory:

```console
cd PyBliss-0.50beta

python setup.py install
```

# How to run

For instance, to run the DBLP task described in the SPNN paper, you should run the following command from the project's root directory

```console
 nohup python main.py DBLP 3 8000 4000 0 100 0.2 &
```

You can check the logs folder for the process outputs, files will be indexed with the dataset folder you are using.

1st parameter is the name of the dataset folder.

2nd parameter is the size of the target subgraph.

3rd parameter is the number of training examples.

4th parameter is the number of testing examples.

Last three parameters are used for accelerating the speed of feature generation.

5th parameter is the maximum number of neighbors. If one node has neighbors more than 5th parameter,

it will sample this number of nodes. If the 5th parameter is 0, this function is disabled.

6th and 7th parameter is used to sample the neighbors. For neighbors larger than 5th parameter,

you only sample 6th parameter proportion of the neighbors.

# Data

## DBLP Dataset
This project is set to run with the DBLP dataset, included in the project's folder. This is a pre-processed dataset, the original one was used in Sun et. al<sup>[1](#myfootnote1)</sup>. If you want to use another dataset with the code, make sure it follows the same patterns, including files names that this dataset has. The data is described in what follows.

graph_1.txt, graph_2.txt, graph_3.txt are the graph files.
First number is source node id. Second number is target node id. Third number is the edge type.
Since DBLP is a simple graph, subgraph type can be determined just based on node type.

node_type.txt is the mapping from node id to node type.
1 is author, 2 is topic, 3 is venue, 4 is paper.

The means of the edges connecting certain types of nodes.
author--topic means author has published in topic.
author--venue means author has published in venue.
author--author means these two authors have coauthored.
venue--topic means the the venue has this topic.

train_x.txt and test_x.txt are the ids of the sampled subgraph. Thank Carlos Teixeira for kindly giving us early access to his subgraph sampling software. Wang et. al<sup>[2](#myfootnote2)</sup> can also be used to sample the subgraphs.

## Friendster Dataset
Friendster dataset is available for download under the name "friendster-public.zip" [here](https://goo.gl/8C7BU9)

Instructions on how to use it are below.

FriendsterComplete is the complete dataset of Friendster. The file nodes.csv contains in each line the node id and its genre, age and relationship status (node_id,node_genre,node_age,node_relationship_status). The file edges.csv contains two nodes ids and one timestamp (node1_id,node2_id,timestamp). This fact means that these two nodes exchanged messages at that timestamp (node1 posted on node2's wall). Usually, this can be used as temporal edges in a graph. Also, please don’t assume anything about node ids in this dataset, treat ids as strings, not numbers.

FriendsterB (Figure 2(b)) and FriendsterC (Figure 2(c)) directories correspond to pre-processed datasets used in Meng et. al., 2018. The files in these two directories are described below:

graph_1.txt, graph_2.txt, graph_3.txt are the graph files. First number is source node id. Second number is target node id. Third number is the edge type.

node_type.txt is the mapping from node id to node type. In these datasets, every node has the same type.

The meaning of an edge here is whether two friends shared messages at the timestep of the graph.

train_x.txt and test_x.txt are the ids of the sampled subgraphs. The method is described in SPNN paper.

train_y.txt and test_y.txt are the labels of the subgraphs, (referenced by line), according to the tasks b) FriendsterB and c) FriensterC in the SPNN paper. The way to use this dataset with the SPNN code is the same as described for DBLP previously. These folders are already ready to use.

If you make use of any of these datasets, please cite the following paper:

```console
@inproceedings{meng2018subgraph,
title={Subgraph Pattern Neural Networks for High-Order Graph Evolution Prediction},
author={Meng, Changping and Mouli, S. Chandra and Ribeiro, Bruno and Neville, Jennifer},
booktitle={AAAI},
year={2018}
}
```

<a name="myfootnote1">1</a>: Sun , Yizhou, et al. "Co-author relationship prediction in heterogeneous bibliographic networks." Advances in Social Networks Analysis and Mining (ASONAM), 2011 International Conference on. IEEE, 2011.

<a name="myfootnote2">2</a>: Wang , Pinghui, et al. "Efficiently estimating motif statistics of large networks." ACM Transactions on Knowledge Discovery from Data (TKDD) 9.2 (2014): 8.
