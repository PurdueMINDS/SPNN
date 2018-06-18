# SPNN
This is the code for AAAI2018 paper
"Subgraph Pattern Neural Networks for High-Order Graph Evolution Prediction"
Changping Meng, S Chandra Mouli, Bruno Ribeiro, Jennifer Neville

It mainly has two parts, subgraph-based feature generation(input_feature.py) and model training(SPNN.py).
We have a C++ version of feature generation code which is fast but hard to use. 
(We will clean and provide the C++ version if people are interested in it.)
Thanks Leonardo Cotta for the python version of subgraph-based feature generation.

# Installation
Pybliss library is needed.  http://www.tcs.hut.fi/Software/bliss/

cd PyBliss-0.50beta

python setup.py install

# How to run

Example,
 nohup python main.py DBLP 3 8000 4000 0 100 0.2 &
 
You can check logs/log-DBLP-features.out for the process.

1st parameter is the name of the data folder.

2nd parameter is the size of the target subgraph.

3rd parameter is the number of training examples.

4th parameter is the number of testing examples.

Last three parameters are used for accelerating the speed of feature generation.

5th parameter is the maximum number of neighbors. If one node has neighbors more than 5th parameter,

it will sample this number of nodes. If the 5th parameter is 0, this function is disabled.

6th and 7th parameter is used to sample the neighbors. For neighbors larger than 5th parameter, 

you only sample 6th parameter proportion of the neighbors.

# Data
The current dataset is the DBLP dataset.

graph_1.txt, graph_2.txt, graph_3.txt are the graph files.
First number is source node id. Second number is target node id. Third number is the edge type.
Since DBLP is a simple graph, subgraph type can be determined just based on node type. 

node_type.txt is the mapping from node id to node type.
1 is author, 2 is topic, 3 is venue, 4 is paper.

The means of the edges connnecting certain types of nodes.
author--topic means author has published in topic.
author--venue means author has published in venue.
author--author means these two authors have coauthored.
venue--topic means the the venue has this topic.

train_x.txt and test_x.txt are the ids of the sampled subgraph. Thank Carlos Teixeira for kindly giving us early access to his subgraph sampling software


Friendster dataset is available for download under the name "friendster-public.zip" [here](ftp://ftp.cs.purdue.edu/pub/MINDS)

To use Friendster data, just make sure you reference the correct directory, e.g. FriendsterB, and ajust the parameters of your task. More information about the dataset is in its README.md file. Please, refer to that.

If you make use of any of these datasets, please cite the following paper:

@inproceedings{meng2018subgraph,
title={Subgraph Pattern Neural Networks for High-Order Graph Evolution Prediction},
author={Meng, Changping and Mouli, S. Chandra and Ribeiro, Bruno and Neville, Jennifer},
booktitle={AAAI},
year={2018}
}

.
