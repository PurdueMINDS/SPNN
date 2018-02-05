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


import sys
from input_features import get_features
from SPNN import run_SPNN
import numpy as np
import timeit
data_dir = sys.argv[1]
k = int(sys.argv[2])
n_train = int(sys.argv[3])
n_test = int(sys.argv[4])
bias = 0
mode = sum
N_SAMPLES = int(sys.argv[5])
Neighbor_SAMPLES = int(sys.argv[6])
Neighbor_prob = float(sys.argv[7])
start_time = timeit.default_timer()

train_x, test_x, unit_size, test_IDS = get_features(data_dir,k,n_train,n_test,bias,mode,N_SAMPLES,Neighbor_SAMPLES,Neighbor_prob)
end_time = timeit.default_timer()
train_y = np.loadtxt("./"+data_dir+"/train_y.txt")[0:n_train].astype('int32')
test_y = np.loadtxt("./"+data_dir+"/test_y.txt")[0:n_test].astype('int32')
lrate = 0.005
l1_val = 0.001
l2_val = 0.001
test_auc = run_SPNN(train_x,train_y,test_x,test_y,mynode=unit_size,learning_rate=lrate, L1_reg=l1_val, L2_reg=l2_val, n_epochs=80000, test_size = n_test)
print(('AUC score on testing data is %f')%(test_auc))
