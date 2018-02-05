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
#
# Thanks the Theano tutorial on DeepLearn.net http://deeplearning.net/tutorial/mlp.html.


__docformat__ = 'restructedtext en'

import os
import sys
import timeit
import numpy
import numpy as np
import theano
import theano.tensor as T
import pdb
import random
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
theano.config.exception_verbosity='high'


# The logistic Regression model
class LogisticRegression(object):
 
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        
        self.W_shade = theano.shared(
                               value=numpy.ones(
                                (n_in, n_out),
                                dtype=theano.config.floatX
                                                 ),
                               name='W_shade',
                               borrow=True
                               )
        
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
        self.b_shade = theano.shared(
                               value=numpy.ones(
                                                 (n_out,),
                                                 dtype=theano.config.floatX
                                                 ),
                               name='b_shade',
                               borrow=True
                               )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y_pred_val = self.p_y_given_x[:,1]
        self.params = [self.W, self.b]
        self.shade = [self.W_shade, self.b_shade]
        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y, weights):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]*weights)
 


    def auc(self, y):
 
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y and y_pred has different dimensions'
            )
        if y.dtype.startswith('int'):
            y_pred = sekf,
            return theano.shared(roc_auc_score(y_true,self.y_pred))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, node_size, W=None, b=None,
                 activation=T.tanh, n_type=20):
        self.input = input

        if W is None:
            W_values = numpy.zeros((n_in, n_out))
            W_shade_val = numpy.zeros((n_in, n_out))
            hide = numpy.ones((n_out,), dtype=theano.config.floatX)
            n_feature = n_in / n_type
            total = 0
            for i in range(0,n_type):
                for j in range(0,node_size[i]):
                    W_values[total+j,i] = rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                                                                high=numpy.sqrt(6. / (n_in + n_out)),
                                                                size=(1, 1)
                                                                )
                    W_shade_val[total+j,i] = 1
                total = total+node_size[i]
            
            
            
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
            W_shade = theano.shared(value=W_shade_val, name='W_shade', borrow=True)
    

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            b_shade_val = numpy.ones((n_out,), dtype=theano.config.floatX)
            b_shade = theano.shared(value=b_shade_val, name='b_shade', borrow=True)
        
        self.W_shade = W_shade
        self.b_shade = b_shade
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        
        
        count = 0
        for i in range(0,n_type):
            sum_val = T.sum(input[:,count:count+node_size[i]])
            tmp_val = T.switch(T.le(sum_val, T.constant(0.000001)), T.constant(0), lin_output[i])
            T.set_subtensor(lin_output[i],tmp_val)
            count = count + node_size[i]

        
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        self.shade = [self.W_shade, self.b_shade]



# SPNN model
class SPNN(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out, in_node_size, n_type ):

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            node_size = in_node_size,
            #activation=theano.tensor.nnet.relu,
            activation=T.tanh,
            n_type = n_type
        )
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.y_pred = self.logRegressionLayer.y_pred
        self.y_pred_val = self.logRegressionLayer.y_pred_val
        
        self.auc = self.logRegressionLayer.auc

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.shade = self.hiddenLayer.shade + self.logRegressionLayer.shade

        self.input = input



# run the SPNN model
def run_SPNN(train_set_x, train_set_y,test_set_x, test_set_y, mynode, learning_rate=0.005, L1_reg=0.001, L2_reg=0.001, n_epochs=80000,batch_size=1000, test_size = 4000):

    n_hidden = len(mynode)
    trainx_size = train_set_x.shape[1]
    trainx_num = train_set_x.shape[0]
    tmpD = numpy.std(train_set_x, 0)
    tmpD[numpy.where(tmpD == 0)] = 1
    tmpM = numpy.mean(train_set_x,axis = 0)
    
    tmpMax = numpy.max(train_set_x,axis = 0)
    tmpMin = numpy.min(train_set_x,axis = 0)

    # Normalize the data.
    for p in range(0, train_set_x.shape[1]):
        train_set_x[:,p] = (train_set_x[:,p] - tmpM[p] )*1.0/(tmpD[p])

    for p in range(0, test_set_x.shape[1]):
        test_set_x[:,p] = (test_set_x[:,p] - tmpM[p])*1.0/(tmpD[p])

    # Use 25% of traing data for validation.
    valid_ratio = 0.75
    valid_set_x = train_set_x[int(trainx_num * valid_ratio):trainx_num,:]
    valid_set_y = train_set_y[int(trainx_num * valid_ratio):trainx_num]
    train_set_x = train_set_x[0:int(trainx_num * valid_ratio),:]
    train_set_y = train_set_y[0:int(trainx_num * valid_ratio)]

    train_prob = np.sum(train_set_y)*1.0/train_set_y.shape[0]
    train_weights = np.zeros(train_set_y.shape[0])
    train_weights[np.where(train_set_y==1)] = train_prob
    train_weights[np.where(train_set_y==0)] = 1-train_prob
    train_weights = theano.shared(train_weights)
    valid_prob = np.sum(valid_set_y)*1.0/valid_set_y.shape[0]
    valid_weights = np.zeros(valid_set_y.shape[0])
    valid_weights[np.where(valid_set_y==1)] = valid_prob
    valid_weights[np.where(valid_set_y==0)] = 1-valid_prob

    test_prob = np.sum(test_set_y)*1.0/test_set_y.shape[0]
    test_weights = np.zeros(test_set_y.shape[0])
    test_weights[np.where(test_set_y==1)] = test_prob
    test_weights[np.where(test_set_y==0)] = 1-test_prob



    train_set_x = theano.shared(train_set_x)
    train_set_y = theano.shared(train_set_y)
    valid_set_x = theano.shared(valid_set_x)
    valid_set_y = theano.shared(valid_set_y)

    trainx_num = int(trainx_num * valid_ratio)
    train_batch_size = trainx_num

    testx_num = test_set_x.shape[0]
    test_size = test_set_x.shape[0];

    test_set_x = theano.shared(test_set_x)
    test_set_y = theano.shared(test_set_y)

    test_batch_size = testx_num

    n_train_batches = 1
    n_valid_batches = 1  #(testx_num//2) // batch_size
    n_test_batches =  1 #(testx_num - testx_num//2) // batch_size

    index = T.lscalar()  
    x = T.matrix('x')  
    y = T.ivector('y')  
    rng = numpy.random.RandomState(1234)
    classifier = SPNN(
        rng=rng,
        input=x,
        n_in=trainx_size,
        n_hidden=n_hidden,
        n_out=2,
        in_node_size = mynode,
        n_type = n_hidden
    )

    cost = (
        classifier.negative_log_likelihood(y,train_weights)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
      
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.y_pred_val,
        givens={
            x: test_set_x[index * test_batch_size:(index + 1) * test_batch_size]
           
        },
        on_unused_input='ignore'
    )
      
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.y_pred_val,
        givens={
            x: valid_set_x[index * train_batch_size:(index + 1) * train_batch_size]
        },
        on_unused_input='ignore'
    )

    gparams = [T.grad(cost, param, disconnected_inputs='ignore') for param in classifier.params]
     
    updates = [
        (param, numpy.multiply((param - learning_rate * gparam),shade))
        for param, shade ,gparam in zip(classifier.params,classifier.shade, gparams)
    ]
      
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * train_batch_size: (index + 1) * train_batch_size],
            y: train_set_y[index * train_batch_size: (index + 1) * train_batch_size]
        }
        
    )
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 1.1  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)

    validation_frequency = 100
    best_validation_auc = -numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
      
    while (epoch < n_epochs) and (not done_looping):
          
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
              
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                valid_res = validate_model(0)
                for i in range(1,n_valid_batches):
                    valid_res = numpy.append(valid_res, validate_model(i))
                    
                this_validation_auc = roc_auc_score(valid_set_y.eval()[0:len(valid_res)],valid_res, sample_weight = valid_weights)
                
                if this_validation_auc > best_validation_auc:
                    if (
                        this_validation_auc > best_validation_auc *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_auc = this_validation_auc
                    best_iter = iter
                    
                    test_res = test_model(0)
                    for i in range(1,n_test_batches):
                        test_res = numpy.append(test_res, test_model(i))

                    test_score = roc_auc_score(test_set_y.eval()[0:len(test_res)],test_res,sample_weight = test_weights)
                    '''
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%,  Valid Score %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100., this_validation_auc*100.))
                    '''
    end_time = timeit.default_timer()
    return test_score
