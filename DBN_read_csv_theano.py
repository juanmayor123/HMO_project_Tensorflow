from __future__ import print_function, division
import os
import numpy as np
import csv
import gzip, cPickle
from numpy import genfromtxt

import theano
import theano.tensor as T
import timeit

import DBN
from theano.sandbox.rng_mrg import MRG_RandomStreams
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM

batch_size=510
k=100
pretraining_epochs=10
pretrain_lr=0.1
finetune_lr=0.1
training_epochs=5000
#theano.config.exception_verbosity='high'
#theano.config.traceback.limit=5000
#theano.config.optimizer='fast_compile'
    
    #Open csv file and read in data
csvFile1 = "train_sub_HMO4.csv"
csvFile2 = "test_sub_HMO4.csv"

training_data_file = genfromtxt(csvFile1, delimiter=',', skip_header=0)
test_data_file = genfromtxt(csvFile2, delimiter=',', skip_header=0)
data_shape = "There are " + repr(training_data_file.shape[0]) + " samples of vector length " + repr(training_data_file.shape[1])
print(data_shape)
num_rows = training_data_file.shape[0] # Number of data samples
num_cols = training_data_file.shape[1] # Length of Data Vector

num_rowst = test_data_file.shape[0] # Number of data samples
num_colst = test_data_file.shape[1] # Length of Data Vector


total_size_train=(num_cols-1)*num_rows
total_size_test=(num_colst-1)*num_rowst
    
data_train = np.arange(total_size_train)
data_train = data_train.reshape(num_rows, num_cols-1) # 2D Matrix of data points
data_train = data_train.astype('float64')
    
label_train = np.arange(num_rows)
label_train = label_train.astype('int32')

data_test = np.arange(total_size_test)
data_test = data_test.reshape(num_rowst, num_cols-1) # 2D Matrix of data points
data_test = data_test.astype('float64')

label_test = np.arange(num_rows)
label_test = label_test.astype('int32')


#Read through data_train file, assume label is in last col
for i in range(training_data_file.shape[0]):
    label_train[i] = training_data_file[i][num_cols-1]
    for j in range(num_cols-1):
        data_train[i][j] = training_data_file[i][j]
#print(data_train[i][j])

#Read through data_test file, assume label is in last col
for i in range(test_data_file.shape[0]):
    label_test[i] = test_data_file[i][num_colst-1]
    for j in range(num_colst-1):
        data_test[i][j] = test_data_file[i][j]
#print(data_test[i][j])

data_train=np.squeeze(np.asarray(data_train))
data_test=np.squeeze(np.asarray(data_test))

label_train=np.squeeze(np.asarray(label_train))-1
label_test=np.squeeze(np.asarray(label_test))-1

#data_train=data_train.flatten()
print(data_train.shape)

n_train_batches = data_train.shape[0] // batch_size
print(n_train_batches)

data_train = theano.shared(data_train)
label_train = theano.shared(label_train)
data_test = theano.shared(data_test)
label_test = theano.shared(label_test)
numpy_rng = np.random.RandomState(0)

dbn = DBN.DBN(numpy_rng=numpy_rng, n_ins=20,
          hidden_layers_sizes=[10],
          n_outs=4)

print('... getting the pretraining functions')
pretraining_fns = dbn.pretraining_functions(train_set_x=data_train,batch_size=batch_size,k=k)

print('... pre-training the model')
start_time = timeit.default_timer()
    # Pre-train layer-wise
for i in range(dbn.n_layers):
        # go through pretraining epochs
    for epoch in range(pretraining_epochs):
            # go through the training set
        c = []
        for batch_index in range(n_train_batches):
            c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
        print('Pre-training layer %i, epoch %d, cost ', i, epoch)
        print(np.mean(c))
end_time = timeit.default_timer()

print('... getting the finetuning functions')
train_fn, test_model, true0, true1, true2, true3, falspos0, falspos1, falspos2, falspos3, falsneg0, falsneg1, falsneg2, falsneg3 = dbn.build_finetune_functions(train_set_x=data_train, train_set_y=label_train, test_set_x=data_test,test_set_y=label_test,batch_size=batch_size,learning_rate=finetune_lr)

print(train_fn,test_model)

epoch = 0
    
while (epoch < training_epochs):
        epoch = epoch + 1
        temp=0
        temp_pr0=0
        temp_pr1=0
        temp_pr2=0
        temp_pr3=0
        temp_re0=0
        temp_re1=0
        temp_re2=0
        temp_re3=0
        for minibatch_index in range(n_train_batches):
            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            test_losses = test_model()
            test_score = np.mean(test_losses)
            true_0=np.sum(true0())
            true_1=np.sum(true1())
            true_2=np.sum(true2())
            true_3=np.sum(true3())
            fals_0=np.sum(falspos0())
            fals_1=np.sum(falspos1())
            fals_2=np.sum(falspos2())
            fals_3=np.sum(falspos3())
            falsn_0=np.sum(falsneg0())
            falsn_1=np.sum(falsneg1())
            falsn_2=np.sum(falsneg2())
            falsn_3=np.sum(falsneg3())
            temp=temp+test_score
            temp_pr0+=true_0/(true_0+fals_0)
            temp_pr1+=true_1/(true_1+fals_1)
            temp_pr2+=true_2/(true_2+fals_2)
            temp_pr3+=true_3/(true_3+fals_3)
            temp_re0+=true_0/(true_0+falsn_0)
            temp_re1+=true_1/(true_1+falsn_1)
            temp_re2+=true_2/(true_2+falsn_2)
            temp_re3+=true_3/(true_3+falsn_3)
            #print(true1())
            #print(true_0/(true_0+fals_0),true_1/(true_1+fals_1),true_2/(true_2+fals_2),true_3/(true_3+fals_3),true_0/(true_0+falsn_0),true_1/(true_1+falsn_1),true_2/(true_2+falsn_2),true_3/(true_3+falsn_3))
        print(('     epoch %i,  test error of '
                    ' %f %%' 'precision0: ' ' %f ' 'precision1: ' ' %f ' 'precision2: ' ' %f ' 'precision3: ' ' %f ' 'recall0: ' ' %f ' 'recall1: ' ' %f ' 'recall2: ' ' %f ' 'recall3: ' ' %f ' ) %
                    (epoch,
                    (temp/n_train_batches) * 100.,(temp_pr0/n_train_batches),(temp_pr1/n_train_batches),(temp_pr2/n_train_batches),(temp_pr3/n_train_batches),(temp_re0/n_train_batches),(temp_re1/n_train_batches),(temp_re2/n_train_batches),(temp_re3/n_train_batches)))
precision_f=((temp_pr0/n_train_batches)+(temp_pr1/n_train_batches)+(temp_pr2/n_train_batches)+(temp_pr3/n_train_batches))/4
recall_f=((temp_re0/n_train_batches)+(temp_re1/n_train_batches)+(temp_re2/n_train_batches)+(temp_re3/n_train_batches))/4
f1=2*precision_f*recall_f/(precision_f+recall_f)
print('Final results:' 'Precision: ' ' %f ' 'Recall: ' ' %f ' 'F1: ' ' %f ',(precision_f,recall_f,f1))


