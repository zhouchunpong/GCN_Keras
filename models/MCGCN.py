# A reference code for MCGCN (Multi-Channel Graph Convolutional Network)
from __future__ import print_function
from keras.layers import Input, Dropout, Embedding, Dense, concatenate, dot, multiply, Lambda, add, Reshape, Softmax
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
import numpy as np
from graph import GraphConvolution
from graph_attention_layer2 import GraphAtt
from GAT import gat
import scipy.sparse as sp
from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.sparse import linalg

# Define parameters
NB_EPOCH = 300
PATIENCE = 30  # early stopping patience

# Get data
# X:features  A:graph  y:labels
A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_original_data('cora')  # {'cora', 'citeseer', 'pubmed'}
X = preprocess_features(X)

A1 = preprocess_adj(A, power=1)
A2 = preprocess_adj(A, power=2)
A3 = preprocess_adj(A, power=3)
alpha = 0.7
A = alpha*A1+ alpha**2*A2+ alpha**3*A3

res=[]
att=[]

X_in=Input(shape=(X.shape[1],))  # the number of feature
G=[Input(shape=(None, None), batch_shape=(None, None), sparse=False)]

drop=0.6
H=Dropout(rate=drop)(X_in)
unit1= 24
unit2=7
regulation= 5e-4
activation='elu'


H=GraphConvolution(unit1,  activation=activation,
                    kernel_regularizer=l2(regulation))([H]+G)  # shared weight 
H=Dropout(rate=drop)(H)
Y=Dense(y_train.shape[1], activation='softmax',
        kernel_regularizer=l2(regulation))(H)  # full-connected layer with softmax

# Compile model
model=Model(inputs=[X_in]+G, outputs=Y)

model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=0.01), weighted_metrics=['acc'])
# model.summary()

# Callbacks for EarlyStopping
es_callback=EarlyStopping(monitor='val_weighted_acc', patience=PATIENCE)
mc_callback=ModelCheckpoint('best_model.h5',
                            monitor='val_weighted_acc',
                            save_best_only=True,
                            save_weights_only=True)

# Train
graph=[X, A]
validation_data=(graph, y_val, val_mask)
history=model.fit(graph, y_train, sample_weight=train_mask,
                    batch_size=A.shape[0],
                    epochs=NB_EPOCH,
                    verbose=0,
                    validation_data=validation_data,
                    shuffle=False,
                    callbacks=[es_callback, mc_callback])

eval_results=model.evaluate(graph, y_test,
                            sample_weight=test_mask,
                            batch_size=A.shape[0])
print('Test Done.\n'
    'Test loss: {}\n'
    'Test accuracy: {}'.format(*eval_results))














