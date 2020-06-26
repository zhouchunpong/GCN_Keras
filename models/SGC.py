from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


from graph import GraphConvolution
from utils import *


# Define parameters
NB_EPOCH = 200
PATIENCE = 200  # early stopping patience


# Get data
# X:features  A:graph  y:labels
X, A, y = load_data(dataset='cora', use_feature=True)
y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(y)

# Normalize X
X /= X.sum(1).reshape(-1, 1)
print('ssss',X.shape[1])

A_ = preprocess_adj(A,power=2)
graph = [X, A_]
G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
X_in = Input(shape=(X.shape[1],)) # the number of feature


Y = GraphConvolution(y_train.shape[1], kernel_regularizer=l2(5e-6),activation='softmax')([X_in]+G)
# Compile model
model = Model(inputs=[X_in]+G, outputs=Y) # inputs=graph=[X,A_]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.2), weighted_metrics=['acc'])
model.summary()



# Callbacks for EarlyStopping
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=PATIENCE)

# Train
validation_data = (graph, y_val, val_mask)
model.fit(graph, y_train, sample_weight=train_mask,
          batch_size=A.shape[0], 
          epochs=NB_EPOCH,
          verbose=1,
          validation_data=validation_data,
          shuffle=False,
          callbacks=[es_callback])

# Evaluate model on the test data
eval_results = model.evaluate(graph,y_test,
                              sample_weight=test_mask,
                              batch_size=A.shape[0])
print('Test Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
