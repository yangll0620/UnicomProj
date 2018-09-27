from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Input
from keras.callbacks import EarlyStopping
from keras import optimizers

import numpy as np

def model_Dense3Layers_training(x_train, y_train):
# input: 	x_train: n_trials * n_features
# 			y_train: n_trials * 1

	batch_size = 100
	epochs = 50
	[_,n_input] = x_train.shape
	_, label_counts = np.unique(y_train, return_counts = True)
	ratio = round(float(label_counts[0])/float(label_counts[1]))
	class_weight = {0:1,1:ratio}

	model = Model()
	input = Input(shape = (n_input,))
	x = Dense(n_input,activation= 'relu')(input)
	#model.add(Dense(n_input,activation= 'relu',input_shape = (n_input,)))
	x = Dropout(0.25)(x)
	x = Dense(int(n_input/2), activation = 'relu')(x)
	x = Dropout(0.25)(x)
	output = Dense(1,activation = 'sigmoid')(x)
	model = Model(inputs = input, outputs = output)

	Adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, 
		patience = 5, mode = 'auto', verbose = 1)
	callbacks_list = [earlystop]
	model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,validation_split=0.2, 
		callbacks=callbacks_list,class_weight = class_weight, verbose=1)
	return model

def model_Dense3Layers_prediction(model, x_test):
	y_prob = model.predict(x_test)
	y_pred = np.zeros(np.array(y_prob).shape)
	y_pred[np.array(y_prob)>=0.5] = 1
	return y_pred



