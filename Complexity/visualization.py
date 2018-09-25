from model_Dense3Layers import model_Dense3Layers_training
from keras.utils.vis_utils import plot_model
import numpy as np

x_train = np.random.rand(5000,44)
y_train = np.zeros((5000,),dtype = int)
y_train[2500:] = 1

model = model_Dense3Layers_training(x_train, y_train)
plot_model(model, to_file ='ANNmodel.png', show_shapes = True, 
	show_layer_names = True)