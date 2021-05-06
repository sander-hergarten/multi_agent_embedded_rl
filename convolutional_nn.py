#%% 
import tensorflow as tf
from tensorflow import keras
# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# %%

image_input = keras.Input(shape=(28, 28, 1))