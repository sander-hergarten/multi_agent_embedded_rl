#%%
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import tensorflow.keras.backend as K
import math
import matplotlib.pyplot as plt
import sparse_autoencoder
#%%
# Settings
img_size = 28**2



# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

#%%

#%%
model = sparse_autoencoder.sparse_autoencoder_model_flat()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mse'])
model.fit(x_train, x_train, batch_size=32)

# %%
a = model.predict(np.array([x_test[1]]))
Image.fromarray(x_test[1].reshape((28, 28))*256).show()
Image.fromarray(a[0].reshape((28, 28))*256).show()
# %%

a = model.call_middle_layer_sat()[0].numpy()
Image.fromarray(a.reshape((28, 28))*256).show()

# %%
k = list(a.nonzero()[0])
sparse_activation_list = []
for s in k:
    temp = np.zeros(img_size)
    temp[s] = 1
    sparse_activation_list.append(temp)

# %%
embedd_out = model.get_final_layer_activation(np.array(sparse_activation_list)).numpy()
# %%

# %%

# %%
