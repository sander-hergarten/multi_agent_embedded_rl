import tensorflow as tf
from tensorflow import keras
import math
import numpy as np


class sparse_autoencoder_model_flat(tf.keras.Model):

    def __init__(self, layer_count=3, input_size=28**2, sparsity_factor=0.1, kl_weight=0.2):
        super(sparse_autoencoder_model_flat, self).__init__()
        
        self.kl_weight = kl_weight
        self.sparsity_factor= sparsity_factor
        self.input_size = input_size
        self.dense_layers = [] 
        self.embed_layer_number = math.ceil(len(self.dense_layers)/2) 

        for n in range(layer_count):
            self.dense_layers.append(keras.layers.Dense(self.input_size,  activation='relu'))

        self.output_layer = keras.layers.Dense(self.input_size, activation='sigmoid')

    # def get_embedding_layer(self):
        

    def call(self, inputs):
        temp_res = inputs
        
        for layers in self.dense_layers:
            temp_res = layers(temp_res)

        return self.output_layer(temp_res)
    
    def call_middle_layer_sat(self):
        """
        returns all activations in the embedding layer when the input is fully saturated
        """

        temp_res = tf.ones((32, self.input_size))

        for n in range(math.ceil(len(self.dense_layers)/2)):
            temp_res = self.dense_layers[n](temp_res)
        return temp_res

    def get_final_layer_activation(self, mult_arr):
        """
        get the final layer activation from the embedding layer.

        for each index a new tensor is created that is only 1 at the selected index
        it is then forward propagated
        """
        temp_res = mult_arr


        for n in range(math.ceil(len(self.dense_layers)/2), len(self.dense_layers)):
            temp_res = self.dense_layers[n](temp_res)
        return temp_res



    def train_step(self, data):
        
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value

            sparsity = tf.reduce_mean(self.call_middle_layer_sat())
            
            loss = self.compiled_loss(y, y_pred,
                                      regularization_losses=self.losses) + (self.kl_weight * keras.losses.kl_divergence([sparsity], self.sparsity_factor))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)



        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
 