#%%
from __future__ import absolute_import, division, print_function

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import gym

import tensorflow as tf
from tensorflow import keras

import tf_agents
#%%
env = gym.make("CartPole-v1")
observation = env.reset()

#%%
num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# %%
time_step_spec = tf_agents.trajectories.time_step_spec(
    observation_spec=tf.TensorSpec(shape=observation.shape)
    reward_spec=tf.TensorSpec(shape=1)
)
action_spec = tf_agents.specs.BoundedArraySpec((), tf.int64, minimum=0, maximum=1)

# %%
ppo_agent = tf_agents.agents.PPOClipAgent(
    time_step_spec = time_step_spec,
    action_spec = action_spec,
    optimizer = keras.optimizer.Adam(learning_rate=learning_rate),
    
)

