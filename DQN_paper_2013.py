
#v2 keep working in TF2.3.0
#v3 added DQN and used a FIFO for the REPLAY_MEMORY (now it is working very slowly)
#v4 used pre-processing 84x84x4 as for DeepMind and adapted DQN for that input
#v5 improved ReplayMemory to use less RAM (derived from code fg91)

import gym
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import datetime
from collections import deque
import time
import random
from tqdm import tqdm
from PIL import Image
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#Possible values are as follows:
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf
from tensorflow import keras
import keras.backend.tensorflow_backend as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution

if tf.executing_eagerly():
    print('Eagerly is running')
else:
    print('Eagerly is disables')

print(f'tensorflow version {tf.__version__}')
print(f'keras version {tf.keras.__version__}')

#env = gym.make("MountainCar-v0")
env = gym.make("Breakout-v4")
#env = gym.make("PongDeterministic-v4")
#env = gym.make("CartPole-v1")


DEBUG = False #True
SHOW_EVERY = 200

DISCOUNT = 0.99                 # Discount factor gamma used in the Q-learning update
LEARNING_RATE = 0.00025         # 0.00001   0.00025 Mnih et all 2015
                                  # Hessel et al. 2017 used 0.0000625
                                # in Pong use 0.00025
REPLAY_MEMORY_SIZE = 1000000      #100000  # How many last steps to keep in memory for model training
MIN_REPLAY_MEMORY_SIZE = 50000   #50000  # Minimum number of random steps before to start training with the memory
                                # This is also the Number of completely random actions before the agent starts learning
MAX_FRAMES = 25000000           #50milion # Total number of frames the agent sees during training
MINIBATCH_SIZE = 32             # How many steps (samples) to use for training
UPDATE_TARGET_MODEL =10000       #10000 # Number of chosen actions between updating the target network.
                                       # According to Mnih et al. 2015 this is measured in the number of
                                       # parameter updates (every four actions), however, in the
                                       # DeepMind code, it is clearly measured in the number
                                       # of actions the agent choses
UPDATE_MODEL = 4

MODEL_NAME = 'DQN_DeepMind'
MIN_REWARD = -200  # For model save

EPISODES = 20000   # number of match played in total during training

# Exploration annealing settings
epsilon = 1  # not a constant, going to be decayed
MIN_EPSILON = 0.01
EPSILON_DECAY_1 = -(1-0.1)/(1000000-MIN_REPLAY_MEMORY_SIZE)
EPSILON_DECAY_2 = -(0.1-MIN_EPSILON)/(MAX_FRAMES-1000000)


#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes/match

# used to print on the terminal
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

#DQN
# Memory fraction, used mostly when training multiple agents
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)])

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir = self.log_dir + '\\train'
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # added OLI according with https://stackoverflow.com/questions/61549039/im-getting-an-error-of-modifiedtensorboard-object-has-no-attribute-write-lo
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class DQNAgent():
    def __init__(self):
        # Main model
        self.model = self.create_model()
        print(self.model.summary())

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), input_shape=[84, 84, 4], strides=4,
                         kernel_initializer=keras.initializers.VarianceScaling(scale=2.0)))  # 4 frame greyscale 84x84
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(64, (4, 4), strides=2, kernel_initializer=keras.initializers.VarianceScaling(scale=2.0)))
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), strides=1, kernel_initializer=keras.initializers.VarianceScaling(scale=2.0)))
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(512, kernel_initializer=keras.initializers.VarianceScaling(scale=2.0), activation='relu'))

        model.add(Dense(env.action_space.n, activation='linear',
                        kernel_initializer=keras.initializers.VarianceScaling(scale=2.0)))  # action_space = how many choices (2)
        #model.compile(loss="mse", optimizer=Adam(lr=0.00025), metrics=['accuracy'])
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        return model

    # Clip reward so it is between -1 and 1
    def clip_reward(self, reward):
        if reward < -1:
            reward = -1
        elif reward > 1:
            reward = 1
        return reward

    # Trains main network every step during episode
    def train(self, minibatch):

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array(minibatch[0])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        new_current_states = np.array(minibatch[3])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        # Now we need to enumerate our batches
        for index in range(MINIBATCH_SIZE):
            action            = minibatch[1][index]
            current_state     = minibatch[0][index]
            reward            = minibatch[2][index]
            new_current_state = minibatch[3][index]
            done              = minibatch[4][index]

            # Bellman equation.  Q = r + gamma*max Q',
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            Y.append(current_qs)

        # Fit on all samples as one batch, log file saved
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[tensorboard_callback])

        # Fit on all samples as one batch, NO log file saved. Quicker simulation
        self.model.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)


    # update target model every 1000000 frame as verified into main (9*)
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

class ReplayMemory(object):  # derived from https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent performed  [a_t]
            frame: A (84, 84, 1) frame of an Atari game in grayscale reached due to the action [s_t+1]
            reward: A float determining the reward the agent received for performing an action [r_t]
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        """
        We store all frames the agent sees in self.frames.
        When a game terminates (terminal=True) at index i, frame at index i belongs
        to a different episode than the frame at i+1. We want to avoid creating a state
        with frames from two different episodes.
        Finally we need to make sure that an index is not smaller than the number of
        frames stacked toghether to create a state (self.agent_history_length=4),
        so that a state and new_state can be sliced out of the array.
        """

        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        minibatch = (np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[
                     self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices])
        return minibatch

def pre_processing (frame):
    # single Frame Processor from 210x160x3 to 84x84x1
    frame_gray = np.dot(frame, [0.299, 0.587, 0.114])  # 210x160  convert gray scale
    #plt.imshow(np.array(np.squeeze(frame_gray)), cmap='gray')
    #plt.show()
    frame_gray = frame_gray[31:195, 0:160]  # crop off upper score (31 lines) and below black area (15 lines)
    #resized_img0 = Image.fromarray(frame_gray).resize(size=(84, 84), resample=Image.BILINEAR)  # 84x84x1
    resized_img0 = Image.fromarray(frame_gray).resize(size=(84, 84), resample=Image.NEAREST)  # 84x84x1
    #plt.imshow(np.array(np.squeeze(resized_img0)), cmap='gray')
    #plt.show()
    return asarray(resized_img0, dtype=np.uint8)

def pre_processing_old1 (frame):
    # single Frame Processor from 210x160x3 to 84x84x1
    screen = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #plt.imshow(np.array(np.squeeze(screen)), cmap='gray')
    #plt.show()
    screen = screen[32:210, 0:160]  # crop off score
    screen = cv2.resize(screen, (84, 84))
    screen = screen.reshape(84, 84)
    #plt.imshow(np.array(np.squeeze(screen)), cmap='gray')
    #plt.show()
    return screen
def pre_processing_old2 (frame):
    # single Frame Processor from 210x160x3 to 84x84x1
    processed = tf.image.rgb_to_grayscale(frame)
    processed = tf.image.crop_to_bounding_box(processed, 34, 0, 160, 160)
    processed = tf.image.resize(processed, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #plt.imshow(np.array(np.squeeze(processed)), cmap='gray')
    #plt.show()
    return asarray(processed, dtype=np.uint8)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

agent = DQNAgent()
print("The environment has the following {} actions: {}".format(env.action_space.n,
                                                                env.unwrapped.get_action_meanings()))


my_replay_memory = ReplayMemory(size=REPLAY_MEMORY_SIZE, batch_size=MINIBATCH_SIZE)   # (★)
frame_number = 0   # total number of step conducted from the first match till the last
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):  #1 episode = 1 match
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_frame = env.reset()   # single frame 84x84
    current_frame = pre_processing(current_frame)
    current_state = np.dstack((current_frame, current_frame, current_frame, current_frame)) # create imm 84x84 grouping in 4 frames

    # Reset flag and start iterating until episode ends
    done = False
    while not done:   #in the environment MountainCar-v0 done=False after 200 step
                      #in the environment Breakout-v4 done=False after lost all lifes (num life =5)
        # Exploration-exploitation trade-off
        # Take new action
        # train main network
        # Set new state
        # Add new reward

        # Exploration-exploitation trade-off after a number of steps with completely random action  (4★)
        if np.random.random() > epsilon and frame_number > MIN_REPLAY_MEMORY_SIZE:
            # Get action from DQN model: exploit
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action: explore
            action = np.random.randint(0, env.action_space.n)

        # Take new action (5★)
        new_frame, reward, done, _ = env.step(action)
        reward = agent.clip_reward(reward)
        # Set and preprocess new state (6★)
        new_frame = pre_processing(new_frame)  # single frame 84x84 preprocessed
        new_state = np.dstack((new_frame, current_state[:, :, 0], current_state[:, :, 1], current_state[:, :, 2]))# create imm 84x84 grouping in 4 frames

        episode_reward += reward

        if episode % SHOW_EVERY == 0: # and DEBUG: # plot one match every SHOW_EVERY
            #time.sleep(0.01)
            #print(f'Step: {step}')
            env.render()

        # Every step/frame we update replay memory with action, new frame due to the actio, reward due to the action  (7★)
        my_replay_memory.add_experience(action, new_frame, reward, done)

        # Every step we evaluate to train main network and/or to update weights of target network
        if frame_number % UPDATE_MODEL == 0 and frame_number > MIN_REPLAY_MEMORY_SIZE:  # model update every 4 frame/action
            # Get a minibatch of random samples from the replay memory
            minibatch = my_replay_memory.get_minibatch()
            agent.train(minibatch)                  # (8★)
            if DEBUG:  # plot of minibatch used to train (only 1 imm over 4 into the frame)
                fig = plt.figure(figsize=(100, 200))
                for i in range(32):
                    plt.subplot(4, 8, i + 1)
                    plt.imshow(minibatch[0][i, :, :, 0], cmap='jet')
                plt.show()
            del minibatch
        if frame_number % UPDATE_TARGET_MODEL == 0 and frame_number > MIN_REPLAY_MEMORY_SIZE: # model target update every 10000 frame/action
            agent.update_target_model()    # (9★)
        #


        # Set new state (9*)
        current_state = new_state
        step += 1
        frame_number += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if episode % 10 == 0 and episode>AGGREGATE_STATS_EVERY:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)
        print(f'Episode:{episode:>5d}, frame_number:{frame_number:>7d}, avg rew:{average_reward:>4.1f}, max rew:{max(ep_rewards[-AGGREGATE_STATS_EVERY:]):>4.1f}, min rew:{min(ep_rewards[-AGGREGATE_STATS_EVERY:]):>4.1f} current epsilon:{epsilon:>1.5f}')

    # Decay epsilon. Only start after replay memory is over min size
    if frame_number > MIN_REPLAY_MEMORY_SIZE:
        if frame_number < 1000000:
            epsilon = 0.1 + EPSILON_DECAY_1*(frame_number-1000000)
        else:
            epsilon = MIN_EPSILON + EPSILON_DECAY_2*(frame_number - MAX_FRAMES)


env.close()
# Save model,
agent.model.save(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=2)
plt.grid(True)
plt.show()
