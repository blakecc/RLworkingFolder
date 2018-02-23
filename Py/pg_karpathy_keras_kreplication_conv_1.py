import numpy as np
import _pickle as pickle
import gym
import os

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.initializers import glorot_uniform
from keras.layers import advanced_activations
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
# from keras.layers.normalization import BatchNormalization
from keras import regularizers

# hyperparameters
batchSize = 512 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
epsilon = 1e-5
resume = False # resume from previous checkpoint?
render = False
ep_limit = 300001
k_hidden_dims = [256, 256]
ep_batch = 8

game_history = []

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float)

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def customLoss(y_true, y_pred):
    loss = -1 * K.mean(y_true * K.log(y_pred))
    return loss

def build_network(input_dim, output_dim, hidden_dims=k_hidden_dims, lrate = learning_rate, drate = decay_rate, eps = epsilon):
    """Create a base network"""
    # TODO: add kernel reg to layers
    model = Sequential()
    # model.add(Conv2D(16, kernel_size = (5, 5), strides = (1, 1), kernel_initializer = glorot_uniform(), input_shape=input_dim))
    model.add(Conv2D(16, kernel_size = (5, 5), strides = (1, 1), input_shape=input_dim, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    # model.add(Conv2D(8, kernel_size = (5, 5), strides = (1, 1), kernel_initializer = glorot_uniform()))
    model.add(Conv2D(8, kernel_size = (5, 5), strides = (1, 1), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer = glorot_uniform(), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim, kernel_initializer = glorot_uniform(), activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

    rmsprop = optimizers.RMSprop(lr = lrate, rho = drate, epsilon = eps, clipvalue=0.5) #Rho is actually what Karpathy thinks of as decay ... in keras decay is the learning rate decay which is not relevant for us

    model.compile(optimizer = rmsprop,
                  loss = customLoss,
                  metrics = ['acc'])

    return model


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,drs,acts = [],[],[]
running_reward = None

# k_input_shape = np.expand_dims(prepro(observation), axis = 2).shape
k_input_shape = (prepro(observation).shape[0], prepro(observation).shape[1], 1)
# k_output_shape = env.action_space.n
k_output_shape = 2

# env.action_space.n

if resume:
    kmodel = build_network(input_dim = k_input_shape, output_dim = k_output_shape, hidden_dims = k_hidden_dims)
    kmodel.load_weights('save_k_conv_1.h5')
    game_history = pickle.load(open('game_history_k_conv_1.p', 'rb'))
    episode_number = game_history[-1][0]
    reward_sum = 0
else:
    episode_number = 0
    reward_sum = 0
    game_history = []
    kmodel = build_network(input_dim = k_input_shape, output_dim = k_output_shape, hidden_dims = k_hidden_dims)

# done = False

while True:
# while not done:
  if render: env.render()

  # for _ in range(10):
  # preprocess the observation, set input to network to be difference image
  # TODO: fix dimensions
  cur_x = np.expand_dims(prepro(observation), axis = 2)
  # cur_x.shape
  # prev_x.shape
  x = cur_x - prev_x if prev_x is not None else np.zeros(k_input_shape)
  # x.shape
  # x = np.expand_dims(np.expand_dims(x, axis = 2), axis = 0)
  x = np.expand_dims(x, axis = 0)
  # x.shape
  prev_x = cur_x

  aprob = np.squeeze(kmodel.predict(x))

  # exploration vs. exploitation
  action = np.random.choice(np.arange(k_output_shape), p=aprob)


  action_onehot = np_utils.to_categorical(action, num_classes=k_output_shape)

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  acts.append(action_onehot)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action + 2)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  # end = 0

  if done: # an episode finished
    game_history.append((episode_number, reward_sum))
    episode_number += 1
    pickle.dump(game_history, open('game_history_k_conv_1.p', 'wb'))

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode {} reward total was {}. running mean: {}'.format(episode_number, reward_sum, running_reward))

    # train_func([epx, eacts, discounted_epr])
    if episode_number % ep_batch == 0:
        epx = np.vstack(xs)
        epr = np.vstack(drs)
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        eacts = np.vstack(acts)

        fake_labels = eacts * discounted_epr
        kmodel.fit(epx, fake_labels, epochs = 1, batch_size=batchSize, verbose = False)
        xs,drs,acts = [],[],[] # reset array memory

    if episode_number % 20 == 0:
        check_weight_1 = np.asarray(kmodel.get_weights()[1])[0].sum()
        check_weight_2 = np.asarray(kmodel.get_weights()[3])[0].sum()
        print('Weight check 1: {}, weight check 2: {}'.format(check_weight_1, check_weight_2))

    if episode_number % 100 == 0: kmodel.save_weights('save_k_conv_1.h5')
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if episode_number == ep_limit:
      os._exit()
