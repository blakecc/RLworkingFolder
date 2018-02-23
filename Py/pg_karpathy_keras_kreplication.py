import numpy as np
import _pickle as pickle
import gym
import os

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
# from keras.layers import advanced_activations
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from tensorflow.nn import l2_normalize
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import glorot_uniform

# hyperparameters
batchSize = 2048 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
epsilon = 1e-5
resume = False # resume from previous checkpoint?
render = False
ep_limit = 300001
k_hidden_dims = 200

game_history = []

D = 80 * 80 # input dimensionality: 80x80 grid

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

# test_obs = prepro(observation)
#
# test_obs.shape
#
# from matplotlib import pyplot as plt
# plt.imshow(test_obs, interpolation='nearest')
# plt.show()


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

# def build_network(input_dim, output_dim, hidden_dims=[32, 32], batch_size = 8):
#     """Create a base network"""
#
#     visible = layers.Input(shape=(input_dim,))
#     # visible = layers.Input(batch_shape = (None, input_dim[0], input_dim[1], input_dim[2]))
#     # bnorm1 = BatchNormalization()(visible)
#     # conv1 = Conv2D(8, kernel_size=8, activation='relu', padding = "same")(visible)
#     # pool1 = MaxPooling2D(pool_size=(4, 4), padding = "same")(conv1)
#     # net = Conv2D(16, kernel_size=4, activation='relu')(net)
#     # net = MaxPooling2D(pool_size=(2, 2))(net)
#
#     # flat1 = layers.Flatten()(pool1)
#
#     hidden1 = layers.Dense(200)(visible)
#     # relu1 = layers.advanced_activations.LeakyReLU()(hidden1)
#     relu1 = layers.Activation("relu")(hidden1)
#
#     # for h_dim in hidden_dims:
#     #     net = layers.Dense(h_dim)(net)
#     #     net = layers.advanced_activations.LeakyReLU()(net)
#     #     # net = layers.Activation("relu")(net)
#     #     # net = layers.Dropout(rate = 0.2)(net)
#
#     output = layers.Dense(output_dim, activation = "softmax")(relu1)
#     # net = layers.Activation("softmax")(net)
#
#     model = Model(inputs=visible, outputs=output)
#
#     return model
# def customLoss(y_true, y_pred):
#     # TODO: change 2 to an automatic function
#     action_1h = K.one_hot(K.argmax(K.abs(y_true), axis = 1), 2)
#     y_hat = action_1h * y_pred
#     sy_hat = K.sum(y_hat, axis = 1)
#     ly_hat = K.log(sy_hat)
#     ly_hat = K.cast(ly_hat, dtype = 'float32')
#     depr = K.sum(y_true, axis = 1)
#     depr = K.cast(depr, dtype = 'float32')
#     loss = -1 * ly_hat * depr
#     losssum = K.sum(loss, axis = 0)
#     return losssum

def customLoss(y_true, y_pred):
    loss = -1 * K.mean(y_true * K.log(y_pred))
    return loss

    # action_1h = K.one_hot(K.argmax(K.abs(fake_labels), axis = 1), 2)
    # y_pred = kmodel.predict(epx)
    # y_hat = action_1h * y_pred
    # sy_hat = K.sum(y_hat, axis = 1)
    # ly_hat = K.log(sy_hat)
    # depr = K.sum(fake_labels, axis = 1)
    # depr = K.cast(depr, dtype = 'float32')
    # loss = -1 * ly_hat * depr
    # losssum = K.sum(loss, axis = 0)

def build_network(input_dim, output_dim, hidden_dims=k_hidden_dims, lrate = learning_rate, drate = decay_rate, eps = epsilon):
    """Create a base network"""

    model = Sequential()
    model.add(Dense(hidden_dims, kernel_initializer = glorot_uniform(), activation='relu', input_dim=input_dim))
    model.add(Dense(output_dim, kernel_initializer = glorot_uniform(), activation='softmax'))

    rmsprop = optimizers.RMSprop(lr = lrate, rho = drate, epsilon = eps) #Rho is actually what Karpathy thinks of as decay ... in keras decay is the learning rate decay which is not relevant for us

    model.compile(optimizer = rmsprop,
                  loss = customLoss,
                  metrics = ['acc'])

    return model


# def build_train_fn(model, output_dim):
#     """Create a train function
#
#     It replaces `model.fit(X, y)` because we use the output of model and use it for training.
#
#     For example, we need action placeholder
#     called `action_one_hot` that stores, which action we took at state `s`.
#     Hence, we can update the same action.
#
#     This function will create
#     `self.train_fn([state, action_one_hot, discount_reward])`
#     which would train the model.
#
#     """
#     action_prob_placeholder = model.output
#     action_onehot_placeholder = K.placeholder(shape=(None, output_dim),
#                                               name="action_onehot")
#     discount_reward_placeholder = K.placeholder(shape=(None,1),
#                                                 name="discount_reward")
#
#     action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
#     log_action_prob = K.log(action_prob)
#
#     loss = - log_action_prob * discount_reward_placeholder
#     loss = K.mean(loss)
#
#     # TODO: adjust parameters more in line with Karpathy
#     rmsprop1 = optimizers.RMSprop()
#
#     updates = rmsprop1.get_updates(params=model.trainable_weights,
#                                # constraints=[],
#                                loss=loss)
#
#     train_fn = K.function(inputs=[model.input,
#                                        action_onehot_placeholder,
#                                        discount_reward_placeholder],
#                                outputs=[],
#                                updates=updates)
#     return train_fn

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,drs,acts = [],[],[]
running_reward = None

# k_input_shape = np.expand_dims(prepro(observation), axis = 2).shape
k_input_shape = prepro(observation).shape[0]
# k_output_shape = env.action_space.n
k_output_shape = 2

# env.action_space.n

if resume:
    kmodel = build_network(input_dim = k_input_shape, output_dim = k_output_shape, hidden_dims = k_hidden_dims)
    kmodel.load_weights('save_k_01.h5')
    game_history = pickle.load(open('game_history_k_01.p', 'rb'))
    episode_number = game_history[-1][0]
    reward_sum = 0
else:
    episode_number = 0
    reward_sum = 0
    game_history = []
    kmodel = build_network(input_dim = k_input_shape, output_dim = k_output_shape, hidden_dims = k_hidden_dims)

# env.render()


# train_func = build_train_fn(kmodel, k_output_shape)

# done = False

while True:
# while not done:
  if render: env.render()

  # for _ in range(10):
  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(k_input_shape)
  x = np.expand_dims(x, axis = 0)
  prev_x = cur_x

  # prev_x = None
  # x.shape

  # np.expand_dims(x, axis = 2)
  # test_x = np.expand_dims(np.expand_dims(x, axis = 2), axis = 0)
  # test_x = np.expand_dims(np.expand_dims(x, axis = 0), axis = 0)
  # np.squeeze(x)
  # aprob.shape
  # test_x.shape
  # kmodel.summary()

  # forward the policy network and sample an action from the returned probability
  # aprob = np.squeeze(kmodel.predict(np.expand_dims(np.expand_dims(x, axis = 0), axis = 0)))
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

    # endvar = 0
    #
    # import matplotlib.pyplot as plt
    #
    # endvar = 0

  if done: # an episode finished
    game_history.append((episode_number, reward_sum))
    episode_number += 1
    pickle.dump(game_history, open('game_history_k_01.p', 'wb'))

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    # epx = np.vstack(xs)
    #eph = np.vstack(hs)
    # epdlogp = np.vstack(dlogps)
    # epr = np.vstack(drs)


    # # compute the discounted reward backwards through time
    # discounted_epr = discount_rewards(epr)
    # # # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    # discounted_epr -= np.mean(discounted_epr)
    # discounted_epr /= np.std(discounted_epr)
    #
    # epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)

    # eacts = np.vstack(acts)
    #
    # plt.plot(eacts[:,1])
    # plt.show()
    #
    # epx.shape
    # epx[0].sum()
    #
    # eacts[:,1].sum()


    # batchTotal = epx.shape[0]
    # batchStart = 0

    # if batchTotal > batchSize:
    #     batchSteps = int(np.floor(batchTotal / batchSize))
    #     for num_steps in range(batchSteps):
    #         # print(eacts[batchStart:(batchStart + batchSize)])
    #         train_func([epx[batchStart:(batchStart + batchSize)], eacts[batchStart:(batchStart + batchSize)], discounted_epr[batchStart:(batchStart + batchSize)]])
    #         batchStart +=batchSize
    #     if batchSize < batchTotal:
    #         # print(eacts[batchStart:batchTotal])
    #         train_func([epx[batchStart:batchTotal], eacts[batchStart:batchTotal], discounted_epr[batchStart:batchTotal]])
    #
    # if batchTotal > batchSize:
    #     batchSteps = int(np.floor(batchTotal / batchSize))
    #     for num_steps in range(batchSteps):
    #         batch_obs = np.random.choice(range(batchTotal), batchSize)
    #         # print(eacts[batchStart:(batchStart + batchSize)])
    #         train_func([epx[batch_obs,], eacts[batch_obs,], discounted_epr[batch_obs,]])
    #         # batchStart +=batchSize
    #     # if batchSize < batchTotal:
    #     #     # print(eacts[batchStart:batchTotal])
    #     #     train_func([epx[batchStart:batchTotal], eacts[batchStart:batchTotal], discounted_epr[batchStart:batchTotal]])

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode {} reward total was {}. running mean: {}'.format(episode_number, reward_sum, running_reward))

    # train_func([epx, eacts, discounted_epr])
    if episode_number % 10 == 0:
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

    if episode_number % 100 == 0: kmodel.save_weights('save_k_01.h5')
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  # if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
  #   print ('ep {}: game finished, reward: {}'.format(episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))

  if episode_number == ep_limit:
      os._exit()
