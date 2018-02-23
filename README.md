## Intro

Building on Andrej Karpathy's Pong learning algorithm. Intention to eventually build a policy gradient learner with either Keras or TensorFlow learning, and incorporate an evolutionary algorithm to find optimal neural network structures.

## Scripts
pg_karpathy.py is mostly Andrej Karpathy original code, but updated for Python 3.

pg_karpathy_keras_kreplication_3.py is latest test version with Keras backend, RMSprop, gradient clipping, L2 reg, and two hidden layers, with flattening in model.

Convolutional version currently not yielding better than random results - trying to reduce regularization in next tests.


