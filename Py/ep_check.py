import pandas as pd
import numpy as np
import _pickle as pickle


game_history = pickle.load(open("game_history_01.p", "rb"))

print(game_history[-1])
