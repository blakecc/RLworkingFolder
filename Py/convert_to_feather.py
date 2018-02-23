import feather
import _pickle as pickle
import pandas as pd

game_history = pickle.load(open('game_history_k_02.p', 'rb'))
game_history = pd.DataFrame(game_history, columns = ["Ep", "Score"])

path = "game_history_k_02.feather"

feather.write_dataframe(game_history, path)
