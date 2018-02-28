import feather
import _pickle as pickle
import pandas as pd

game_history = pickle.load(open('../Histories/game_history_k_conv_3_LR2e5_ClipNorm.p', 'rb'))
game_history = pd.DataFrame(game_history, columns = ["Ep", "Score"])

path = "../Histories/game_history_k_conv_3_LR2e5_ClipNorm.feather"

feather.write_dataframe(game_history, path)
