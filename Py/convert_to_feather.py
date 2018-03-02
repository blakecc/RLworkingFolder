# import feather
import _pickle as pickle
import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline
# from ggplot import *

#change working directory
# import os
# os.chdir("/Users/blakecuningham/Dropbox/MScDataScience/Thesis/RLworkingFolder/Py/")

name = "game_history_H2FCA2048FCB256reluBS4096LR0.0001CV0CN32EG0.005"
filename = "../Histories/" + name + ".p"

game_history = pickle.load(open(filename, 'rb'))
game_history_plot = pd.DataFrame(game_history, columns = ["Ep", "Score"])

path = "../Histories/" + name + ".feather"

feather.write_dataframe(game_history, path)

# game_history_plot["EMA100"] = game_history_plot["Score"].ewm(span = 100).mean()
# newplot = (ggplot(aes(x="Ep", y="Score"), data = game_history_plot) +
#             geom_point(color = "green") +
#             geom_line(aes(x = "Ep", y = "EMA100"), color = "blue") +
#             geom_hline(y = 0, color = "darkorange") +
#             ggtitle(name))
#
#
# newplot.save(name + ".png")
