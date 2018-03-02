library(feather)
library(tidyverse)
library(TTR)

game_history <- read_feather("../Histories/game_history_H2FCA2048FCB256reluBS4096LR0.0001CV0CN32EG0.005.feather")

game_history <- game_history %>%
  mutate(EMA100 = EMA(Score, 100))

gh_plot <- game_history %>%
  ggplot() +
    geom_point(aes(x = Ep, y = Score, colour = "Episode scores")) +
    # geom_smooth(aes(x = Ep, y = Score, colour = "Trend")) +
    geom_line(aes(x = Ep, y = EMA100, colour = "100 Episode EMA")) +
    geom_hline(aes(yintercept = 0, colour = "Win line")) +
    ggtitle("History: H2FCA2048FCB256reluBS4096LR0.0001CV0CN32EG0.005") +
    xlab("Episode number") +
    ylab("Episode net score") +
    scale_colour_discrete(name = "Legend") +
    theme_light()

ggsave(filename = "figs/game_history_H2FCA2048FCB256reluBS4096LR0.0001CV0CN32EG0.005.png", plot = gh_plot, device = "png"
       # , units = "mm",
       # width = 100,
       # height = 50
       )
