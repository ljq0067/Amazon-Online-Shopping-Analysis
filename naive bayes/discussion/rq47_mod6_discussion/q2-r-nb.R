# ANLY-501 mod6 discussion naïve bayes with R

# Rui Qiu (rq47)
# 2021-11-08

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, naivebayes, tidymodels)

dat <- read.csv("data/pokemon_data_science.csv")
glimpse(dat)

set.seed(42)

dat <- dat %>%
    select(-c(Type_1, Type_2, Number,
              Name, Egg_Group_1, Egg_Group_2,
              Body_Style, Catch_Rate))

dat_split <- initial_split(dat, prop=0.8)
train <- training(dat_split)
test <- testing(dat_split)

nb_clf <- naive_bayes(isLegendary ~., data = train, usekernel = T)
train_pred <- predict(nb_clf, train)
test_pred <- predict(nb_clf, test)

table(train_pred, train$isLegendary)

(526+36)/(526+36+14)

table(test_pred, test$isLegendary)

(132+10)/(132+10+3)
