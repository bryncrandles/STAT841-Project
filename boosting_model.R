#setwd("~/projects/def-wjmarsha/bc11xx/STAT841-project")
setwd("/Users/bryncrandles/Documents/STAT841/Winter/Project/stat-441-w2022-data-challenge/")
#pacman::p_load(caret, RANN, imputeMissings)
#install.packages("caret")
#install.packages("RANN")
#install.packages("glmnet")
#install.packages("imputeMissings")
# library(RANN, glmnet, imputeMissings)

library(caret)
library(glmnet)
library(RANN)
library(imputeMissings)

rm(list = ls())
# load training data and testing data 
train.data <- read.csv("train.csv")
train.data$health <- as.factor(train.data$health)
test.data <- read.csv("test.csv")
#all.data <- rbind(train.data[ , -1209], test.data)
# convert to numeric to remove NA
#all.data <- data.frame(sapply(all.data, function(x) as.numeric(as.character(x))))
#str(all.data)

train.data <- data.frame(sapply(train.data, function(x) as.numeric(as.character(x))))
str(train.data)

# threshold of missing data: remove variables that are missing more than threshold%
threshold <- 0.8
names(train.data)[sapply(train.data, function(x) mean(is.na(x)) > threshold)]

train.data.no.health <- train.data[ , -1209]
threshold.missing <- unname(which(sapply(train.data.no.health, function(x) mean(is.na(x)) > threshold)))
threshold.missing
train.data.no.health <- train.data.no.health[ , - threshold.missing]
#train.data <- cbind(train.data.no.health, train.data[ , 1209, drop = FALSE])


test.data <- test.data[ , -threshold.missing]
test.data <- data.frame(sapply(test.data, function(x) as.numeric(as.character(x))))
na.test.data <- sapply(test.data, function(x) sum(is.na(x)))
all.na.test <- unname(which(na.test.data == dim(test.data)[1]))
all.na.test

# impute missing variables using median/mode for training
train.data.impute <- impute(train.data.no.health, method = "median/mode")
# remove uniqueid, year, id variables
train.data.impute <- train.data.impute[ , -c(1:3)]
na.train.data.impute <- data.frame(sapply(train.data.impute, function(x) as.numeric(as.character(x))))
na.train.data.impute <- sapply(na.train.data.impute, function(x) sum(is.na(x)))
any(na.train.data.impute > 0)

# remove correlated variables
#correlation.matrix <- cor(train.data.impute)
#which.high.cor <- which(correlation.matrix > 0.7, arr.ind = TRUE)

# impute missing variables using median/mode for testing
test.data.impute <- impute(test.data, method = "median/mode")
na.test.data.impute <- data.frame(sapply(test.data.impute, function(x) as.numeric(as.character(x))))
na.test.data.impute <- sapply(na.test.data.impute, function(x) sum(is.na(x)))
any(na.test.data.impute > 0)

test.data.impute <- test.data.impute[ , -c(1:3)]

# convert to factors

#lower.half.not.categorical <- c("x22", "x23", "x24", "x25", "x26", "x236", "x237", "x386", "x453", "x454", 
#                                "x455")
not.categorical.var <- c("x236", "x237", "x386", "x453", "x454", 
                         "x455", "x631", "x633", "x665", "x666", "x671", "x683", 
                         "x684", "x685","x691", "x692", "x697", "x702", "x714", "x715","x723", "x725", 
                         "x726", "x728", "x729", "x730", "x758", "x760", "x791", "x792", "x797", "x798",
                         "x845", "x847", "x916", "x917", "x918", "x922", "x925", "x926", "x949",
                         "x950", "x955", "x963", "x967", "x969", "x977",
                         "x979", "x987", "x989", "x1035", "x1036", "x1037", "x1038",
                         "x1040", "x1042", "x1043", "x1049", "x1050", "x1134", "x1140", "x1141",
                         "x1142", "x1147", "x1148", "x1149", "x1150", "x1151", "x1152",
                         "x1154", "x1183")
# upper.half.not.categorical <- c("x631", "x633", "x665", "x666", "x671", "x674", "x675", "x683", 
#                                 "x684", "x685","x691", "x692", "x702", "x714", "x715","x723", "x725", 
#                                 "x726", "x728", "x729", "x730", "x758", "x760", "x761", "x797", "x798",
#                                 "x845", "x847", "x916", "x917", "x918", "x922", "x925", "x926", "x949",
#                                 "x950", "x955", "x958", "x959", "x962", "x963", "x967", "x969", "x977",
#                                 "x979", "x987", "x989", "x1035", "x1036", "x1037", "x1038",
#                                 "x1040", "x1042", "x1043", "x1049", "x1050", "x1134", "x1140", "x1141",
#                                 "x1142", "x1145", "x1147", "x1148", "x1149", "x1150", "x1151", "x1152",
#                                 "x1154")
#not.categorical.var <- c(lower.half.not.categorical, upper.half.not.categorical)


categorical.var <- which(!colnames(train.data.impute)%in%not.categorical.var)

for(i in categorical.var){
  train.data.impute[ , i] <- as.factor(train.data.impute[ , i])
  test.data.impute[ ,i] <- as.factor(test.data.impute[ ,i])
}


colnames(train.data.impute) <- paste(colnames(train.data.impute),"var",sep="_")
train.data.impute <- cbind(train.data.impute, train.data[ , 1209, drop = FALSE])
head(train.data.impute)
colnames(test.data.impute) <- paste(colnames(test.data.impute),"var",sep="_")
head(test.data.impute)

# run boosting on training set 

# set.seed
set.seed(841)

# response variable
train.health <- train.data.impute$health

# method of fitting model
fit.control <- trainControl(method = "cv", number = 10)
# tuning parameters
boost.grid <- expand.grid(interaction.depth = c(1, 2),
                          n.trees = (1:20)*100,
                          shrinkage = c(0.01, 0.1),
                          n.minobsinnode = 5)


# run model 
boost.model <- train(health ~ . , data = train.data.impute, method = "gbm",
                     verbose = FALSE, trControl = fit.control, tuneGrid = boost.grid)

best.int.depth <- boost.model$bestTune$interaction.depth
best.shrinkage <- boost.model$bestTune$shrinkage

boost.order.importance <- data.frame(boost.importance$importance[order(boost.importance$importance$Overall, decreasing = TRUE), ])
rownames(boost.order.importance) <- rownames(boost.importance$importance)[order(boost.importance$importance$Overall, decreasing = TRUE)]
colnames(boost.order.importance) <- "Importance"

write.csv(boost.order.importance, paste("boost_model_", Sys.Date(), ".csv", sep = ""))

# predictions
boost.pred <- predict(boost.model, newdata = test.data.impute, type = "prob")
boost.pred <- cbind(test.data[ , 1, drop = FALSE], test.pred)

write.csv(boost.pred, paste("boost_model", Sys.Date(), ".csv", sep = ""))


# try a small model
# train.data.impute.small <- train.data.impute[c(1:300), c(1:3, 681)]
# train.data.impute.small[ ,1] <- as.numeric(train.data.impute.small[ ,1])
# train.data.impute.small[ ,2] <- as.numeric(train.data.impute.small[ ,2])
# train.data.impute.small[ ,3] <- as.numeric(train.data.impute.small[ ,3])
# boost.model.small1 <- train(health ~., data = train.data.impute.small,  method = "gbm",
#                            verbose = FALSE, trControl = fit.control, tuneGrid = boost.grid)
# # numeric
# test.data.impute.small <- test.data.impute[ , c(1:3)]
# test.data.impute.small[ , 1] <- as.numeric(test.data.impute.small[ ,1])
# test.data.impute.small[ , 2] <- as.numeric(test.data.impute.small[ , 2])
# test.data.impute.small[ , 3] <- as.numeric(test.data.impute.small[ , 3])
# pred1 <- predict(boost.model.small1, newdata = test.data.impute.small, type = "raw")
# pred1
# 
# 
# # categorical
# train.data.impute.small <- train.data.impute[c(1:300), c(1:3, 681)]
# test.data.impute.small <- test.data.impute[ , c(1:3)]
# boost.model.small2 <- train(health ~., data = train.data.impute.small,  method = "gbm",
#                            verbose = FALSE, trControl = fit.control, tuneGrid = boost.grid)
# pred2 <- predict(boost.model.small, newdata = test.data.impute.small, type = "raw")
# mean(pred2 == pred1)
# 
# pred1
# 


# quit R
q()
