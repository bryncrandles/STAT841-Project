setwd("~/projects/def-wjmarsha/bc11xx/STAT841-project")
#setwd("/Users/bryncrandles/Documents/STAT841/Winter/Project/stat-441-w2022-data-challenge/")

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

# remove health variable
train.data.no.health <- train.data[ , -1209]


# remove x1145, x1146, x1147, x1149, x1150: redundancies in times
remove.time.var <- c("x1145", "x1146", "x1147", "x1148", "x1149", "x1150")
remove.time.var.ind <- which(colnames(train.data.no.health)%in%remove.time.var)
colnames(train.data.no.health)[remove.time.var.ind]
colnames(test.data)[remove.time.var.ind]
train.data.no.health <- train.data.no.health[ , -remove.time.var.ind]
test.data <- test.data[ , -remove.time.var.ind]

# remove variables missing over 75%
# threshold of missing data: remove variables that are missing more than threshold%
threshold <- 0.75
threshold.missing <- unname(which(sapply(train.data.no.health, function(x) mean(is.na(x)) > threshold)))
threshold.missing
train.data.no.health <- train.data.no.health[ , - threshold.missing]
test.data <- test.data[ , -threshold.missing]
# check that there are no all NA in test data
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

# impute missing variables using median/mode for testing
test.data.impute <- impute(test.data, method = "median/mode")
na.test.data.impute <- data.frame(sapply(test.data.impute, function(x) as.numeric(as.character(x))))
na.test.data.impute <- sapply(na.test.data.impute, function(x) sum(is.na(x)))
any(na.test.data.impute > 0)
# remove uniqueid, year, id variables
test.data.impute <- test.data.impute[ , -c(1:3)]

# remove correlated variables
correlation.matrix <- cor(train.data.impute)
length(findCorrelation(correlation.matrix,  cutoff = 0.8))
which.correlated <- findCorrelation(correlation.matrix,  cutoff = 0.8)
train.data.impute <- train.data.impute[ , - which.correlated]
test.data.impute <- test.data.impute[ , -which.correlated]


# add back health variable
train.data.impute <- cbind(train.data.impute, train.data[ , 1209, drop = FALSE])
train.data.impute$health <- as.factor(train.data.impute$health)

# add "var" to colnames
colnames(train.data.impute) <- paste(colnames(train.data.impute),"var",sep="_")
head(train.data.impute)
colnames(test.data.impute) <- paste(colnames(test.data.impute),"var",sep="_")
head(test.data.impute)

# select important variables from random forest variable selection
rforest.importance <- read.csv("/Users/bryncrandles/Documents/STAT841/Winter/Project/STAT841-Project/random_forest_importance2_2022-03-21.csv")
important.var <- rforest.importance$X[rforest.importance$Importance >= 5]
important.var.ind <- which(colnames(train.data.impute)%in%important.var)
train.data.impute.imp.var <- train.data.impute[ , c(important.var.ind, 339)]


# convert to factors

# not.categorical.var <- c("x236", "x237", "x386", "x453", "x454", 
#                           "x455", "x631", "x633", "x665", "x666", "x671", "x683", 
#                                 "x684", "x685","x691", "x692", "x697", "x702", "x714", "x715","x723", "x725", 
#                                 "x726", "x728", "x729", "x730", "x758", "x760", "x791", "x792", "x797", "x798",
#                                 "x845", "x847", "x916", "x917", "x918", "x922", "x925", "x926", "x949",
#                                 "x950", "x955", "x963", "x967", "x969", "x977",
#                                 "x979", "x987", "x989", "x1035", "x1036", "x1037", "x1038",
#                                 "x1040", "x1042", "x1043", "x1049", "x1050", "x1134", "x1140", "x1141",
#                                 "x1142", "x1151", "x1152",
#                                 "x1154", "x1183")
# categorical.var <- which(!colnames(train.data.impute)%in%not.categorical.var)
# 
# for(i in categorical.var){
#   train.data.impute[ , i] <- as.factor(train.data.impute[ , i])
#   test.data.impute[ ,i] <- as.factor(test.data.impute[ ,i])
# }


# filterCtrl <- sbfControl(functions = rfSBF, method = "repeatedcv", repeats = 1, number = 10)
# set.seed(841)
# rfWithFilter <- sbf(x, y, sbfControl = filterCtrl)
# rfWithFilter




