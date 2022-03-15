#setwd("~/projects/def-wjmarsha/bc11xx/STAT841-project")
setwd("/Users/bryncrandles/Documents/STAT841/Winter/Project/stat-441-w2022-data-challenge/")
pacman::p_load(caret, RANN, imputeMissings)
rm(list = ls())
# load training data and testing data 
train.data <- read.csv("train.csv")
test.data <- read.csv("test.csv")
all.data <- rbind(train.data[ , -1209], test.data)
# convert to numeric to remove NA
all.data <- data.frame(sapply(all.data, function(x) as.numeric(as.character(x))))
str(all.data)

# threshold of missing data: remove variables that are missing more than threshold%
threshold <- 0.85
names(all.data)[sapply(all.data, function(x) mean(is.na(x)) > threshold)]

threshold.missing <- unname(which(sapply(all.data, function(x) mean(is.na(x)) > threshold)))
threshold.missing
all.data <- all.data[ , - threshold.missing]
head(all.data)
head(train.data)

# impute missing variables using median/mode
all.data.impute <- impute(all.data, method = "median/mode")
na.all.data.impute <- data.frame(sapply(all.data.impute, function(x) as.numeric(as.character(x))))
na.all.data.impute <- sapply(na.all.data.impute, function(x) sum(is.na(x)))
any(na.all.data.impute > 0)

# remove uniqueid, year, id variables
all.data.impute <- all.data.impute[ , -c(1:3)]

# separate into train and test
train.data.impute <- all.data.impute[1:nrow(train.data), ]
test.data.impute <- all.data.impute[(nrow(train.data) + 1):nrow(all.data.impute), ]

# run multinomial regression on training set

# set.seed
set.seed(841)
fit.control <- trainControl(method = "cv", number = 10)
multinomial.reg.model <- train(health ~ ., data = train.data.impute, method = "glmnet", family = "multinomial")

# predict test data 
test.pred <- predict(multinomial.reg.model, test.data.impute, type = "prob")
test.pred <- cbind(test.data[ , 1, drop = FALSE], test.pred)
colnames(test.pred) <- c("uniqueid", "p1", "p2", "p3", "p4", "p5")

# write predictions to CSV
write.csv(test.pred, paste("multinomial_log_reg_", Sys.Date(), ".csv", sep = ""))

# quit R
q()


test.model.matrix <- model.matrix(test.data.impute)

