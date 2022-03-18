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
colnames(all.data) <- paste(colnames(all.data),"var",sep="_")
head(all.data)
head(train.data)

# impute missing variables using median/mode
all.data.impute <- impute(all.data, method = "median/mode")
na.all.data.impute <- data.frame(sapply(all.data.impute, function(x) as.numeric(as.character(x))))
na.all.data.impute <- sapply(na.all.data.impute, function(x) sum(is.na(x)))
any(na.all.data.impute > 0)

# remove uniqueid, year, id variables
all.data.impute <- all.data.impute[ , -c(1:3)]
# convert to factors 
# step1.categorical <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
#                        "x11", "x12", "x13", "x14", "x15", "x17", "x19", "x20", "x27", "x28", "x29", "x30", "x31", "x32", "x33", "x34",
#                        "x35", "x36", "x37", "x38", "x39", "x40", "x41", "x42", "x43", "x44", 
#                        "x45", "x46", "x47", "x48", "x49", "x50", "x61", "x62", "x63", "x64", 
#                        "x65", "x66", "x67", "x68", "x69", "x70", "x71", "x72", "x73", "x74", 
#                        "x76", "x77", "x78", "x79", "x80", "x81", "x82", "x84", "x85", "x86", 
#                        "x87", "x88", "x89", "x90", "x91", "x92", "x93", "x94", "x95", "x111", 
#                        "x112", "x113", "x114", "x117", "x118", "x119", "x120", "x121", "x122",
#                        "x123", "x124", "x125", "x126", "x127", "x128", "x130", "x131", "x132", 
#                        "x133", "x134", "x135", "x136", "x137", "x138", "x139", "x140", "x141",
#                        "x142", "x143", "x144", "x145", "x146", "x147", "x148", "x149", "x150", 
#                        "x151", "x152", "x153", "x154", "x155", "x156", "x157", "x158", "x159", 
#                        "x160", "x161", "x162", "x163", "x164", "x166", "x167", "x168", "x169", 
#                        "x170", "x171", "x172", "x173", "x174", "x175", "x177", "x179", "x180", 
#                        "x181", "x182", "x183", "x184", "x185", "x188", "x189", "x190", "x191", 
#                        "x192", "x193", "x194", "x195", "x196", "x197", "x198", "x199", "x200", 
#                        "x201", "x202", "x203", "x204", "x205", "x206", "x207", "x208", "x209", 
#                        "x210", "x211", "x212", "x214", "x215", "x216", "x217", "x218", "x219", 
#                        "x220", "x221", "x222", "x223", "x224", "x225", "x226", "x227", "x228", 
#                        "x229", "x230", "x231", "x232", "x233", "x234", "x235", "x238", "x239", 
#                        "x240", "x241", "x242", "x243", "x244", "x245", "x246", "x247", "x248", 
#                        "x249", "x250", "x251", "x252", "x253", "x254", "x255", "x256", "x257",
#                        "x258", "x259", "x260", "x261", "x262", "x263", "x264", "x265", "x266", 
#                        "x267", "x268", "x269", "x270", "x271" , "x272", "x273", "x274", "x275",
#                        "x276", "x277", "x278", "x279", "x280", "x281", "x282", "x283", "x284", 
#                        "x285", "x286", "x287", "x288", "x289", "x290", "x291", "x292", "x293", 
#                        "x294", "x303", "x304", "x305", "x307”", "x308", "x309", "x310", "x311", 
#                        "x312", "x322", "x323", "x324", "x325", "x326", "x327", "x328", "x329",
#                        "x333", "x334", "x354", "x361", "x362", "x363", "x364", "x365", "x366", 
#                        "x367", "x370", "x373", "x374", "x375", "x376", "x377", "x385", "x387", 
#                        "x388", "x389", "x390", "x391", "x392", "x394", "x396", "x397", "x402", 
#                        "x403", "x404", "x405", "x406", "x408", "x409", "x410", "x411", "x412", 
#                        "x413", "x414", "x415", "x416", "x417", "x418", "x419", "x420", "x436", 
#                        "x446", "x448", "x450", "x452", "x456", "x457", "x458", "x459", "x460", 
#                        "x461", "x462", "x463", "x464", "x472", "x477", "x497", "x544", "x545", 
#                        "x546", "x547", "x548", "x563", "x564", "x565", "x566", "x567", "x568", 
#                        "x569", "x595", "x596", "x597", "x598", "x599", "x600", "x601", "x602", 
#                        "x603", "x604", "x605", "x606", "x607", "x608", "x609", "x610")

lower.half.not.categorical <- c("x22", "x23", "x24", "x25", "x26", "x236", "x237", "x386", "x453", "x454", 
                                "x455")
upper.half.not.categorical <- c("x631", "x633", "x665", "x666", "x671", "x674", "x675", "x683", 
                         "x684", "x685","x691", "x692", "x702", "x714", "x715","x723", "x725", 
                         "x726", "x728", "x729", "x730", "x758", "x760", "x761", "x797", "x798",
                         "x845", "x847", "x916", "x917", "x918", "x922", "x925", "x926", "x949",
                         "x950", "x955", "x958", "x959", "x962", "x963", "x967", "x969", "x977",
                         "x979", "x987", "x989", "x1035", "x1036", "x1037", "x1038",
                         "x1040", "x1042", "x1043", "x1049", "x1050", "x1134", "x1140", "x1141",
                         "x1142", "x1145", "x1147", "x1148", "x1149", "x1150", "x1151", "x1152",
                         "x1154")
not.categorical.var <- c(lower.half.not.categorical, upper.half.not.categorical)
categorical.var <- which(!colnames(all.data.impute)%in%not.categorical.var)

for(i in categorical.var){
  all.data.impute[ , i] <- as.factor(all.data.impute[ , i])
}

# separate into train and test
train.data.impute <- cbind(all.data.impute[1:nrow(train.data), ], train.data[ , 1209, drop = FALSE])
test.data.impute <- all.data.impute[(nrow(train.data) + 1):nrow(all.data.impute), ]

# run multinomial regression on training set

# set.seed
set.seed(841)
fit.control <- trainControl(method = "cv", number = 10)
tune.grid <-expand.grid(alpha = 1,
                       lambda = seq(0.001,0.1,by = 0.001))
# create model matrix
train.data.impute.matrix <- model.matrix(health ~ ., data = train.data.impute)
options(expressions = 5e5)
train.health <- train.data.impute$health
multinomial.reg.model <- train(x = train.data.impute.matrix, y = train.health, data = train.data.impute.matrix, method = "glmnet", 
                               family = "multinomial", trControl = fit.control, 
                               tuneGrid = tune.grid)

# predict test data 
test.pred <- predict(multinomial.reg.model, test.data.impute, type = "prob")
test.pred <- cbind(test.data[ , 1, drop = FALSE], test.pred)
colnames(test.pred) <- c("uniqueid", "p1", "p2", "p3", "p4", "p5")

# write predictions to CSV
write.csv(test.pred, paste("multinomial_log_reg_", Sys.Date(), ".csv", sep = ""))


# quit R
# q()



