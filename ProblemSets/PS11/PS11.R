#ThePack
#install.packages("glmnet", repos='http://cran.us.r-project.org')
#install.packages("mlr", repos='http://cran.us.r-project.org')
#install.packages("rpart", repos='http://cran.us.r-project.org')
#install.packages("e1071", repos='http://cran.us.r-project.org')
#install.packages("kknn", repos='http://cran.us.r-project.org')
#install.packages("nnet", repos='http://cran.us.r-project.org')
library(kknn)
library(e1071)
library(rpart)
library(nnet)
library(mlr)
library(tidyverse)
library(magrittr)
library(glmnet)
library(dplyr)
set.seed(100)
############################################################################################################################

#ReadInCSV
gradstats <- read.csv('https://stats.idre.ucla.edu/stat/data/binary.csv')
gradstats2 <- gradstats %>% slice(seq(1,10)) %>% select(gre, gpa, rank)

gradstats2[1,] <- c(500,4.00,1)

############################################################################################################################

#GradStatsDescription
#admit: binary
#GRE: continuous gre scores
#GPA: continuous on 4.0 scale
#Rank: categorical rank 1 to 4 with 4 being the lowest
#
############################################################################################################################

#Data Cleaning
gradstats$gre <- as.numeric(gradstats$gre)
gradstats$gpa <- as.numeric(gradstats$gpa)
gradstats$rank <- as.factor(gradstats$rank)

formula <- as.formula(admit ~ .^3 +
                        poly (gre, 6) +
                        poly (gpa, 6))
mod_matrix <- data.frame(model.matrix(formula, gradstats))
#now replace the intercept column by the response since MLR will do
#"y ~ ." and get the intercept by default
mod_matrix[, 1] = gradstats$admit
colnames(mod_matrix )[1] = "admit" #make sure to rename it otherwise MLR won ’t find it
head(mod_matrix ) #just make sure everything is hunky -dory

############################################################################################################################

#Data Cleaning for ficticious data
gradstats2$gre <- as.numeric(gradstats2$gre)
gradstats2$gpa <- as.numeric(gradstats2$gpa)
gradstats2$rank <- as.factor(gradstats2$rank)

formula <- as.formula( ~ .^3 +
                        poly (gre, 6) +
                        poly (gpa, 6))
gradstats2.test <- data.frame(model.matrix(formula, gradstats2))

#gradstats2.test[, 1] <- gradstats$admit[seq(1,nrow(gradstats2))]
gradstats2.test[, 1] <- NA
colnames(gradstats2.test )[1] = "admit" #make sure to rename it otherwise MLR won ’t find it
head(gradstats2.test ) #just make sure everything is hunky -dory
############################################################################################################################
############################################################################################################################

#Break up data

n <- nrow(gradstats)
train <- sample(n, size= .8*n)
test <- setdiff(1:n, train)
gradstats.train <- mod_matrix[train,]
gradstats.test <- mod_matrix[test,]
############################################################################################################################

######################
# Classification tasks
######################
# Same classification task for all
classifTask <- makeClassifTask(data = gradstats.train, target = "admit")

# Same CV strategy for all
cv.info <- makeResampleDesc(method = "CV", iters = 3)

# Same tuning strategy for all
tuneMethod <- makeTuneControlRandom(maxit = 30L)

# Set all prediction algorithms up front
alg.tree  <- makeLearner("classif.rpart",      predict.type = "prob")
alg.logit <- makeLearner("classif.glmnet",     predict.type = "prob")
#alg.nnet  <- makeLearner("classif.nnet",       predict.type = "prob")
alg.nb    <- makeLearner("classif.naiveBayes", predict.type = "prob")
alg.knn   <- makeLearner("classif.kknn",       predict.type = "prob")
#alg.svm   <- makeLearner("classif.svm",        predict.type = "prob")

#---------------------------------------------------------------
# Set up hyperparameter space for CV for all algorithms up front
#---------------------------------------------------------------
# tree model: tree depth, min leaf, etc.
parms.tree  <- makeParamSet(makeIntegerParam("minsplit",lower = 10, upper = 50), makeIntegerParam("minbucket", lower = 5,   upper = 50), makeNumericParam("cp", lower = 0.001, upper = 0.2))
# logit model: lambda and elastic net weights
parms.logit <- makeParamSet(makeNumericParam("lambda",  lower = 0,  upper = 3 ), makeNumericParam("alpha",     lower = 0,   upper = 1  ))
# neural network model: lambda and size of hidden layer
#parms.nnet  <- makeParamSet(makeIntegerParam("size",    lower = 1,  upper = 10), makeNumericParam("decay",     lower = 0.1, upper = 0.5), makeNumericParam("maxit", lower = 5000, upper = 5000))
# knn: number of neighbors
parms.knn <- makeParamSet(makeIntegerParam("k",  lower = 1,  upper = 30 ))
# svm: cost, gamma
#parms.svm <- makeParamSet(makeDiscreteParam("cost", values = c(2^-2,2^-1,2^0,2^1,2^2,2^10)), makeDiscreteParam("gamma", values = c(2^-2,2^-1,2^0,2^1,2^2,2^10)), makeDiscreteParam("kernel", values = c("radial")))
# naive bayes: none!


# Do the tuning for all at once
tuner.tree  <- tuneParams(learner = alg.tree,  task = classifTask, resampling = cv.info, measures = list(f1,gmean), par.set = parms.tree,  control = tuneMethod, show.info = TRUE)
tuner.logit <- tuneParams(learner = alg.logit, task = classifTask, resampling = cv.info, measures = list(f1,gmean), par.set = parms.logit, control = tuneMethod, show.info = TRUE)
#tuner.nnet  <- tuneParams(learner = alg.nnet,  task = classifTask, resampling = cv.info, measures = list(f1,gmean), par.set = parms.nnet,  control = tuneMethod, show.info = TRUE)
tuner.knn   <- tuneParams(learner = alg.knn,   task = classifTask, resampling = cv.info, measures = list(f1,gmean), par.set = parms.knn,   control = tuneMethod, show.info = TRUE)
#tuner.svm   <- tuneParams(learner = alg.svm,   task = classifTask, resampling = cv.info, measures = list(f1,gmean), par.set = parms.svm,   control = tuneMethod, show.info = TRUE)

# Apply the optimal algorithm parameters to the model
alg.tree  <- setHyperPars(learner=alg.tree , par.vals = tuner.tree$x )
alg.logit <- setHyperPars(learner=alg.logit, par.vals = tuner.logit$x)
#alg.nnet  <- setHyperPars(learner=alg.nnet , par.vals = tuner.nnet$x )
alg.knn   <- setHyperPars(learner=alg.knn  , par.vals = tuner.knn$x  )
#alg.svm   <- setHyperPars(learner=alg.svm  , par.vals = tuner.svm$x  )

# Verify performance on cross validated sample sets
#resample(alg.tree,classifTask,cv.info,measures=list(f1,kappa,gmean))

# Train the final models
final.tree  <- train(learner = alg.tree,  task = classifTask)
final.logit <- train(learner = alg.logit, task = classifTask)
#final.nnet  <- train(learner = alg.nnet,  task = classifTask)
final.nb    <- train(learner = alg.nb,    task = classifTask)
final.knn   <- train(learner = alg.knn,   task = classifTask)
#final.svm   <- train(learner = alg.svm,   task = classifTask)

# Predict in test set!
pred.tree  <- predict(final.tree,  newdata = gradstats.test)
pred.logit <- predict(final.logit, newdata = gradstats.test)
#pred.nnet  <- predict(final.nnet,  newdata = gradstats.test)
pred.nb    <- predict(final.nb,    newdata = gradstats.test)
pred.knn   <- predict(final.knn,   newdata = gradstats.test)
#pred.svm   <- predict(final.svm,   newdata = gradstats.test)

# Performance
print("tree performance")
print(performance(pred.tree,  measures = list(f1,gmean)))
print("logit performance")
print(performance(pred.logit, measures = list(f1,gmean)))
#print("nnet performance")
#print(performance(pred.nnet,  measures = list(f1,gmean)))
print("nb performance")
print(performance(pred.nb,    measures = list(f1,gmean)))
print("knn performance")
print(performance(pred.knn,   measures = list(f1,gmean)))
#print("svm performance")
#print(performance(pred.svm,   measures = list(f1,gmean)))

#########################################################################################################

pred2.tree  <- predict(final.tree,  newdata = gradstats2.test)
pred2.logit <- predict(final.logit, newdata = gradstats2.test)
#pred2.nnet  <- predict(final.nnet,  newdata = gradstats2.test)
pred2.nb    <- predict(final.nb,    newdata = gradstats2.test)
pred2.knn   <- predict(final.knn,   newdata = gradstats2.test)
#pred2.svm   <- predict(final.svm,   newdata = gradstats2.test)

print(pred2.knn$data$prob.1)

