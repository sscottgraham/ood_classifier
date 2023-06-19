
# OD Prediction Modeling  -------------------------------------------------
# This script supports modeling and ensembling for binary predictions. 

# Pre-Flight --------------------------------------------------------------

# Load Libraries 
library(tidyverse)
library(caret)
library(caretEnsemble)
library(xgboost)


#load training & testing data prepared using feature engineering code (select only one feature set).

# TFIDF features 
train_set_od <- read_rds("tfidf_features_train_set.RDS") %>% select(-patient_id)
test_set_od <- read_rds("tfidf_features_test_set_corrected.RDS") %>% select(-patient_id)

# Flag features  
train_set_od <- read_rds("flag_features_train_set.RDS") %>% select(-patient_id)
test_set_od <- read_rds("flag_features_test_set_corrected.RDS") %>% select(-patient_id)

# ClinspaCy features 
train_set_od <- read_rds("spacy_features_train_set.RDS") %>% select(-patient_id)
test_set_od <- read_rds("spacy_features_test_set_corrected.RDS") %>% select(-patient_id)


# Modeling ---------------------------------------------------------------

# set train control
my_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 10,
  classProbs = TRUE)


# Create individual models and assemble into caret list 
model_list <- caretList(
  class~., data=train_set_od,
  trControl=my_control,
  methodList=c("glm"),
  tuneList=list(
    nnet=caretModelSpec(method = "nnet", trace=FALSE, tuneLength=8),
    naive_bayes=caretModelSpec(method = "naive_bayes", tuneLength=8),
    xgbTree=caretModelSpec(method = "xgbTree", tuneLength=8)
  )
)


# Use all models to predict class probability on test set
p_model_list <- as.data.frame(predict(model_list, newdata=test_set_od))


# Predict test set classes using each model 
p_model_raw_nnet <- predict(model_list$nnet, newdata=test_set_od, type="raw")

p_model_raw_glm<- predict(model_list$glm, newdata=test_set_od, type="raw")

p_model_raw_naive_bayes <- predict(model_list$naive_bayes , newdata=test_set_od, type="raw")

p_model_raw_xgbTree <- predict(model_list$xgbTree, newdata=test_set_od, type="raw")


# Assemble ensemble model
ensemble <- caretEnsemble(
  model_list, 
  metric="ROC",
  trControl=trainControl(
    number=2,
    summaryFunction=twoClassSummary,
    classProbs=TRUE
  ))


# Predict test set class probabilities using the ensemble model 
p_ensemble_prob <- predict(ensemble, newdata=test_set_od, type = "prob")

# Predict test set classes using the ensemble model 
p_ensemble_raw<- predict(ensemble, newdata=test_set_od)

