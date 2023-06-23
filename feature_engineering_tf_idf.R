
# TFIDF Feature Engineering -----------------------------------------------

# This code is designed to support EMS narrative data feature engineering for subsequent modeling using TF-IDF. 
# Note: This code splits the train and test sets to avoid data leakage in IDF calculation.    

# Pre-Flight --------------------------------------------------------------

# Load Libraries 
library(tidyverse)
library(tidytext)


# Load data with the following column names
#[1] "patient_id"         "nature"             "primary_impression" "chief_complaint"    "chief_narrative"    "rx_ref_list"        "full_dx_hx"        
#[8] "class" 


# Merge text data for multi-field TF-IDF 
data_merged <- data %>% mutate(text = paste(chief_complaint,chief_narrative,sep=" "))

# preserve labels
labels <- data %>% select(patient_id,class)

# FEATURE ENGINEERING -----------------------------------------------------


# Train set engineering ---------------------------------------------------

set.seed(2023)

# Randomly select 80% of cases for the training set 
train_set <- data %>% slice_sample(prop =.8)

# unnest tokens and get word counts 
data_counts_train <- map_df(1:2,
                            ~ unnest_tokens(train_set, word, text, 
                                            token = "ngrams", n = .x)) %>%
  anti_join(stop_words, by = "word") %>%
  count(patient_id, word, sort = TRUE)

# Feature reduction (use only tokens that appear in at least 500 cases )
words_reduce_train <- data_counts_train %>%
  group_by(word) %>%
  summarise(n = n()) %>% 
  filter(n >= 500) %>%
  select(word)

# Bind TF-IDF and cast to DTM 
train_dtm <- data_counts_train %>%
  right_join(words_reduce_train, by = "word") %>%
  bind_tf_idf(word, patient_id, n) %>%
  cast_dtm(patient_id, word, tf_idf) %>% as.matrix() %>% 
  as.data.frame() %>% 
  mutate(patient_id=rownames(.)) 


# Test set engineering ----------------------------------------------------

# Reserve the remaining 20% for the test set 
test_set <- data %>% anti_join(train_set, by="patient_id")

# unnest tokens and get word counts 
data_counts_test<- map_df(1:2,
                          ~ unnest_tokens(test_set, word, text, 
                                          token = "ngrams", n = .x)) %>%
  anti_join(stop_words, by = "word") %>%
  count(patient_id, word, sort = TRUE)

# Retain features that appear in training set 
words_train_test <- data_counts_test %>%
  group_by(word) %>%
  summarise(n = n()) %>% 
  filter(word %in% colnames(train_dtm)[-ncol(train_dtm)]) %>%
  select(word)

# Bind TF-IDF and cast to DTM 
test_dtm <- data_counts_test %>%
  right_join(words_train_test, by = "word") %>%
  bind_tf_idf(word, patient_id, n) %>%
  cast_dtm(patient_id, word, tf_idf) %>% as.matrix() %>% 
  as.data.frame() %>% 
  mutate(patient_id=rownames(.)) 


# Align test and train columns --------------------------------------------

train_dtm_labeled <- train_dtm %>% 
  left_join(labels) 

test_set_labeled <- 
  test_dtm %>% 
  left_join(labels) 


# Write out train and test sets  ------------------------------------------

saveRDS(train_dtm_labeled,"tfidf_features_train_set.RDS")

saveRDS(test_set_labeled,"tfidf_features_test_set.RDS")
