

# EMS Feature Engineering - ClinSpaCy -------------------------------------

# This code uses ClinSpacy to return non-negated embeddings for Chief Narratives in EMS records. 


# Pre-Flight --------------------------------------------------------------

# Load Libraries 
library(tidyverse)
library(clinspacy)
library(caret)

# Load data with format patient_id | nature | primary_impression | chief_complaint | secondary_complaint | rx_ref_list | full_hx_dx | class
data <- ""

# Get ClinSpaCy vectors ---------------------------------------------------

# Extract target columns 
data_to_vec <- data %>% select(class,patient_id,chief_narrative) 

# NER data and retrieve embeddings 
# (This step may require batching or parallelization depending on data size and available compute)
data_parsed<- data_to_vec  %>% clinspacy(df_col = "chief_narrative", return_scispacy_embeddings = TRUE) 

# Filter out negated entities and combine with full data 
data_combined <-  data_parsed %>% filter(is_negated != TRUE) %>% 
  bind_clinspacy_embeddings(data_to_vec)



# Split and write out -----------------------------------------------------

set.seed(2023)

# Randomly select 80% of cases for the training set 
train_set <- all_batches %>% slice_sample(prop =.8)

# Write out 
saveRDS(train_set,"spacy_features_train_set.RDS")


# Reserve the remaining 20% for the test set 
test_set <- all_batches %>% anti_join(train_set, by="patient_id")

# Write out 
saveRDS(test_set,"spacy_features_test_set.RDS")


