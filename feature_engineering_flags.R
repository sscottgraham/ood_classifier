
# EMS Feature Engineering  ------------------------------------------------

# This code is designed to support EMS narrative data feature engineering for subsequent modeling using the keyword flagging approach.  


# Pre-Flight --------------------------------------------------------------

# Load Libraries 
library(tidyverse)


# Load data with format patient_id | nature | primary_impression | chief_complaint | secondary_complaint | rx_ref_list | full_hx_dx | class
data <- ""

# Load term list
opioid_terms <- c("opioid", "opiate", "narcan", "naloxone", "heroin", "methadone", "fentanyl","percocet", "oxycontin", "oxycodone", "vicodin", "morphine", "crack", "cocaine", "tylenol 3", "codeine", "oxy", "tramadol") %>% paste0(.,collapse = "|")

od_terms <- c("ingestion", "substance", "abuse", "intox", "poisoning","\\bod\\b", "overdose") %>% paste0(.,collapse = "|")


# Get flag data -----------------------------------------------------------

data_flagged <- data %>% pivot_longer(cols = nature:full_dx_hx, names_to = "field", values_to = "text") %>% 
  mutate(
    opioid_flag = str_count(text,regex(opioid_terms, ignore_case = T)),
    od_flag = str_count(text,regex(od_terms, ignore_case = T)),
    across(.cols=opioid_flag:od_flag,~replace_na(.x,0))
  ) %>% 
  pivot_longer(cols=opioid_flag:od_flag, names_to = "flag", values_to = "count") %>% 
  select(patient_id,class,field,flag,count) %>% 
  mutate(variable = paste(field,flag,sep="_")) %>% 
  unique() %>% 
  pivot_wider(id_cols=patient_id:class,names_from=variable,values_from=count) 


# Create splits and write out ---------------------------------------------

set.seed(2023)

# Randomly select 80% of cases for the training set 
train_set <- data_flagged %>% slice_sample(prop =.8)

# Write out 
saveRDS(train_set,"flag_features_train_set.RDS")


# Reserve the remaining 20% for the test set 
test_set <- data_flagged %>% anti_join(train_set, by="patient_id")

# Write out 
saveRDS(test_set,"flag_features_test_set.RDS")




