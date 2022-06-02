library(MASS)
library(xgboost)
library(caret)
library(tidyverse)
library(dplyr)
library(brant)
library(randomForest)
require(fmsb)

df_for_modeling <- readRDS("df_for_modeling.rds")

df_for_modeling %>% 
  filter(atc == "C10AA05") %>% 
  count(dienas_deva) %>% 
  arrange(desc(n)) %>% 
  as.data.frame() %>% 
  head(5)

medication_1_df <- df_for_modeling %>%  
  filter(atc == "C10AA05" & 
           dienas_deva %in% c(80, 40, 20)) %>% 
  mutate(dienas_deva = case_when(dienas_deva == 20 ~ "mg20",
                                 dienas_deva == 40 ~ "mg40",
                                 dienas_deva == 80 ~ "mg80",)) %>% 
  mutate(dienas_deva = as.factor(dienas_deva)) %>% 
  dplyr::select(dzim, vec, grupa_Acute_infarction, grupa_Cancer, grupa_Cerebrovascular , 
                grupa_Congestive_failure, grupa_Connective_disorder, grupa_Dementia, grupa_Diabetes, 
                grupa_Diabetes_complications, grupa_Hemiplegia_paraplegia, grupa_HIV, grupa_Liver, grupa_Metastatic, 
                grupa_Peptic , grupa_Peripheral_disease, grupa_Pulmonary,  grupa_Renal, grupa_Severe_disease, hosp_n, dienas_deva) 

summary(medication_1_df)
medication_1_df <- medication_1_df %>% 
  select_if(function(col) length(unique(col)) > 1) 

medication_1_df <- medication_1_df %>% 
  dplyr::select(-grupa_Metastatic)

medication_1_df$dienas_deva <- factor(medication_1_df$dienas_deva, ordered = TRUE, levels = c("mg20", "mg40", "mg80")) 

summary(medication_1_df)
str(medication_1_df)
prop_df <- medication_1_df %>% 
  group_by(dienas_deva) %>% 
  summarise(cnt = n()) %>% 
  mutate(prop = cnt/sum(cnt)) %>% as.data.frame()

#### "dumb" logLoss ####

M <- 3
dumb_log_loss <- -log(1/M)
print(dumb_log_loss)

#### non-informative log loss ####

non_info_log_loss <- -(prop_df$prop[1]*log(prop_df$prop[1]) + prop_df$prop[2] * log(prop_df$prop[2]) + prop_df$prop[3] * log(prop_df$prop[3]))


#### prepare data for training and test set  ####

index_1 <- createDataPartition(medication_1_df$dienas_deva, p = .8, 
                               list = FALSE, 
                               times = 1) %>% as.vector()

### ORDINAL REGRESSION ######

medication_1_ord_df <- medication_1_df
summary(medication_1_ord_df)
str(medication_1_ord_df)

#### Creating training and test set ####

datatrain_1 <-  medication_1_ord_df[index_1,]
datatest_1 <-  medication_1_ord_df[-index_1,]
summary(datatest_1)

datatrain_1 %>% 
  group_by(dienas_deva) %>% 
  summarise(cnt = n()) %>% 
  mutate(prop = cnt/sum(cnt)) %>% as.data.frame()

### POLR ####

med_1_polr <- polr(dienas_deva ~ dzim + vec + grupa_Acute_infarction + grupa_Cancer + grupa_Cerebrovascular + grupa_Congestive_failure +
                     grupa_Connective_disorder + grupa_Dementia + grupa_Diabetes + grupa_Diabetes_complications + grupa_Hemiplegia_paraplegia +
                     grupa_Liver + grupa_Peptic + grupa_Peripheral_disease + grupa_Pulmonary +
                     grupa_Renal + hosp_n, data = datatrain_1, Hess = TRUE, method = "logistic")
summary(med_1_polr)

require(stats)
med_1_predict_polr <- predict (med_1_polr, newdata = datatest_1, type = "probs")
require(ModelMetrics)
logloss_polr_med1 <- mlogLoss(actual = datatest_1$dienas_deva , predicted = med_1_predict_polr )
logloss_polr_med1
#Compute confusion table and misclassification error
predict_dienas_deva_polr <- predict(med_1_polr, datatest_1)
table(datatest_1$dienas_deva, predict_dienas_deva_polr)
mean(as.character(datatest_1$dienas_deva) != as.character(predict_dienas_deva_polr))
med_1_polr_error <- mean(as.character(datatest_1$dienas_deva) != as.character(predict_dienas_deva_polr))
Kappa.test(table(datatest_1$dienas_deva, predict_dienas_deva_polr))
caret::confusionMatrix(datatest_1$dienas_deva, predict_dienas_deva_polr, mode = "everything")


polr_mod_med_1 <- med_1_polr



### RANDOM FOREST ######

medication_1_forest_df <- medication_1_df
str(medication_1_forest_df)
forest_train_1 <- medication_1_forest_df[index_1,]
forest_valid_1 <- medication_1_forest_df[-index_1,]
summary(forest_train_1)
summary(forest_valid_1)

# Labāko parametru meklēšanai. Pēc tās svarīgi pārlādēt sesiju, jo kaut kādai pakotnei kaut kas nepatīk 
# med_1_forest <- randomForest(dienas_deva ~ ., data = forest_train_1, ntree = 150, mtry = 10, importance = TRUE)
# med_1_forest
# 
# tuneRF(forest_train_1[,-19],forest_train_1$dienas_deva, 
#        ntreeTry = 150,
#        mtryStart = 5,
#        stepFactor = 1.5,
#        improve = 0.01,
#        trace = TRUE)
# 
# 
# require(stats)
# predict_forest <- predict(med_1_forest, data = forest_valid_1, type = "response")
# require(ModelMetrics)
# mlogLoss(actual = forest_valid_1$dienas_deva, predicted = predict_forest$predictions)


##################### RANDOM FOREST WITH RANGER ##########################
require(ranger)
# set.seed(1234)
# tune.gridranger <- expand.grid(mtry = c(2:15),
#                               splitrule = "gini",
#                               min.node.size = 1)## Cross validation for ordered log regression #####
# train_controlKFCV <- trainControl(method = "cv",
#                                       number =10 ,
#                                       classProbs = TRUE ,
#                                       summaryFunction = mnLogLoss)
# 
# train.rf <- train (dienas_deva ~ dzim + vec + grupa_Acute_infarction + grupa_Cancer + grupa_Cerebrovascular + grupa_Congestive_failure +
#                        grupa_Connective_disorder + grupa_Dementia + grupa_Diabetes + grupa_Diabetes_complications + grupa_Hemiplegia_paraplegia +
#                        grupa_Liver  + grupa_Peptic + grupa_Peripheral_disease + grupa_Pulmonary +
#                        grupa_Renal + hosp_n,
#                       data = forest_train_1 ,
#                       trControl = train_controlKFCV ,
#                       tuneGrid = tune.gridranger ,
#                       method = "ranger",
#                       metric = "logLoss")
# print (train.rf) #12 vai #11
# require (magicfor)
# magic_for(print, silent = TRUE )
# 
# for (i in seq(from = 100, to = 5000, by = 50)) {
#   set.seed(1234)
#   med_1_rf_trees <- ranger(dienas_deva ~ dzim + vec + grupa_Acute_infarction + grupa_Cancer + grupa_Cerebrovascular + grupa_Congestive_failure +
#                              grupa_Connective_disorder + grupa_Dementia + grupa_Diabetes + grupa_Diabetes_complications + grupa_Hemiplegia_paraplegia +
#                              grupa_Liver + grupa_Peptic + grupa_Peripheral_disease + grupa_Pulmonary +
#                              grupa_Renal + hosp_n,
#                            data = forest_train_1,
#                            num.trees = i,
#                            mtry = 11,
#                            splitrule = "gini",
#                            min.node.size = 1,
#                            probability = TRUE)
#   predict_rf_trees <- stats::predict(med_1_rf_trees,
#                                      data = forest_train_1,
#                                      type = "response")
#   print(ModelMetrics::mlogLoss(actual = forest_train_1$dienas_deva,
#                                predicted = predict_rf_trees$prediction))
# }
# 
# require(forcats)
# train.rf.trees <- magic_result_as_dataframe()
# colnames(train.rf.trees)
# 
# names(train.rf.trees)[names(train.rf.trees) == "i"] <- "num.trees"
# names(train.rf.trees)[names(train.rf.trees) == "ModelMetrics::mlogLoss(actual=forest_train_1$dienas_deva,predicted=predict_rf_trees$prediction)"] <- "logLoss"
# 
# view(train.rf.trees)
# require(ggplot2)
# ggplot(data = train.rf.trees, aes(train.rf.trees$num.trees,
#                                   train.rf.trees$logLoss)) + geom_smooth(method = "loess", se = FALSE)







require(ranger)
med_1_forest_model <- ranger(dienas_deva ~.,
                             data = forest_train_1,
                             mtry = 11,
                             num.trees = 150,
                             splitrule = "gini",
                             probability = TRUE,
                             importance = "impurity",
                             classification = TRUE)


print(med_1_forest_model)
class(med_1_forest_model$variable.importance)

require(stats)
predict_forest <- predict(med_1_forest_model, forest_valid_1, type = "response")
require(ModelMetrics)
logloss_forest_med_1 <- mlogLoss(actual = forest_valid_1$dienas_deva, predicted = predict_forest$predictions)

#Izmetam not important variables 
med_1_forest_model <- ranger(dienas_deva ~ dzim + vec + grupa_Acute_infarction + grupa_Cancer + grupa_Cerebrovascular + 
                               grupa_Congestive_failure + grupa_Diabetes + grupa_Diabetes_complications + grupa_Liver +
                               grupa_Peptic + grupa_Peripheral_disease + grupa_Pulmonary + grupa_Renal +  hosp_n,
                             data = forest_train_1,
                             mtry = 11,
                             num.trees = 350,
                             splitrule = "gini",
                             probability = TRUE,
                             importance = "impurity",
                             classification = TRUE)
print(med_1_forest_model)

### predict
require(stats)
predict_forest <- predict(med_1_forest_model, forest_valid_1, type = "response")
require(ModelMetrics)
logloss_forest_med_1 <- mlogLoss(actual = forest_valid_1$dienas_deva, predicted = predict_forest$predictions)

med_1_forest_model <- ranger(dienas_deva ~ dzim + vec + grupa_Acute_infarction + grupa_Cancer + grupa_Cerebrovascular + 
                               grupa_Congestive_failure + grupa_Diabetes + grupa_Diabetes_complications + grupa_Liver +
                               grupa_Peptic + grupa_Peripheral_disease + grupa_Pulmonary + grupa_Renal +  hosp_n,
                             data = forest_train_1,
                             mtry = 11,
                             num.trees = 350,
                             splitrule = "gini",
                             probability = FALSE,
                             importance = "impurity",
                             classification = TRUE)


#Compute confusion table and misclassification error
predict_deva_forest <- predict(med_1_forest_model, forest_valid_1, type = "response")
table(forest_valid_1$dienas_deva, predict_deva_forest$predictions)
mean(as.character(forest_valid_1$dienas_deva) != as.character(predict_deva_forest$predictions))
med_1_forest_error <- mean(as.character(forest_valid_1$dienas_deva) != as.character(predict_deva_forest$predictions))
importance(med_1_forest_model)        

Kappa.test(table(forest_valid_1$dienas_deva, predict_deva_forest$predictions))
forest_mod_med_1 <- med_1_forest_model
caret::confusionMatrix(forest_valid_1$dienas_deva, predict_deva_forest$predictions, mode = "everything")
forest_mod_med_1 <- med_1_forest_model


#### XGBOOOST #################

require(xgboost)
xgb_medication_1_df <- medication_1_df 

for (i in (1:17) ) {
  xgb_medication_1_df[i] <- as.numeric(unlist(xgb_medication_1_df[i]))
}


dienas_deva  <-  xgb_medication_1_df$dienas_deva
label <-  as.integer(xgb_medication_1_df$dienas_deva)-1
xgb_medication_1_df$dienas_deva <-  NULL

n <-  nrow(xgb_medication_1_df)

train_data <-  as.matrix(xgb_medication_1_df[index_1,])
train_label <-  label[index_1]
test_data <-  as.matrix(xgb_medication_1_df[-index_1,])
test_label <-  label[-index_1]


# Transform the two data sets into xgb.Matrix
xgb_train_med_1 <- xgb.DMatrix(data = train_data, label = train_label)
xgb_test_med_1 <-  xgb.DMatrix(data = test_data, label = test_label)


# trctrl <- trainControl(method = "cv", number = 5)
# 
# tune_grid <- expand.grid(nrounds=c(100,200,300,400,500),
#                          max_depth = c(3:7),
#                          eta = c(0.05, 0.3, 1),
#                          gamma = c(0.01),
#                          colsample_bytree = c(0.75),
#                          subsample = c(0.50),
#                          min_child_weight = c(0))
# 
# rf_fit <- train(dienas_deva ~., data = forest_train_1, method = "xgbTree",
#                 trControl=trctrl,
#                 tuneGrid = tune_grid,
#                 tuneLength = 10)

# Define the parameters for multinomial classification
num_class <-  length(levels(dienas_deva))
params = list(
  booster="gbtree",
  eta=0.3,
  max_depth=6,
  gamma=0.01,
  #subsample=0.75,
  colsample_bytree = 0.75,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = num_class
)

# Train the XGBoost classifer
xgb_mod_med_1 <- xgb.train(
  params = params,
  data = xgb_train_med_1,
  nrounds = 300,
  nthreads = 1,
  early_stopping_rounds = 20,
  watchlist = list(val1 = xgb_train_med_1, val2 = xgb_test_med_1),
  verbose = 0
)

# Review the final model and results
xgb_mod_med_1 

# Predict outcomes with the test data
xgb_pred_mod_1 <-  predict(xgb_mod_med_1, test_data, reshape = T)
xgb_pred_mod_1 <-  as.data.frame(xgb_pred_mod_1)
colnames(xgb_pred_mod_1) <- levels(dienas_deva)

#Use the predicted label with the highest probability
xgb_pred_mod_1$prediction <-  apply(xgb_pred_mod_1, 1, function(x) colnames(xgb_pred_mod_1)[which.max(x)])
xgb_pred_mod_1$label <-  levels(dienas_deva)[test_label + 1]


# Calculate the final accuracy
xgb_result  <-  sum(xgb_pred_mod_1$prediction==xgb_pred_mod_1$label)/nrow(xgb_pred_mod_1)
table(xgb_pred_mod_1$prediction, xgb_pred_mod_1$label)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*xgb_result)))

caret::confusionMatrix(as.factor(xgb_pred_mod_1$label), as.factor(xgb_pred_mod_1$prediction), mode = "everything")
#### final
xgb_mod_med_1


#### ADHERENCE ################

rec_adh_365_all_info_df <- readRDS("rec_adh_365_all_info_df.rds")

one_year_df_all_info <- rec_adh_365_all_info_df %>% 
  filter(atpr_dat < 750) 

one_year_med_1_df <- one_year_df_all_info %>% 
  filter(atc == "C10AA05")

#### predict polr ####
one_year_med_1_df$pred_deva_polr <- predict(polr_mod_med_1, one_year_med_1_df)

#### predict random forest #####
one_year_med_1_df$pred_deva_forest <- (predict(forest_mod_med_1, one_year_med_1_df, type = "response"))$prediction

#### predict XGB ####

dienas_deva  <-  one_year_med_1_df %>% filter(dienas_deva %in% c(20, 40, 80)) %>% mutate(dienas_deva = as.factor(dienas_deva))
dienas_deva <- dienas_deva$dienas_deva

data_for_xgb_med_1 <- one_year_med_1_df %>% 
  dplyr::select(colnames(medication_1_df))

for (i in (1:17) ) {
  data_for_xgb_med_1[i] <- as.numeric(unlist(data_for_xgb_med_1[i]))
}

data_for_xgb_med_1$dienas_deva <-  NULL
data_for_xgb_med_1 <- as.matrix(data_for_xgb_med_1)

# Predict outcomes with the test data
xgb_pred_med_1 <-  predict(xgb_mod_med_1, data_for_xgb_med_1, reshape = T)
xgb_pred_med_1 <-  as.data.frame(xgb_pred_med_1)
colnames(xgb_pred_med_1) <- levels(dienas_deva)


#Use the predicted label with the highest probability
xgb_pred_med_1$prediction <-  apply(xgb_pred_med_1, 1, function(x) colnames(xgb_pred_med_1)[which.max(x)])

#############

one_year_med_1_df$pred_deva_xgb <- as.numeric(xgb_pred_med_1$prediction)

### Modify the data  ###############

one_year_med_1_df <- one_year_med_1_df %>% 
  mutate(pred_deva_polr = as.numeric(case_when(pred_deva_polr == "mg80" ~ 80,
                                               pred_deva_polr == "mg40" ~ 40,
                                               pred_deva_polr == "mg20" ~ 20)),
         pred_deva_forest = as.numeric(case_when(pred_deva_forest == "mg80" ~ 80,
                                                 pred_deva_forest == "mg40" ~ 40,
                                                 pred_deva_forest == "mg20" ~ 20))) %>% 
  left_join(df_for_modeling %>% 
              dplyr::select(pid, atpr_dat, atc, dienas_deva, is_good_for_sample) %>% 
              filter (dienas_deva %in% c(20, 40, 80)) %>% 
              dplyr::select(-dienas_deva) , by = c("pid", "atc", "atpr_dat")) 

one_year_med_1_df <- one_year_med_1_df %>% 
  mutate(pred_deva_polr = if_else(!is.na(is_good_for_sample) & !is.na(dienas_deva),
                                  dienas_deva,
                                  pred_deva_polr),
         pred_deva_forest = if_else(!is.na(is_good_for_sample)  & !is.na(dienas_deva),
                                    dienas_deva,
                                    pred_deva_forest),
         pred_deva_xgb = if_else(!is.na(is_good_for_sample)  & !is.na(dienas_deva),
                                 dienas_deva,
                                 pred_deva_xgb))

one_year_med_1_df <- one_year_med_1_df %>% 
  mutate(nosegt_dienas_polr = atpr_mg/pred_deva_polr,
         nosegt_dienas_forest = atpr_mg/pred_deva_forest,
         nosegt_dienas_xgb = atpr_mg/pred_deva_xgb)

# Tas ir nepieciešams funkcijas CMA7() darbībai 
one_year_med_1_df <- one_year_med_1_df %>% 
  mutate(atpr_dat_mod = as.Date("2020-01-01") + atpr_dat)



######### skiergriezuma tabulas modeļi vs 1 tab dienā

table(one_year_med_1_df$stipr_mg, one_year_med_1_df$pred_deva_polr)
table(one_year_med_1_df$stipr_mg, one_year_med_1_df$pred_deva_forest)
table(one_year_med_1_df$stipr_mg, one_year_med_1_df$pred_deva_xgb)

require(AdhereR)

# 
# df_for_adh_med_1 <- one_year_med_1_df %>% 
#   select(pid, atpr_dat_mod, nosegt_dienas_polr) 

adhere_polr_med1 <- CMA7(data = one_year_med_1_df,
                         ID.colname = "pid",
                         event.date.colname = "atpr_dat_mod",
                         event.duration.colname = "nosegt_dienas_polr",
                         #medication.groups = "atc",
                         carry.only.for.same.medication = TRUE,
                         followup.window.start=0, 
                         followup.window.duration = 365,
                         observation.window.start=0, 
                         observation.window.duration=365) 


adhere_forest_med1 <- CMA7(data = one_year_med_1_df,
                           ID.colname = "pid",
                           event.date.colname = "atpr_dat_mod",
                           event.duration.colname = "nosegt_dienas_forest",
                           #medication.groups = "atc",
                           carry.only.for.same.medication = TRUE,
                           followup.window.start=0, 
                           followup.window.duration = 365,
                           observation.window.start=0, 
                           observation.window.duration=365) #,
#date.format="%Y-%m-%d")


adhere_xgb_med1 <- CMA7(data = one_year_med_1_df,
                        ID.colname = "pid",
                        event.date.colname = "atpr_dat_mod",
                        event.duration.colname = "nosegt_dienas_xgb",
                        #medication.groups = "atc",
                        carry.only.for.same.medication = TRUE,
                        followup.window.start=0, 
                        followup.window.duration = 365,
                        observation.window.start=0, 
                        observation.window.duration=365) #,



adhere_ddd_med1 <- CMA7(data = one_year_med_1_df,
                        ID.colname = "pid",
                        event.date.colname = "atpr_dat_mod",
                        event.duration.colname = "nosegt_dienas_ddd",
                        #medication.groups = "atc",
                        carry.only.for.same.medication = TRUE,
                        followup.window.start=0, 
                        followup.window.duration = 365,
                        observation.window.start=0, 
                        observation.window.duration=365) #,

adhere_tab_med1 <- CMA7(data = one_year_med_1_df,
                        ID.colname = "pid",
                        event.date.colname = "atpr_dat_mod",
                        event.duration.colname = "nosegt_dienas_tab",
                        #medication.groups = "atc",
                        carry.only.for.same.medication = TRUE,
                        followup.window.start=0, 
                        followup.window.duration = 365,
                        observation.window.start=0, 
                        observation.window.duration=365) #,


adherence_med_1 <- adhere_ddd_med1$CMA %>% 
  rename(DDD = CMA) %>% 
  left_join(adhere_tab_med1$CMA %>% 
              rename(TAB = CMA), by = "pid") %>% 
  left_join(adhere_polr_med1$CMA %>% 
              rename(PI = CMA), by = "pid") %>% 
  left_join(adhere_forest_med1$CMA %>% 
              rename(GM = CMA), by = "pid") %>% 
  left_join(adhere_xgb_med1$CMA %>% 
              rename(XGB = CMA), by = "pid") 



saveRDS(adherence_med_1, "adherence_med_1.rds")

adherence_med_1_for_plot <- adherence_med_1 %>% 
  pivot_longer(!pid, names_to = "metode", values_to = "adherence")

adherence_med_1_for_plot %>% 
  group_by(metode) %>% 
  summarise(mediana = median(adherence),
            iqr = IQR(adherence))

adh_p1 <- ggplot(adherence_med_1_for_plot, aes(y = adherence, x = metode, fill = metode)) +
  geom_boxplot() +
  theme(
    plot.title = element_text(size = 30, face = "bold"),
    text = element_text(size=25),
    legend.position="none"
  ) +
  ggtitle("Atorvastatīns (C10AA05)", ) +
  xlab("") +
  font("xy.text", size = 25) +
  ylab("zāļu līdzestība") +
  scale_y_continuous(labels = scales::percent) 


# ggsave(adh_p1, filename = "New folder/bildes/adh_med_1.png",
#        height = 9,
#        width = 16
# )


# par(mfrow=c(2,2))
# par(mfrow=c(1,1))

#ggplot(adherence_med_1_for_plot, aes(y = adherence, fill = metode)) +
#  geom_histogram()

plot(adhere_ddd_med1, 
     patients.to.plot=c("L_0f6c9dd07b06fc"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title = "DDD",
     print.dose=TRUE, plot.dose=TRUE,
     show.cma = FALSE)

plot(adhere_polr_med1, 
     patients.to.plot=c("L_0f6c9dd07b06fc"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title = "PI",
     show.cma = FALSE)


plot(adhere_tab_med1, 
     patients.to.plot=c("L_42e1cae8bf1e7f"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title = "TAB",
     show.cma = FALSE)

plot(adhere_forest_med1, 
     patients.to.plot=c("L_0f6c9dd07b06fc"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title  = "GM",
     show.cma = FALSE)

plot(adhere_xgb_med1, 
     patients.to.plot=c("L_0f6c9dd07b06fc"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title  = "XGBoost",
     show.cma = FALSE)

adherence_med_1 %>% filter(ddd != po & tab != rf & tab!=po & po !=rf) %>% View()










