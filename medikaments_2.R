library(MASS)
library(caret)
library(tidyverse)
library(brant)
library(car)
library(effects)
library(randomForest)
library(caret)
require(fmsb)

df_for_modeling <- readRDS("df_for_modeling.rds")

df_for_modeling %>% 
  filter(atc == "C07AB02") %>% 
  count(dienas_deva) %>% 
  arrange(desc(n)) %>% 
  as.data.frame() %>% 
  head(5)

medication_2_df <- df_for_modeling %>% 
  mutate(dienas_deva = round(dienas_deva, 0)) %>% 
  filter(atc == "C07AB02"  & 
           dienas_deva %in% c(23, 100, 50)) %>%
  # mutate(dienas_deva = if_else(dienas_deva == 23,
  #                              25,
  #                              dienas_deva)) %>% 
  mutate(dienas_deva = case_when(dienas_deva == 23 ~ "mg23",
                                 dienas_deva == 50 ~ "mg50",
                                 dienas_deva == 100 ~ "mg100",)) %>% 
  mutate(dienas_deva = as.factor(dienas_deva)) %>% 
  dplyr::select(dzim, vec, grupa_Acute_infarction, grupa_Cancer, grupa_Cerebrovascular , 
                grupa_Congestive_failure, grupa_Connective_disorder, grupa_Dementia, grupa_Diabetes, 
                grupa_Diabetes_complications, grupa_Hemiplegia_paraplegia, grupa_HIV, grupa_Liver, grupa_Metastatic, 
                grupa_Peptic , grupa_Peripheral_disease, grupa_Pulmonary,  grupa_Renal, grupa_Severe_disease, hosp_n, dienas_deva) 



summary(medication_2_df)
medication_2_df <- medication_2_df %>% 
  select_if(function(col) length(unique(col)) > 1) 

summary(medication_2_df)

medication_2_df <- medication_2_df %>% 
  dplyr::select(-grupa_Connective_disorder, -grupa_Hemiplegia_paraplegia, -grupa_Metastatic)


medication_2_df$dienas_deva <- factor(medication_2_df$dienas_deva, ordered = TRUE, levels = c("mg23", "mg50", "mg100")) 

medication_2_df %>% 
  group_by(dienas_deva) %>% 
  summarise(cnt = n()) %>% 
  mutate(prop = cnt/sum(cnt)) %>% as.data.frame()

prop_df <- medication_2_df %>% 
  group_by(dienas_deva) %>% 
  summarise(cnt = n()) %>% 
  mutate(prop = cnt/sum(cnt)) %>% as.data.frame()

#### "dumb" logLoss ####

M <- 3
dumb_log_loss <- -log(1/M)
print(dumb_log_loss)

#### non-informative log loss ####

non_info_log_loss <- -(prop_df$prop[1] * log(prop_df$prop[1]) + prop_df$prop[2] * log(prop_df$prop[2]) + prop_df$prop[3] * log(prop_df$prop[3]))


#### prepare data for training and test set  ####
index_2 <- createDataPartition(medication_2_df$dienas_deva, p = .8, 
                               list = FALSE, 
                               times = 1) %>% as.vector()

### ORDINAL REGRESSION ######

medication_2_ord_df <- medication_2_df
#medication_2_ord_df$dienas_deva <- factor(medication_2_ord_df$dienas_deva, ordered = TRUE, levels = c("mg23", "mg50", "mg100")) 
summary(medication_2_ord_df)
str(medication_2_ord_df)

#### Creating training and test set ####

datatrain_2 <-  medication_2_ord_df[index_2,]
datatest_2 <-  medication_2_ord_df[-index_2,]
summary(datatrain_2)

datatrain_2 %>% 
  group_by(dienas_deva) %>% 
  summarise(cnt = n()) %>% 
  mutate(prop = cnt/sum(cnt)) %>% as.data.frame()

# POLR ####

med_2_polr <- polr(dienas_deva ~ dzim + vec + grupa_Acute_infarction + grupa_Cancer + grupa_Cerebrovascular + grupa_Congestive_failure +
                     grupa_Dementia + grupa_Diabetes + grupa_Diabetes_complications +
                     grupa_Liver + grupa_Peptic + grupa_Peripheral_disease + grupa_Pulmonary +
                     grupa_Renal + hosp_n, data = datatrain_2, Hess = TRUE, method = "logistic")

summary(med_2_polr)
#brant(med_2_polr)
require(stats)
med_2_predict_polr <- predict(med_2_polr, newdata = datatest_2, type = "prob")
require(ModelMetrics)
mlogLoss(actual = datatest_2$dienas_deva , predicted = med_2_predict_polr )

#Compute confusion table and misclassification error
predict_dienas_deva_polr <- predict(med_2_polr, datatest_2)
table(datatest_2$dienas_deva, predict_dienas_deva_polr)
mean(as.character(datatest_2$dienas_deva) != as.character(predict_dienas_deva_polr))
med_2_polr_error <- mean(as.character(datatest_2$dienas_deva) != as.character(predict_dienas_deva_polr))
caret::confusionMatrix(datatest_2$dienas_deva, predict_dienas_deva_polr, mode = "everything")

polr_mod_med_2 <- med_2_polr

### RANDOM FOREST ######

medication_2_forest_df <- medication_2_df
str(medication_2_forest_df)
forest_train_2 <- medication_2_forest_df[index_2,]
forest_valid_2 <- medication_2_forest_df[-index_2,]
summary(forest_train_2)
summary(forest_valid_2)

require(ranger)
#JA palaiž kodu zemāk, tad pēc tam vajag restartet. Kaut kādai funkcijai savdāk pēc tam kaut kas nepatīk 
# train_controlKFCV <- trainControl(method = "cv",
#                                   number =10 ,
#                                   classProbs = TRUE ,
#                                   summaryFunction = mnLogLoss)
# tune.gridranger <- expand.grid(mtry = c(2:10),
#                                splitrule = "gini",
#                                min.node.size = 1)
# 
# train.rf <- train (dienas_deva ~ dzim + vec + grupa_Acute_infarction + grupa_Cerebrovascular +
#                      grupa_Dementia + grupa_Diabetes + grupa_Diabetes_complications +
#                      grupa_Peptic + grupa_Peripheral_disease + hosp_n,
#                       data = forest_train_2 ,
#                       trControl = train_controlKFCV ,
#                       tuneGrid = tune.gridranger ,
#                       method = "ranger" ,
#                       metric = "logLoss")
# print (train.rf)
# require (magicfor)
# magic_for(print, silent = TRUE )
# 
# for (i in seq(from = 100, to = 5000, by = 50)) {
#   set.seed(1234)
#   med_2_rf_trees <- ranger(dienas_deva ~ dzim + vec + grupa_Acute_infarction + grupa_Cerebrovascular +
#                              grupa_Dementia + grupa_Diabetes + grupa_Diabetes_complications +
#                              grupa_Peptic + grupa_Peripheral_disease + hosp_n,
#                            data = forest_train_2,
#                            num.trees = i,
#                            mtry = 7,
#                            splitrule = "gini",
#                            min.node.size = 1,
#                            probability = TRUE)
#   predict_rf_trees <- stats::predict(med_2_rf_trees,
#                                      data = forest_train_2,
#                                      type = "response")
#   print(ModelMetrics::mlogLoss(actual = forest_train_2$dienas_deva,
#                                predicted = predict_rf_trees$prediction))
# }
# 
# require(forcats)
# train.rf.trees <- magic_result_as_dataframe()
# colnames(train.rf.trees)
# 
# names(train.rf.trees)[names(train.rf.trees) == "i"] <- "num.trees"
# names(train.rf.trees)[names(train.rf.trees) == "ModelMetrics::mlogLoss(actual=forest_train_2$dienas_deva,predicted=predict_rf_trees$prediction)"] <- "logLoss"
# 
# view(train.rf.trees)
# require(ggplot2)
# ggplot(data = train.rf.trees, aes(train.rf.trees$num.trees,
#                                   train.rf.trees$logLoss)) + geom_smooth(method = "loess", se = FALSE)


med_2_forest_model <- ranger(dienas_deva ~.,
                             data = forest_train_2,
                             mtry = 7,
                             num.trees = 100,
                             splitrule = "gini",
                             probability = TRUE,
                             importance = "impurity",
                             classification = TRUE)


print(med_2_forest_model)
med_2_forest_model$variable.importance

### predict
require(stats)
predict_forest <- predict(med_2_forest_model, forest_valid_2, type = "response")
require(ModelMetrics)
mlogLoss(actual = forest_valid_2$dienas_deva, predicted = predict_forest$predictions)

med_2_forest_model <- ranger(dienas_deva ~.,
                             data = forest_train_2,
                             mtry = 7,
                             num.trees = 100,
                             splitrule = "gini",
                             probability = FALSE,
                             importance = "impurity",
                             classification = TRUE)
importance(med_2_forest_model)        



#Compute confusion table and misclassification error
predict_deva_forest <- predict(med_2_forest_model, forest_valid_2, type = "response")
table(forest_valid_2$dienas_deva, predict_deva_forest$predictions)
Kappa.test(table(forest_valid_2$dienas_deva, predict_deva_forest$predictions))


forest_mod_med_2 <- med_2_forest_model

#predict_dienas_deva

mean(as.character(forest_valid_2$dienas_deva) != as.character(predict_deva_forest$predictions))
med_2_polr_error_mod <- mean(as.character(forest_valid_2$dienas_deva) != as.character(predict_deva_forest$predictions))
importance(med_2_forest_model)        
caret::confusionMatrix(forest_valid_2$dienas_deva,  predict_deva_forest$predictions, mode = "everything")


#### XGBOOOST #################

require(xgboost)
xgb_medication_2_df <- medication_2_df 

for (i in 1:15) {
  print(i)
  xgb_medication_2_df[,i] <- as.numeric(unlist(xgb_medication_2_df[,i]))
}

dienas_deva  <-  xgb_medication_2_df$dienas_deva
label <-  as.integer(xgb_medication_2_df$dienas_deva)-1
xgb_medication_2_df$dienas_deva <-  NULL

n <-  nrow(xgb_medication_2_df)

train_data <-  as.matrix(xgb_medication_2_df[index_2,])
train_label <-  label[index_2]
test_data <-  as.matrix(xgb_medication_2_df[-index_2,])
test_label <-  label[-index_2]


# Transform the two data sets into xgb.Matrix
xgb_train_med_2 <- xgb.DMatrix(data = train_data, label = train_label)
xgb_test_med_2 <-  xgb.DMatrix(data = test_data, label = test_label)

trctrl <- trainControl(method = "cv", number = 5)

# tune_grid <- expand.grid(nrounds=c(100,200,300,400,500),
#                          max_depth = c(3:7),
#                          eta = c(0.05, 1),
#                          gamma = c(0.01),
#                          colsample_bytree = c(0.7, 0.75),
#                          subsample = c(0.50, 0.60, 0.7),
#                          min_child_weight = c(0))
# 
# rf_fit <- train(dienas_deva ~., data = forest_train_2, method = "xgbTree",
#                 trControl=trctrl,
#                 tuneGrid = tune_grid,
#                 tuneLength = 10)

# Define the parameters for multinomial classification
num_class <-  length(levels(dienas_deva))
params = list(
  booster="gbtree",
  eta=0.1,
  max_depth=6,
  gamma=0.01,
  subsample=0.5,
  colsample_bytree = 0.75,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = num_class
)

# Train the XGBoost classifer
xgb_mod_med_2 <- xgb.train(
  params = params,
  data = xgb_train_med_2,
  nrounds = 500,
  nthreads = 1,
  early_stopping_rounds = 20,
  watchlist = list(val1 = xgb_train_med_2, val2 = xgb_test_med_2),
  verbose = 0
)

# Review the final model and results
xgb_mod_med_2 

# Predict outcomes with the test data
xgb_pred_mod_2 <-  predict(xgb_mod_med_2, test_data, reshape = T)
xgb_pred_mod_2 <-  as.data.frame(xgb_pred_mod_2)
colnames(xgb_pred_mod_2) <- levels(dienas_deva)

#Use the predicted label with the highest probability
xgb_pred_mod_2$prediction <-  apply(xgb_pred_mod_2, 1, function(x) colnames(xgb_pred_mod_2)[which.max(x)])
xgb_pred_mod_2$label <-  levels(dienas_deva)[test_label + 1]


# Calculate the final accuracy
xgb_result  <-  sum(xgb_pred_mod_2$prediction==xgb_pred_mod_2$label)/nrow(xgb_pred_mod_2)
table(xgb_pred_mod_2$label, xgb_pred_mod_2$prediction)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*xgb_result)))
caret::confusionMatrix(as.factor(xgb_pred_mod_2$label), as.factor(xgb_pred_mod_2$prediction), mode = "everything")

require(fmsb)
Kappa.test(table(xgb_pred_mod_2$label, xgb_pred_mod_2$prediction))


#### final 
xgb_mod_med_2



#### ADHERENCE ################

rec_adh_365_all_info_df <- readRDS("rec_adh_365_all_info_df.rds")


one_year_df_all_info <- rec_adh_365_all_info_df %>% 
  filter(atpr_dat < 731) #%>% 
#mutate(fup_end = if_else(!is.na(fup_end),
#                          fup_end,
#                         as.integer(730)))

one_year_df_all_info %>% 
  filter(atc == "C07AB02") %>% 
  group_by(stipr_mg) %>% 
  count()


#### 2 MEDICATION #######

one_year_med_2_df <- one_year_df_all_info %>% 
  filter(atc == "C07AB02")

#### predict polr ####
one_year_med_2_df$pred_deva_polr <- predict(polr_mod_med_2, one_year_med_2_df)

#### predict random forest #####
one_year_med_2_df$pred_deva_forest <- (predict(forest_mod_med_2, one_year_med_2_df, type = "response"))$prediction

#### predict XGB ####

dienas_deva  <-  one_year_med_2_df %>% 
  mutate(dienas_deva = round(dienas_deva,0)) %>% 
  filter(dienas_deva %in% c(23, 50, 100)) %>% mutate(dienas_deva = as.factor(dienas_deva))
dienas_deva <- dienas_deva$dienas_deva

data_for_xgb_med_2 <- one_year_med_2_df %>% 
  dplyr::select(colnames(medication_2_df))

for (i in (1:(ncol(data_for_xgb_med_2)-1))) {
  data_for_xgb_med_2[i] <- as.numeric(unlist(data_for_xgb_med_2[i]))
}

data_for_xgb_med_2$dienas_deva <-  NULL
data_for_xgb_med_2 <- as.matrix(data_for_xgb_med_2)

# Predict outcomes with the test data
xgb_pred_med_2 <-  predict(xgb_mod_med_2, data_for_xgb_med_2, reshape = T)
xgb_pred_med_2 <-  as.data.frame(xgb_pred_med_2)
colnames(xgb_pred_med_2) <- levels(dienas_deva)


#Use the predicted label with the highest probability
xgb_pred_med_2$prediction <-  apply(xgb_pred_med_2, 1, function(x) colnames(xgb_pred_med_2)[which.max(x)])

#############

one_year_med_2_df$pred_deva_xgb <- as.numeric(xgb_pred_med_2$prediction)

### Modify the data  ###############

one_year_med_2_df <- one_year_med_2_df %>% 
  mutate(pred_deva_polr = as.numeric(case_when(pred_deva_polr == "mg100" ~ 100,
                                               pred_deva_polr == "mg50" ~ 50,
                                               pred_deva_polr == "mg23" ~ 23)),
         pred_deva_forest = as.numeric(case_when(pred_deva_forest == "mg100" ~ 100,
                                                 pred_deva_forest == "mg50" ~ 50,
                                                 pred_deva_forest == "mg23" ~ 23))) %>% 
  left_join(df_for_modeling %>% 
              dplyr::select(pid, atpr_dat, atc, dienas_deva, is_good_for_sample) %>% 
              filter (dienas_deva %in% c(23, 50, 100)) %>% 
              dplyr::select(-dienas_deva) , by = c("pid", "atc", "atpr_dat")) 

one_year_med_2_df <- one_year_med_2_df %>% 
  mutate(pred_deva_polr = if_else(!is.na(is_good_for_sample) & !is.na(dienas_deva),
                                  dienas_deva,
                                  pred_deva_polr),
         pred_deva_forest = if_else(!is.na(is_good_for_sample)  & !is.na(dienas_deva),
                                    dienas_deva,
                                    pred_deva_forest),
         pred_deva_xgb = if_else(!is.na(is_good_for_sample)  & !is.na(dienas_deva),
                                 dienas_deva,
                                 pred_deva_xgb))

one_year_med_2_df <- one_year_med_2_df %>% 
  mutate(nosegt_dienas_polr = atpr_mg/pred_deva_polr,
         nosegt_dienas_forest = atpr_mg/pred_deva_forest,
         nosegt_dienas_xgb = atpr_mg/pred_deva_xgb)

# Tas ir nepieciešams funkcijas CMA7() darbībai 
one_year_med_2_df <- one_year_med_2_df %>% 
  mutate(atpr_dat_mod = as.Date("2020-01-01") + atpr_dat)

require(AdhereR)


#df_for_adh_med_2 <- one_year_med_2_df %>% 
#select(pid, atpr_dat_mod, nosegt_dienas_polr) 

adhere_polr_med2 <- CMA7(data = one_year_med_2_df,
                         ID.colname = "pid",
                         event.date.colname = "atpr_dat_mod",
                         event.duration.colname = "nosegt_dienas_polr",
                         #medication.groups = "atc",
                         carry.only.for.same.medication = TRUE,
                         followup.window.start=0, 
                         followup.window.duration = 365,
                         observation.window.start=0, 
                         observation.window.duration=365) 


# ggplot(adhere_polr_med2$CMA, aes(y = CMA)) + 
#   geom_boxplot() 
# boxplot(adhere_polr_med2$CMA$CMA)  

adhere_forest_med2 <- CMA7(data = one_year_med_2_df,
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
# 
# boxplot(adhere_forest_med2$CMA$CMA)  

adhere_xgb_med2 <- CMA7(data = one_year_med_2_df,
                        ID.colname = "pid",
                        event.date.colname = "atpr_dat_mod",
                        event.duration.colname = "nosegt_dienas_xgb",
                        #medication.groups = "atc",
                        carry.only.for.same.medication = TRUE,
                        followup.window.start=0, 
                        followup.window.duration = 365,
                        observation.window.start=0, 
                        observation.window.duration=365) #,

# boxplot(adhere_xgb_med2$CMA$CMA)  


adhere_ddd_med2 <- CMA7(data = one_year_med_2_df,
                        ID.colname = "pid",
                        event.date.colname = "atpr_dat_mod",
                        event.duration.colname = "nosegt_dienas_ddd",
                        #medication.groups = "atc",
                        carry.only.for.same.medication = TRUE,
                        followup.window.start=0, 
                        followup.window.duration = 365,
                        observation.window.start=0, 
                        observation.window.duration=365) #,
# 
# boxplot(adhere_ddd_med2$CMA$CMA)  

adhere_tab_med2 <- CMA7(data = one_year_med_2_df,
                        ID.colname = "pid",
                        event.date.colname = "atpr_dat_mod",
                        event.duration.colname = "nosegt_dienas_tab",
                        #medication.groups = "atc",
                        carry.only.for.same.medication = TRUE,
                        followup.window.start=0, 
                        followup.window.duration = 365,
                        observation.window.start=0, 
                        observation.window.duration=365) #,

# boxplot(adhere_tab_med2$CMA$CMA)  

adherence_med_2 <- adhere_ddd_med2$CMA %>% 
  rename(DDD = CMA) %>% 
  left_join(adhere_tab_med2$CMA %>% 
              rename(TAB = CMA), by = "pid") %>% 
  left_join(adhere_polr_med2$CMA %>% 
              rename(PI = CMA), by = "pid") %>% 
  left_join(adhere_forest_med2$CMA %>% 
              rename(GM = CMA), by = "pid") %>% 
  left_join(adhere_xgb_med2$CMA %>% 
              rename(XGB = CMA), by = "pid") 

saveRDS(adherence_med_2, "adherence_med_2.rds")

adherence_med_2_for_plot <- adherence_med_2 %>% 
  pivot_longer(!pid, names_to = "metode", values_to = "adherence")

adherence_med_2_for_plot %>% 
  group_by(metode) %>% 
  summarise(mediana = median(adherence),
            iqr = IQR(adherence))

adh_p2 <- ggplot(adherence_med_2_for_plot, aes(y = adherence, x = metode, fill = metode)) +
  geom_boxplot() +
  theme(
    plot.title = element_text(size = 30, face = "bold"),
    text = element_text(size=25),
    legend.position="none"
  ) +
  ggtitle("Metoprolols (C07AB02)", ) +
  xlab("") +
  font("xy.text", size = 25) +
  ylab("zāļu līdzestība") +
  scale_y_continuous(labels = scales::percent) 

one_year_med_2_df %>% 
  group_by(pred_deva_xgb) %>% 
  count()


# ggsave(adh_p2, filename = "New folder/bildes/adh_med_2.png",
#        height = 9,
#        width = 16
# )

plot(adhere_ddd_med2, 
     patients.to.plot=c("L_c440cc8c8ea6b8"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title = "DDD",
     print.dose=TRUE, plot.dose=TRUE,
     show.cma = FALSE)

plot(adhere_polr_med2, 
     patients.to.plot=c("L_c440cc8c8ea6b8"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title = "PI",
     show.cma = FALSE)


plot(adhere_tab_med2, 
     patients.to.plot=c("L_c440cc8c8ea6b8"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title = "TAB",
     show.cma = FALSE)

plot(adhere_forest_med2,
     patients.to.plot=c("L_c440cc8c8ea6b8"), # plot only patient 76
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība (24,7%)",
     title  = "GM",
     show.cma = FALSE)

plot(adhere_xgb_med2, 
     patients.to.plot=c("L_c440cc8c8ea6b8"), # plot only patient 76 
     #legend.x=260); # place the legend in a nice way
     show.legend=FALSE,
     xlab = "dienas",
     ylab = "pacienta id un zāļu līdzestība",
     title  = "XGBoost (74,0%)",
     show.cma = FALSE)


