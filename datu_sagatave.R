library(tidyverse)
Sys.setlocale("LC_ALL", "Latvian")
library(ggpubr)
### funkcija, lai noteiktu, vai pacientam bija konkrēta slimīna pirms zāļu atprečošanas 

find_disease_before_the_event <- function(patient, day){
  
  disease_df_mod <- disease_df %>% 
    dplyr::select(-diag_pamat, -diag_papild) %>% 
    filter(pid == patient)
  
  x <- disease_df_mod %>% 
    filter(hosp_dat <= day)
  
  if (nrow(x) > 0) {
    df <- x %>% 
      group_by(pid) %>% 
      summarise(across(everything(), max), .groups = "drop")
  } else {
    df <- disease_df_mod %>% 
      arrange(hosp_dat) %>% 
      slice(1)
    df[,5:21] <- rep(0)
  }
  return(df)
}

### nepieciešamu datu ielāde 

load("dati/rec.RData")
load("dati/rec_adh.RData") 
load("dati/stac.RData")
load("dati/charlson.RData")


####  Slīmības tabulas veidošana  #######

stac_mod <- stac %>% 
  mutate(diagnoses = paste0(diag_papild, ";", diag_pamat)) %>% 
  mutate(diagnoses_no_dupl = sapply(strsplit(diagnoses, ";"), function(x) paste(unique(x), collapse = ";"))) 

stac_sep_diag <- stac_mod %>% 
  dplyr::select(-diagnoses) %>% 
  separate(diagnoses_no_dupl, paste0("diag_", 1:19),  ";") #one patient has 19 papild_diag, not good line of code 

stac_sep_diag <- stac_sep_diag %>% 
  pivot_longer(!(1:6), names_to = "papildus_diag", values_to = "kods",  values_drop_na = TRUE) %>% 
  left_join(charlson, by = "kods") %>% 
  fastDummies::dummy_cols(select_columns = "grupa") 

disease_df <- stac_sep_diag %>% #table with patient and his disaeses when he was hospitilized 
  dplyr::select(-grupa_NA) %>% 
  filter(!is.na(grupa)) %>% 
  dplyr::select(-kods, -grupa, -papildus_diag) %>% 
  group_by(pid, hosp_dat, diag_pamat, diag_papild, vec, dzim, ) %>% 
  summarise(across(everything(), max), .groups = "drop")

### tabulu apvienošana 

x <- colnames(disease_df)
new_df_colnames <- x[startsWith(x, "grupa_")]
new_df <- as.data.frame(matrix(numeric(),nrow = 0, ncol = length(new_df_colnames)))
colnames(new_df) <- new_df_colnames 
new_df <- new_df%>% 
  mutate(pid = NA_character_,
         atrp_dat = NA_integer_) %>% 
  dplyr::select(pid, atrp_dat, everything())  

rec_adh_365 <- rec_adh %>% 
  filter(atpr_dat > 364) %>% 
  dplyr::select(pid, atpr_dat) %>% 
  group_by(pid, atpr_dat) %>% 
  summarise()

## sis cikls aiznem loti daudz laika, tapec tas tika iekomentets. rezultata tika ieguta tabula new_df
# for (i in seq_len(nrow(rec_adh_365))){
#   patient <- rec_adh_365$pid[i]
#   day <- rec_adh_365$atpr_dat[i]
#   df <- find_disease_before_the_event(patient = patient, day = day) %>%
#     mutate(atpr_dat = day) %>%
#     dplyr::select(pid, atpr_dat, everything())
#   new_df <- rbind(new_df, df)
#   print(i)
# }

#saveRDS(new_df, "new_df.rds")
new_df <- readRDS("new_df.rds")
names(new_df) <- gsub(" ", "_", names(new_df))

####################### Atrast dienu skaitu starp atprecosanas datumiem ########################

rec_adh_mod_df <- rec_adh %>%
  arrange(desc(atpr_dat)) %>%
  group_by(pid, atc) %>%
  mutate(days_interval = lag(atpr_dat) -atpr_dat)

####Atrast regulāras dienas devas ####

regular_df <- rec_adh_mod_df %>% 
  mutate(target_day  = case_when(days_interval %in% (28:32) ~ 30,
                                 days_interval %in% (58:62) ~ 60,
                                 TRUE ~ NA_real_)) %>% 
  ungroup()


regular_df <- regular_df %>% 
  mutate(dienas_deva = atpr_mg/target_day) %>% 
  filter(atpr_dat > 364)

regular_df %>%
  filter(!is.na(dienas_deva)) %>% 
  count(dienas_deva, atc) %>% 
  arrange(atc, desc(n)) 


######### days distribution graph ###############
library("gridExtra")
p1 <- regular_df %>% 
  filter(atc == "C10AA05" & days_interval < 100) %>% 
  ggplot(aes(x = days_interval))  + 
  geom_histogram() +
  xlab("dienu skaits starp zāļu atprečošanu") +
  ylab("atprečoto recepšu skaits") + 
  ggtitle("Atorvastatīns (C10AA05)") +
  theme(text = element_text(size=20))

p2 <- regular_df %>% 
  filter(atc == "C07AB02" & days_interval < 100) %>% 
  ggplot(aes(x = days_interval))  + 
  geom_histogram() +
  xlab("dienu skaits starp zāļu atprečošanu") +
  ylab("atprečoto recepšu skaits") + 
  ggtitle("Metoprolols (C07AB02)") +
  theme(text = element_text(size=20))

p3 <- regular_df %>% 
  filter(atc == "C07AB07" & days_interval < 100) %>% 
  ggplot(aes(x = days_interval))  + 
  geom_histogram() +
  xlab("dienu skaits starp zāļu atprečošanu") +
  ylab("atprečoto recepšu skaits") + 
  ggtitle("Bisoprolols (C07AB07)") +
  theme(text = element_text(size=20))

grid.arrange(p1, p2, p3 , 
          ncol = 3, nrow = 1)



#### Jaunais mainīgais hospitalizēšanas skaits #############

# y_df <- data.frame(pid = character(0),
#                    atpr_dat = integer(0),
#                    hosp_n = integer(0))
# pid_atpr <- regular_df %>% 
#   group_by(pid, atpr_dat) %>% 
#   summarise()
# 
# for (i in seq_len(nrow(regular_df))){
#   pid_i <- pid_atpr[i, ]$pid
#   atpr_i <- pid_atpr[i, ]$atpr_dat
#   n_hosp <- stac %>% 
#     filter(pid == pid_i & hosp_dat <= atpr_i) %>% 
#     count()
#   df <- data.frame(pid = pid_i,
#                         atpr_dat = atpr_i,
#                         hosp_n = n_hosp$n)
#   y_df <- rbind(y_df, df)
#   print(i)
# }

y_df <- readRDS("y_df.rds")

rec_adh_365_all_info_df <- regular_df %>%
  left_join(y_df, by = c("pid", "atpr_dat")) %>% 
  left_join(new_df, by = c("pid", "atpr_dat")) %>% 
  mutate(dzim = case_when(dzim == "sieviete" ~ "0",
                          TRUE ~ "1")) %>% 
  mutate(dzim = as.factor(dzim))

cats <- rec_adh_365_all_info_df %>% 
  dplyr::select(starts_with("grupa_")) %>% 
  colnames()

rec_adh_365_all_info_df[ ,cats] <- lapply(rec_adh_365_all_info_df[, cats], as.factor)

summary(rec_adh_365_all_info_df) 

saveRDS(rec_adh_365_all_info_df, "rec_adh_365_all_info_df.rds")


#### atrast regulārass dienas devas, uz kurām tiks konstruēts modelis #####

df_with_lag <- rec_adh_365_all_info_df %>% 
  filter(!is.na(dienas_deva)) %>% 
  group_by(pid, atc) %>% 
  arrange(atpr_dat) %>% 
  mutate(lag_1 = lag(dienas_deva),
         lag_2 = lag(dienas_deva, n = 2)
  )

df_with_lag <- df_with_lag %>% 
  mutate(is_good_for_sample = case_when(dienas_deva == lag_1 & dienas_deva == lag_2 ~ 1,
                                        TRUE ~ 0)) 

df_for_modeling <- df_with_lag %>% 
  filter(is_good_for_sample == 1) %>% ungroup()
  
saveRDS(df_for_modeling, "df_for_modeling.rds")
