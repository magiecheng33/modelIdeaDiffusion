######################################################################
######################################################################
####R code for "Modeling the Growth of Scientific Concepts" ##########
####R code for final analysis ########################################
####Correspondence:###################################################
####Daniel A. McFarland###############################################
####e-mail: mcfarland@stanford.edu####################################
######################################################################
######################################################################


##########################################
### A. Preparation for final analysis#####
##########################################
# Set the work directy that is used for reading/saving data tables
setwd("/dfs/scratch0/hanchcao/magiecheng/")

# Load R required packages
# If this is done for the first time, it might need to first download and install the package
install.packages("lme4", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("dplyr", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("tidyverse", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("haven", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("standardize", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("broom", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("data.table", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("readr", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("tzdb", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("vroom", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("withr", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("cli", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("dplyr", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("tidyr", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("devtools", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("DescTools", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("stringr", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("pastecs", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("ggplot2", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("labeling", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("digest", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("regclass", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("bestglm", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("leaps", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("RColorBrewer", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("corrplot", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("RCurl", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("car", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("optimx", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("minqa", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("tidyverse", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
install.packages("olsrr", lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")



library(RCurl, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(crayon, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(dplyr, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(tidyverse, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(haven, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(standardize, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(data.table, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(tzdb, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(vroom, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(withr, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(readr, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(cli, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(dplyr, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(tidyr, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(DescTools, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(stringr, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(pastecs, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(ggplot2, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(labeling, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(digest, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(leaps, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(bestglm, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(regclass, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(RColorBrewer, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(corrplot, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(car, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(tidyverse, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(lme4,lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(optimx,lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(minqa,lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")
library(parallel)
library(olsrr,lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage")


d <- fread("/dfs/scratch0/hanchcao/data/isi_random/final_regression_table_loose_burnin1993_sharing.csv",
           select = c("id",
                      "age",
                      "df",
                      "df_year",
                      "term_start_year",
                      "no_usage", 
                      "pub_num", 
                      "avg_venue_citation_cur", 
                      "avg_entropy_cur", 
                      "avg_entropy_s_cur", 
                      "new_neighbor_wgt_pct_cur",
                      "new_neighbor_avg_pct_cur", 
                      "new_author_pct_avg_cur", 
                      "new_author_pct_wgt_cur", 
                      "avg_author_size_cur", 
                      "avg_author_pub_num_cur", 
                      "author_female_cur", 
                      "avg_neighbor_pop_cur", 
                      "avg_physics_pct_cur",
                      "avg_engineering_pct_cur", 
                      "avg_agriculture_pct_cur", 
                      "avg_bio_pct_cur", 
                      "avg_ss_pct_cur", 
                      "author_density_cur",
                      "ecological_cur",
                      "positivity_cur",
                      "negativity_cur",
                      "readability_cur",
                      "wordcnt_cur", 
                      "venue_abstract_history_cur", 
                      "new_venue_pct_cur", 
                      "lagging_venue_abstract_his_cur",
                      "impact_factor_cur", 
                      "inward_citation_cur",
                      "avg_author_ruse_preceding_1", 
                      "neighbor_ruse_preceding_1",
                     "word_length",
                     "word_char_length",
                      "son_concept_num",
                     "concept_age"),
           showProgress = FALSE)
nrow(d) ##1146976 
d <- as_tibble(d) %>%
  arrange(id, age)
head(d)

###data cleaning
my_x <- d %>%
  select("pub_num", 
         "no_usage",
         contains("ruse_preceding"), 
         ends_with("_cur")) %>%
  names()

my_vars <- c("df_log", "df_dif",  my_x)

###replace all the -1234 vslue as na
d <- d %>% 
  group_by(id) %>%
  mutate(df_log = log(df+1),
         df_log_lead = lead(df_log),
         df_lead = lead(df),
         df_dif = df_log_lead - df_log,
         ecological_cur = na_if(ecological_cur,  -1234),
         impact_factor_cur = na_if(impact_factor_cur,  -1234),
         inward_citation_cur = na_if(inward_citation_cur, -1234),
         venue_abstract_history_cur = na_if(venue_abstract_history_cur, -1234),
         readability_cur = na_if(readability_cur, -1),
         wordcnt_cur = na_if(wordcnt_cur, -1),
         new_venue_pct_cur = na_if(new_venue_pct_cur, -1),
         lagging_venue_abstract_his_cur = na_if(lagging_venue_abstract_his_cur, -1),
         avg_author_ruse_preceding_1 = replace_na(avg_author_ruse_preceding_1, 0),
         author_density_cur = as.numeric(author_density_cur),
         author_density_cur = na_if(author_density_cur, -1234),
         neighbor_ruse_preceding_1  = replace_na(neighbor_ruse_preceding_1, 0)) %>%
  ungroup() 

nrow(d) #1146976
nrow(na.omit(d)) #995945 (entries without any na value is 995945)


d$son_concept_num_log = log(d$son_concept_num + 1)
d$avg_author_size_cur_log = log(d$avg_author_size_cur + 1)
d$avg_author_pub_num_cur_log = log(d$avg_author_pub_num_cur + 1)
d$avg_neighbor_pop_cur_log = log(d$avg_neighbor_pop_cur + 1)
d$avg_venue_citation_cur_log = log(d$avg_venue_citation_cur + 1)
d$author_density_cur_log = log(d$author_density_cur + 1)
d$impact_factor_cur_log = log(d$impact_factor_cur + 1)
d$readability_cur_log = log(d$readability_cur + 1)
d$wordcnt_cur_log = log(d$wordcnt_cur + 1)
d$avg_author_ruse_preceding_1_log = log(d$avg_author_ruse_preceding_1 + 1)
d$neighbor_ruse_preceding_1_log = log(d$neighbor_ruse_preceding_1 + 1)
d$new_neighbor_wgt_pct_cur_log = log(d$new_neighbor_wgt_pct_cur + 1)
d$new_neighbor_avg_pct_cur_log = log(d$new_neighbor_avg_pct_cur + 1)


m_avg_author_pub_num <- mean(d$avg_author_pub_num_cur_log, na.rm = T)
sd_avg_author_pub_num <- sd(d$avg_author_pub_num_cur_log, na.rm = T)
msd1_avg_author_pub_num <- m_avg_author_pub_num + (3*sd_avg_author_pub_num)

m_avg_author_ruse_preceding_1 <- mean(d$avg_author_ruse_preceding_1_log, na.rm = T)
sd_avg_author_ruse_preceding_1 <- sd(d$avg_author_ruse_preceding_1_log, na.rm = T)
msd1_avg_author_ruse_preceding_1 <- m_avg_author_ruse_preceding_1 + (3*sd_avg_author_ruse_preceding_1)

m_neighbor_ruse_preceding_1 <- mean(d$neighbor_ruse_preceding_1_log, na.rm = T)
sd_neighbor_ruse_preceding_1 <- sd(d$neighbor_ruse_preceding_1_log, na.rm = T)
msd1_neighbor_ruse_preceding_1 <- m_neighbor_ruse_preceding_1 + (3*sd_neighbor_ruse_preceding_1)

m_new_neighbor_wgt_pct <- mean(d$new_neighbor_wgt_pct_cur_log, na.rm = T)
sd_new_neighbor_wgt_pct <- sd(d$new_neighbor_wgt_pct_cur_log, na.rm = T)
msd_new_neighbor_wgt_pct <- m_new_neighbor_wgt_pct + (3*sd_new_neighbor_wgt_pct)

m_new_neighbor_avg_pct <- mean(d$new_neighbor_avg_pct_cur_log, na.rm = T)
sd_new_neighbor_avg_pct <- sd(d$new_neighbor_avg_pct_cur_log, na.rm = T)
msd_new_neighbor_avg_pct <- m_new_neighbor_avg_pct + (3*sd_new_neighbor_avg_pct)

d <- d %>% 
  mutate(avg_author_pub_num_cur_log = 
           if_else(avg_author_pub_num_cur_log
                   > msd1_avg_author_pub_num, 
                   msd1_avg_author_pub_num, avg_author_pub_num_cur_log),
         avg_author_ruse_preceding_1_log = 
           if_else(avg_author_ruse_preceding_1_log
                   > msd1_avg_author_ruse_preceding_1, 
                   msd1_avg_author_ruse_preceding_1, avg_author_ruse_preceding_1_log),
         neighbor_ruse_preceding_1_log = 
           if_else(neighbor_ruse_preceding_1_log
                   > msd1_neighbor_ruse_preceding_1, 
                   msd1_neighbor_ruse_preceding_1, neighbor_ruse_preceding_1_log),
         new_neighbor_wgt_pct_cur_log = 
           if_else(new_neighbor_wgt_pct_cur_log
                   > msd_new_neighbor_wgt_pct, 
                   msd_new_neighbor_wgt_pct, new_neighbor_wgt_pct_cur_log),
         new_neighbor_avg_pct_cur_log = 
           if_else(new_neighbor_avg_pct_cur_log
                   > msd_new_neighbor_avg_pct, 
                   msd_new_neighbor_avg_pct, new_neighbor_avg_pct_cur_log)) 

d$age_2 = d$age*d$age

##############################
###record summary stats #log
myx <- c("age", "age_2",
          "author_density_cur",
          "ecological_cur",
          "avg_author_pub_num_cur_log", 
          "avg_neighbor_pop_cur_log",
          "avg_author_ruse_preceding_1_log",
          "neighbor_ruse_preceding_1_log",
          "impact_factor_cur_log",
          "inward_citation_cur",
          "venue_abstract_history_cur",
          "avg_entropy_s_cur",
          "avg_engineering_pct_cur",
          "avg_physics_pct_cur",
          "avg_agriculture_pct_cur",
          "avg_ss_pct_cur",
          "wordcnt_cur_log",
          "readability_cur_log",
          "pub_num",
          "author_female_cur",
          "no_usage",
          "positivity_cur",
          "negativity_cur",
          "term_start_year",
           "son_concept_num_log")

          
my_vars <- c("df_lead", my_x)

tmp <- d %>% 
  select(all_of(my_vars)) %>% drop_na() %>%
  stat.desc() %>%
  t() %>%
  as_tibble() %>%
  mutate(variable = my_vars) %>%
  select(variable, nbr.val, min, max, mean, std.dev)

write.table(tmp , file = "sum_stats_loose_burnin1993_log.txt")

##############################
###record summary stats #raw
my_x <- c("age", "age_2",
          "author_density_cur",
          "ecological_cur",
          "avg_author_pub_num_cur", 
          "avg_neighbor_pop_cur",
          "avg_author_ruse_preceding_1",
          "neighbor_ruse_preceding_1",
          "impact_factor_cur",
          "inward_citation_cur",
          "venue_abstract_history_cur",
          "avg_entropy_s_cur",
          "avg_engineering_pct_cur",
          "avg_physics_pct_cur",
          "avg_agriculture_pct_cur",
          "avg_ss_pct_cur",
          "wordcnt_cur",
          "readability_cur",
          "pub_num",
          "author_female_cur",
          "no_usage",
          "positivity_cur",
          "negativity_cur",
          "term_start_year",
         "son_concept_num")

my_vars <- c("df_lead", my_x)

tmp <- d %>% 
  select(all_of(my_vars)) %>% drop_na() %>%
  stat.desc() %>%
  t() %>%
  as_tibble() %>%
  mutate(variable = my_vars) %>%
  select(variable, nbr.val, min, max, mean, std.dev)


write.table(tmp , file = "sum_stats_loose_burnin1993_raw.txt")


##############################
###correlation of the variables
my_x <- c("avg_author_pub_num_cur_log",
          "author_density_cur",
          "avg_author_ruse_preceding_1_log",
          "avg_neighbor_pop_cur_log",
          "ecological_cur",
          "neighbor_ruse_preceding_1_log",
          "son_concept_num_log",
          "avg_entropy_s_cur",
          "avg_bio_pct_cur",
          "avg_physics_pct_cur",
          "avg_engineering_pct_cur",
          "avg_agriculture_pct_cur",
          "avg_ss_pct_cur",
          "author_female_cur",
          "wordcnt_cur_log",
          "readability_cur_log",
          "positivity_cur",
          "negativity_cur",
          "impact_factor_cur_log",
           "age", "age_2", 
          "term_start_year","venue_abstract_history_cur","inward_citation_cur")
          
my_vars <- c("df_lead", my_x)


mycor <- d %>% 
  select(df_lead, all_of(my_vars)) %>%
  cor(use = "pairwise.complete.obs")


short_names <- c("Term document frequency", "Social prominence",
          "Social embeddedness",
          "Social consistency",
          "Ideational prominence",
          "Ideational embeddedness",
          "Ideational consistency",
           "Root term",
          "Interdisciplinary",
          "Biological and health sciences",
          "Physical sciences and mathematics",
          "Engineering",
          "Agricultural sciences",
          "Humanities and social sciences",
          "Female author usage",
          "Abstract-length",
          "Abstract-readability",
          "Abstract-positivity",
          "Abstract-negativity",
          "Journal impact factor",
            "Age", "Age_2", "Start year", "Journal abstract history","WoS inward citation")         


rownames(mycor) <- short_names
colnames(mycor) <- short_names

pdf(file = "corrplot_loose_burnin1993.pdf")
corrplot(mycor,
         method = "shade", 
         type = "lower",
         tl.col = "black",
         tl.cex = 0.75,
         tl.srt = 45)
dev.off()


##############################
###standardize variables
d$ecological_distance = 1-d$ecological_cur
tobescaled <- c(
                "df_log_lead",
                "df_log",
                "age", "age_2",
                "author_density_cur",
                "ecological_cur",
                "author_density_cur_log",
                "avg_author_pub_num_cur_log",
                "avg_neighbor_pop_cur_log",
                "avg_author_ruse_preceding_1_log",
                "neighbor_ruse_preceding_1_log",
                "new_author_pct_wgt_cur",
                "new_neighbor_wgt_pct_cur",
                "impact_factor_cur_log",
                "inward_citation_cur",
                "venue_abstract_history_cur",
                "avg_entropy_s_cur",
                "avg_engineering_pct_cur",
                "avg_physics_pct_cur",
                "avg_agriculture_pct_cur",
                "avg_bio_pct_cur",
                "avg_ss_pct_cur",
                "wordcnt_cur_log",
                "readability_cur_log",
                "pub_num",
                "term_start_year",
                "author_female_cur",
                "positivity_cur",
                "negativity_cur",
                "df_lead",
                "ecological_distance",
                "word_length",
                "word_char_length","son_concept_num_log")


lmer_data <- d %>%
  select(id, no_usage, all_of(tobescaled)) %>%
  mutate(age_original = age,
         df_lead_original = df_lead,
         age_original_2 = age*age,
         ideationa_orignal = ecological_cur,
         author_orginal = author_density_cur,
         df_log_lead_original = df_log_lead, 
         across(all_of(tobescaled), scale),
         id_num = as.numeric(as.factor(id))) %>%
  na.omit()

glimpse(lmer_data)
summary(lmer_data)
write.csv(lmer_data,"standardized_value_loose_burnin1993.csv", row.names = FALSE)

nrow(lmer_data) #995945
nrow(na.omit(lmer_data)) #995945 

#############################################
### B. Regressions for our Main Analysis#####
#############################################
#########################################################################
#####Multilevel models: df_lead_original (overdispersed-Poisson)#########

# Use over-dispersed function to get over-dispersed Poisson GLMM result
overdisp_fun <- function(model) {
  rdf <- df.residual(model)
  rp <- residuals(model,type="pearson")
  Pearson.chisq <- sum(rp^2)
  prat <- Pearson.chisq/rdf
  pval <- pchisq(Pearson.chisq, df=rdf, lower.tail=FALSE)
  c(chisq=Pearson.chisq,ratio=prat,rdf=rdf,p=pval)
}


## Model 1: Model with only intercept + age + age^2 + term start year, dependent variable df_lead use original and we didn't standardize it as Poisson, DV needs to be positive count variable.
m1_func <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year +
                (1 | id), 
           data = lmer_data, family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m1 <- parLapply(cls, 1, m1_func))
sink("poisson_result_burnin1993.txt")
summary(m1[[1]])
sink()


cc <- coef(summary(m1[[1]]))
phi <- overdisp_fun(m1[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)

## Model 2: Model with intercept + age + age^2 + term start year + controls
m2_func_controls <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year + 
               impact_factor_cur_log +
               inward_citation_cur +
               venue_abstract_history_cur +
               avg_entropy_s_cur +
               avg_engineering_pct_cur +
               avg_physics_pct_cur +
               avg_agriculture_pct_cur +
               avg_ss_pct_cur + 
               wordcnt_cur_log +
               readability_cur_log +
               author_female_cur +
               positivity_cur + negativity_cur + son_concept_num_log + (1 | id), 
           data = lmer_data, family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m2controls <- parLapply(cls, 1, m2_func_controls))
sink("intercept_age_age2_startyear_controls_burnin1993.txt")
summary(m2controls[[1]])
sink()            

cc <- coef(summary(m2controls[[1]]))
phi <- overdisp_fun(m2controls[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)


## Model 3: Model with intercept + age + age^2 + term start year + key independent variables
m3_func_keyindvars <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year +
               author_density_cur + 
               ecological_cur +
               avg_author_pub_num_cur_log +
               avg_neighbor_pop_cur_log + 
               avg_author_ruse_preceding_1_log +
               neighbor_ruse_preceding_1_log +
                (1 | id), 
           data = lmer_data, family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m3keyindvars <- parLapply(cls, 1, m3_func_keyindvars))
sink("intercept_age_age2_startyear_indvar_burnin1993.txt")
summary(m3keyind[[1]])
sink()

cc <- coef(summary(m3keyindvars[[1]]))
phi <- overdisp_fun(m3keyindvars[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)


## Model 4: Model with intercept + age + age^2 + term start year + key independent variables + controls
m4_func_all <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year +
               author_density_cur + 
               ecological_cur + 
               avg_author_pub_num_cur_log +
               avg_neighbor_pop_cur_log + 
               avg_author_ruse_preceding_1_log +
               neighbor_ruse_preceding_1_log +
               impact_factor_cur_log +
               inward_citation_cur +
               venue_abstract_history_cur +
               avg_entropy_s_cur +
               avg_engineering_pct_cur +
               avg_physics_pct_cur +
               avg_agriculture_pct_cur +
               avg_ss_pct_cur + 
               wordcnt_cur_log +
               readability_cur_log + 
               author_female_cur +
               positivity_cur + negativity_cur + son_concept_num_log  + (1 | id), data = lmer_data, family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m4all <- parLapply(cls, 1, m4_func_all))
sink("intercept_all_burnin1993.txt")          
summary(m4all[[1]])
sink()

cc <- coef(summary(m4all[[1]]))
phi <- overdisp_fun(m4all[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)

#########################################################################
###compute variance explained
# Model with only intercept
m1_intercept <- glmer(df_lead_original ~ 1 + (1 | id), 
           data = lmer_data,
           family = poisson)
# variance explained for model 1
VarF <- var(as.vector(fixef(m1[[1]]) %*% t(m1[[1]]@pp$X)))
VarF/(VarF + VarCorr(m1[[1]])$id[1] + log(1 + 1/exp(as.numeric(fixef(m1_intercept)))))

# variance explained for model 2
VarF <- var(as.vector(fixef(m2controls[[1]]) %*% t(m2controls[[1]]@pp$X)))
VarF/(VarF + VarCorr(m2controls[[1]])$id[1] + log(1 + 1/exp(as.numeric(fixef(m1_intercept)))))

# variance explained for model 3
VarF <- var(as.vector(fixef(m3keyind[[1]]) %*% t(m3keyind[[1]]@pp$X)))
VarF/(VarF + VarCorr(m3keyind[[1]])$id[1] + log(1 + 1/exp(as.numeric(fixef(m1_intercept)))))

# variance explained for model 4
VarF <- var(as.vector(fixef(m4all[[1]]) %*% t(m4all[[1]]@pp$X)))
VarF/(VarF + VarCorr(m4all[[1]])$id[1] + log(1 + 1/exp(as.numeric(fixef(m1_intercept)))))


          
##########################################################################
### C. Interaction of Social and Ideational Conditions with Term Age #####
##########################################################################

m4_func_author_density_cur <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year +
               author_density_cur + 
               I(author_density_cur*age) + 
               I(author_density_cur*age_2) + 
               ecological_cur +
               avg_author_pub_num_cur_log +
               avg_neighbor_pop_cur_log + 
               avg_author_ruse_preceding_1_log +
               neighbor_ruse_preceding_1_log +
               impact_factor_cur_log +
               inward_citation_cur +
               venue_abstract_history_cur +
               avg_entropy_s_cur +
               avg_engineering_pct_cur +
               avg_physics_pct_cur +
               avg_agriculture_pct_cur +
               avg_ss_pct_cur + 
               wordcnt_cur_log +
               readability_cur_log +
               author_female_cur +
               positivity_cur + negativity_cur + son_concept_num_log +
               (1 + age + age_2 | id),
           data = lmer_data,family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m4_author_density_cur <- parLapply(cls, 1, m4_func_author_density_cur))
sink("intercept_burnin1993_author_density_interaction.txt")
summary(m4_author_density_cur[[1]])
sink()

cc <- coef(summary(m4_author_density_cur[[1]]))
phi <- overdisp_fun(m4_author_density_cur[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)


m4_func_ecological_cur <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year +
               author_density_cur + 
               ecological_cur + 
               I(ecological_cur*age) +
               I(ecological_cur*age_2) +
               avg_author_pub_num_cur_log +
               avg_neighbor_pop_cur_log + 
               avg_author_ruse_preceding_1_log +
               neighbor_ruse_preceding_1_log +
               impact_factor_cur_log +
               inward_citation_cur +
               venue_abstract_history_cur +
               avg_entropy_s_cur +
               avg_engineering_pct_cur +
               avg_physics_pct_cur +
               avg_agriculture_pct_cur +
               avg_ss_pct_cur + 
               wordcnt_cur_log +
               readability_cur_log +
               author_female_cur +
               positivity_cur + negativity_cur + son_concept_num_log +
               (1 + age + age_2 | id),
           data = lmer_data,family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m4_ecological_cur <- parLapply(cls, 1, m4_func_ecological_cur))
sink("intercept_burnin1993_ecological_interaction.txt")
summary(m4_ecological_cur[[1]])
sink()

cc <- coef(summary(m4_ecological_cur[[1]]))
phi <- overdisp_fun(m4_ecological_cur[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)


m4_func_author_pub_cur <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year +
               author_density_cur + 
               ecological_cur +
               avg_author_pub_num_cur_log +
               I(avg_author_pub_num_cur_log*age) +
               I(avg_author_pub_num_cur_log*age_2) +
               avg_neighbor_pop_cur_log + 
               avg_author_ruse_preceding_1_log +
               neighbor_ruse_preceding_1_log +
               impact_factor_cur_log +
               inward_citation_cur +
               venue_abstract_history_cur +
               avg_entropy_s_cur +
               avg_engineering_pct_cur +
               avg_physics_pct_cur +
               avg_agriculture_pct_cur +
               avg_ss_pct_cur + 
               wordcnt_cur_log +
               readability_cur_log + 
               author_female_cur +
               positivity_cur + negativity_cur + son_concept_num_log +
               (1 + age + age_2 | id), data = lmer_data,family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m4_author_pub_cur <- parLapply(cls, 1, m4_func_author_pub_cur))
sink("intercept_burnin1993_author_pubnum_interaction.txt")
summary(m4_author_pub_cur[[1]])
sink()

cc <- coef(summary(m4_author_pub_cur[[1]]))
phi <- overdisp_fun(m4_author_pub_cur[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)

m4_func_neighbor_pop_cur <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year +
               author_density_cur + 
               ecological_cur +
               avg_author_pub_num_cur_log +
               avg_neighbor_pop_cur_log + 
               I(avg_neighbor_pop_cur_log*age) +
               I(avg_neighbor_pop_cur_log*age_2) +
               avg_author_ruse_preceding_1_log +
               neighbor_ruse_preceding_1_log +
               impact_factor_cur_log +
               inward_citation_cur +
               venue_abstract_history_cur +
               avg_entropy_s_cur +
               avg_engineering_pct_cur +
               avg_physics_pct_cur +
               avg_agriculture_pct_cur +
               avg_ss_pct_cur + 
               wordcnt_cur_log +
               readability_cur_log +
               author_female_cur +
               positivity_cur + negativity_cur + son_concept_num_log +
               (1 + age + age_2 | id),
           data = lmer_data,family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m4_neighbor_pop_cur <- parLapply(cls, 1, m4_func_neighbor_pop_cur))
sink("intercept_burnin1993_neighbor_pop_interaction.txt")
summary(m4_neighbor_pop_cur[[1]])
sink()

cc <- coef(summary(m4_neighbor_pop_cur[[1]]))
phi <- overdisp_fun(m4_neighbor_pop_cur[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)


m4_func_neighbor_ruse <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year +
               author_density_cur + 
               ecological_cur +
               avg_author_pub_num_cur_log +
               avg_neighbor_pop_cur_log + 
               avg_author_ruse_preceding_1_log +
               neighbor_ruse_preceding_1_log +
               I(neighbor_ruse_preceding_1_log*age) +
               I(neighbor_ruse_preceding_1_log*age_2) +
               impact_factor_cur_log +
               inward_citation_cur +
               venue_abstract_history_cur +
               avg_entropy_s_cur +
               avg_engineering_pct_cur +
               avg_physics_pct_cur +
               avg_agriculture_pct_cur +
               avg_ss_pct_cur + 
               wordcnt_cur_log +
               readability_cur_log +
               author_female_cur +
               positivity_cur + negativity_cur + son_concept_num_log +
               (1 + age + age_2 | id),
           data = lmer_data,family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m4_neighbor_ruse <- parLapply(cls, 1, m4_func_neighbor_ruse))
sink("intercept_burnin1993_neighbor_ruse_interaction.txt")
summary(m4_neighbor_ruse[[1]])
sink()

cc <- coef(summary(m4_neighbor_ruse[[1]]))
phi <- overdisp_fun(m4_neighbor_ruse[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)

m4_func_author_ruse <- function(i){
 glmer(df_lead_original ~ 
               age + 
               age_2 + term_start_year + 
               author_density_cur + 
               ecological_cur +
               avg_author_pub_num_cur_log +
               avg_neighbor_pop_cur_log + 
               avg_author_ruse_preceding_1_log +
               I(avg_author_ruse_preceding_1_log*age) +
               I(avg_author_ruse_preceding_1_log*age_2) +
               neighbor_ruse_preceding_1_log +
               impact_factor_cur_log +
               inward_citation_cur +
               venue_abstract_history_cur +
               avg_entropy_s_cur +
               avg_engineering_pct_cur +
               avg_physics_pct_cur +
               avg_agriculture_pct_cur +
               avg_ss_pct_cur + 
               wordcnt_cur_log +
               readability_cur_log +
               author_female_cur +
               positivity_cur + negativity_cur + son_concept_num_log +
               (1 + age + age_2 | id),
           data = lmer_data,family = poisson, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e6, maxit = 1e9,xtol_abs = 1e-11, ftol_abs = 1e-11), check.conv.grad=.makeCC("warning", tol=0.008)))}
cls <- makeCluster(10)
clusterEvalQ(cls, library(lme4, lib = "/dfs/scratch0/hanchcao/hypecycle_xiang_complete/rpackage"))
clusterExport(cls, c("lmer_data"), envir=environment())
system.time(m4_author_ruse <- parLapply(cls, 1, m4_func_author_ruse))
sink("intercept_burnin1993_author_ruse_interaction.txt")
summary(m4_author_ruse[[1]])
sink()

cc <- coef(summary(m4_author_ruse[[1]]))
phi <- overdisp_fun(m4_author_ruse[[1]])["ratio"]
cc <- within(as.data.frame(cc),
             {   `Std. Error` <- `Std. Error`*sqrt(phi)
             `z value` <- Estimate/`Std. Error`
             `Pr(>|z|)` <- 2*pnorm(abs(`z value`), lower.tail=FALSE)
             })
printCoefmat(cc,digits=3)


