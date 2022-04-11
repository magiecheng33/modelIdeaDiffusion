######################################################################
######################################################################
####code for "Modeling the Growth of Scientific Concepts" ############
####code for merging all the variables to generate regression table ##
####Correspondence:###################################################
####Daniel A. McFarland###############################################
####e-mail: mcfarland@stanford.edu####################################
######################################################################
######################################################################

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import numpy as np
import random
from collections import defaultdict
dir = '/dfs/scratch0/hanchcao/data/isi_random/'
file_resource_feature = dir + 'isi_resources_features_loose.txt' 
file_total_df_year = dir + 'isi/ISI_paper_each_year.txt'
file_term_df_year = dir + 'ISI_df_year_full_corpus_lemmitized_loose_corrected_final.txt'
file_publication_num_year = dir + 'ISI_paper_each_year.txt'
file_ideational_condition = dir + 'isi_ideational_features_loose.txt'
file_ecological_features = dir + 'ISI_loose_ecological_features_word2vec_v3/ISI_loose_ecological_features_word2vec_v3_full.txt'
file_author_density = "/dfs/scratch0/hanchcao/data/isi_random/ISI_loose_ecological_features_author_density_0808_2021/ISI_loose_ecological_features_author_density_08082021.txt"
file_venue_control = dir + 'isi_venue_control_loose_03132021.txt'
file_inward_citation = dir + 'In_WoS_citation_percentage_loose_term.txt'
file_impact_factor = dir + 'concept_journal_impact_factor/concept_impact_factor_loose_term.txt'


BURN_IN_YEAR = 1992
START_YEAR = 1992
END_YEAR = 2016

def load_pub_num(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2:
                features[str(line[0])] = int(line[1])
    return features

def load_resource(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2:
                if line[0] not in features:
                    features[line[0]] = []
                features[line[0]].append([line[0]] + list(map(lambda x: float(x), line[1].split())))
                #features[line[0]].append(map(lambda x: float(x), line[1].split()))
    return features

def load_translation(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2:
                if line[0] not in features:
                    features[line[0]] = []
                features[line[0]].append(map(lambda x: float(x), line[1].split()))
    return features    

def load_ecological(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2:
                features[line[0]] = list(map(lambda x: float(x), line[1].split()))
    return features

def load_density(fname):
    features = {}
    cnt = 0
    with open(fname, 'r') as f:
        for line in f:
            cnt += 1
            line = line.strip().split('\t')
            if len(line) == 2:
                features[line[0]] = list(map(lambda x: float(x), line[1].split()))
    return features


# In[5]:


print("load pub num")
pub_num = load_pub_num(file_publication_num_year)
print("load resource features")
resource_features = load_resource(file_resource_feature)
print("load ecological features")
ecological_features = load_ecological(file_ecological_features)
print("load author density")
author_density = load_ecological(file_author_density)
print("load son number to denote root concept!")
son_concepts = pd.read_csv("/dfs/scratch0/hanchcao/data/isi/concept_substring_match/loose_SonConcept/SonConcept_Num_loose.txt", sep='\t', names = ['id', 'son_concept_num'])


# In[5]:


print("load inward citation")
inward_citation = load_ecological(file_inward_citation)
print("load impact factor")
impact_factor = load_ecological(file_impact_factor)


# In[10]:


def load_authornetworks(fname):
    features_list = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2:
                concept = line[0]
                features_time_list = line[1].split()
                for idx, feature in enumerate(features_time_list):
                    tmp = []
                    tmp.append(concept)
                    time = idx + 1992
                    tmp.append(time)
                    author_network = feature.split('_')[:-1]
                    tmp.extend(author_network)
                    features_list.append(tmp)
    return features_list

print("load author network data")

print("load ideation features")
ideation_features = load_translation(file_ideational_condition)

print("load venue control features")
venue_control = load_translation(file_venue_control)

print("checking variables!")
for key in author_density.keys()[0:10]:
    print("key:", key)
    print("author density:", author_density[key])
    print("author density:", author_density[key][2])
for key in ecological_features.keys()[0:10]:
    print("ecological feature:", key)
    print("ecological feature:", ecological_features[key][2])
    
for key in inward_citation.keys()[0:5]:
    print("inward citation:", inward_citation[key])
    print("inward citation:", inward_citation[key][2])
    
for key in impact_factor.keys()[0:5]:
    print("impact factor:", impact_factor[key])
    print("impact factor:", impact_factor[key][2])


def load_df(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            features[line[0].strip()] = list(map(lambda x: float(x), line[1:][START_YEAR - 1900:END_YEAR-1900+1]))
    return features

def load_df_burnin(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            previous = list(map(lambda x: float(x), line[1:][:BURN_IN_YEAR - 1900]))
            if sum(previous) != 0.0:
                continue
            else:
                features[line[0].strip()] = list(map(lambda x: float(x), line[1:][START_YEAR - 1900:END_YEAR-1900+1]))
    print ('load # terms: %d', len(features))
    return features


print("loading burn in document frequency")
df = load_df_burnin(file_term_df_year)
print("# terms", len(df))

more_than_120_set = set()
less_than_120_pd = []
for key in df:
    df_list = df[key]
    sum_list = sum(df_list)
    if sum_list >= 120:
        more_than_120_set.add(key)
    else:
        less_than_120_pd.append([key, sum_list])
len(more_than_120_set)

def get_word_length(row):
    length = len(row['id'].split())
    return length

tmp = pd.DataFrame(more_than_120_set, columns = ['id'])
tmp['word_length'] = tmp.apply(get_word_length, axis = 1)
selected_keys = more_than_120_set



def merge(df, resource_features, selected_keys, ideational_condition, venue_control, author_density, ecological_features, impact_factor, inward_citation):
    keys = set()
    cnt = 0
    intersect = set(selected_keys).intersection(resource_features.keys()).intersection(ideational_condition.keys()).intersection(venue_control.keys()).intersection(ecological_features.keys())
    print("intersection concept length:", len(intersect))
    for key in intersect:
        keys.add(key)
    features = []
    for idx, key in enumerate(keys):
        start_year = -1
        start = -1234
        start_author = -1234
        df_cum = 0
        cnt += 1
        if cnt % 30000 == 0:
            print(cnt)
        for i in range(25):
            if resource_features[key][i][-1] == 0:
                start_year = i+START_YEAR
        for i in range(25):
            #try:
            if ecological_features[key][i] != -1234 and start == -1234:
                start0 = ecological_features[key][i]
                start = -1
            if author_density[key][i] != -1234 and start_author == -1234:
                start2 = author_density[key][i]
                start_author = -1
            df_cum = df_cum + df[key][i]  
            if i >= 1:
                features.append(resource_features[key][i] + [start2] + [author_density[key][i-1]] + [author_density[key][i]] + [start0] + [ecological_features[key][i-1]] + [ecological_features[key][i]] +  ideational_condition[key][i] + venue_control[key][i] + [i] + [df_cum] + [df[key][i]] +[start_year] + [START_YEAR + i] + [impact_factor[key][i]] + [inward_citation[key][i]])  
            else:
                features.append(resource_features[key][i] + [start2] + [0] + [author_density[key][i]] + [start0] + [0] + [ecological_features[key][i]] +  ideational_condition[key][i] + venue_control[key][i] + [i] + [df_cum] + [df[key][i]] + [start_year] + [START_YEAR + i] + [impact_factor[key][i]] + [inward_citation[key][i]])  
    return np.array(features)

print("merging features")
features = merge(df, resource_features, selected_keys, ideation_features, venue_control, author_density, ecological_features,impact_factor, inward_citation)
print ("merge finished")
print(len(features))
print(features.shape)


names  = ['id', 'avg_venue_citation_start', 'avg_venue_citation_total', 'avg_venue_citation_preceding_1',
         'avg_venue_citation_first_diff', 'avg_venue_citation_cur', 'avg_entropy_start', 'avg_entropy_total', 'avg_entropy_preceding_1', 'avg_entropy_first_diff',
         'avg_entropy_cur','avg_entropy_s_start', 'avg_entropy_s_total', 'avg_entropy_s_preceding_1', 'avg_entropy_s_first_diff',
         'avg_entropy_s_cur', 'new_neighbor_wgt_pct_start', 'new_neighbor_wgt_pct_total', 'new_neighbor_wgt_pct_preceding_1', 'new_neighbor_wgt_pct_first_diff', 'new_neighbor_wgt_pct_cur',
          'new_neighbor_avg_pct_start', 'new_neighbor_avg_pct_total', 'new_neighbor_avg_pct_preceding_1','new_neighbor_avg_pct_first_diff','new_neighbor_avg_pct_cur',
          'new_author_pct_avg_start', 'new_author_pct_avg_total', 'new_author_pct_avg_preceding_1', 'new_author_pct_avg_first_diff', 'new_author_pct_avg_cur',
          'new_author_pct_wgt_start', 'new_author_pct_wgt_total', 'new_author_pct_wgt_preceding_1', 'new_author_pct_wgt_first_diff', 'new_author_pct_wgt_cur', 'avg_author_size_start', 'avg_author_size_total', 'avg_author_size_preceding_1', 'avg_author_size_first_diff', 'avg_author_size_cur', 
         'avg_author_pub_num_start', 'avg_author_pub_num_total', 'avg_author_pub_num_preceding_1', 'avg_author_pub_num_first_diff', 'avg_author_pub_num_cur', 
         'avg_author_ruse_start', 'avg_author_ruse_preceding_1', 'author_female_start', 'author_female_total', 'author_female_preceding_1', 'author_female_first_diff', 'author_female_cur', 
         'avg_neighbor_pop_start', 'avg_neighbor_pop_total', 'avg_neighbor_pop_preceding_1', 'avg_neighbor_pop_first_diff', 'avg_neighbor_pop_cur', 
         'sum_neighbor_pop_start', 'sum_neighbor_pop_total', 'sum_neighbor_pop_preceding_1', 'sum_neighbor_pop_first_diff', 'sum_neighbor_pop_cur', 
         'mid_neighbor_pop_start', 'mid_neighbor_pop_total', 'mid_neighbor_pop_preceding_1', 'mid_neighbor_pop_first_diff', 'mid_neighbor_pop_cur', 
         'neighbor_ruse_start', 'neighbor_ruse_preceding_1', 'avg_physics_pct_start', 'avg_physics_pct_total', 'avg_physics_pct_preceding_1', 'avg_physics_pct_first_diff',
         'avg_physics_pct_cur', 'avg_engineering_pct_start', 'avg_engineering_pct_total', 'avg_engineering_pct_preceding_1', 'avg_engineering_pct_first_diff',
         'avg_engineering_pct_cur', 'avg_agriculture_pct_start', 'avg_agriculture_pct_total', 'avg_agriculture_pct_preceding_1', 'avg_agriculture_pct_first_diff',
         'avg_agriculture_pct_cur', 'avg_bio_pct_start', 'avg_bio_pct_total', 'avg_bio_pct_preceding_1', 'avg_bio_pct_first_diff',
         'avg_bio_pct_cur', 'avg_ss_pct_start', 'avg_ss_pct_total', 'avg_ss_pct_preceding_1', 'avg_ss_pct_first_diff',
         'avg_ss_pct_cur', 'concept_age', 
         'author_density_start', 'author_density_preceding_1', 'author_density_cur',
         'ecological_start', 'ecological_preceding_1', 'ecological_cur',
         'positivity_start', 'positivity_sum', 'positivity_preceding_1', 'positivity_firstdif', 'positivity_cur',
         'negativity_start', 'negativity_sum', 'negativity_preceding_1', 'negativity_firstdif', 'negativity_cur',
         'readability_start', 'readability_sum', 'readability_preceding_1', 'readability_firstdif', 'readability_cur',
         'wordcnt_start', 'wordcnt_sum', 'wordcnt_preceding_1', 'wordcnt_firstdif', 'wordcnt_cur', 
         'venue_abstract_history_start', 'venue_abstract_history_sum', 'venue_abstract_history_preceding_1',
         'venue_abstract_history_firstdif', 'venue_abstract_history_cur', 'new_venue_pct_start', 'new_venue_pct_sum',
         'new_venue_pct_preceding_1', 'new_venue_pct_firstdif',  'new_venue_pct_cur', 'lagging_venue_abstract_his_start',
         'lagging_venue_abstract_his_sum', 'lagging_venue_abstract_his_preceding_1', 'lagging_venue_abstract_his_firstdif',
         'lagging_venue_abstract_his_cur', 'age','df_cum', 'df', 'term_start_year', 'df_year', 'impact_factor_cur',
         'inward_citation_cur']

print(len(names))


data = pd.DataFrame(features, columns=names)
print(data.shape)
data['term_start_year'] = data['term_start_year'].astype(int)
data['df_year'] = data['df_year'].astype(int)
data12 = data[data['df_year'] >= data['term_start_year']]
print(data12.shape)


pub = pd.DataFrame(pub_num.items(), columns=['df_year', 'pub_num'])
data12['df_year'] = data12['df_year'].astype(int)
pub['df_year'] = pub['df_year'].astype(int)
data122 = data12.merge(pub, on = 'df_year')
print(data122.shape)


def get_no_usage(row):
    if float(row['readability_preceding_1']) == 0:
        return 1
    else:
        return 0
print('get no usage var...')

data122['no_usage'] = data122.apply(get_no_usage, axis = 1)


selected_features = ['id', 'avg_venue_citation_start', 'avg_venue_citation_preceding_1', 'avg_venue_citation_cur', 
                     'avg_entropy_start', 'avg_entropy_preceding_1', 'avg_entropy_cur',
                     'avg_entropy_s_start', 'avg_entropy_s_preceding_1', 'avg_entropy_s_cur',
         'new_neighbor_wgt_pct_start',  'new_neighbor_wgt_pct_preceding_1', 'new_neighbor_wgt_pct_cur',
          'new_neighbor_avg_pct_start', 'new_neighbor_avg_pct_preceding_1','new_neighbor_avg_pct_cur',
          'new_author_pct_avg_start',  'new_author_pct_avg_preceding_1',  'new_author_pct_avg_cur',
          'new_author_pct_wgt_start', 'new_author_pct_wgt_preceding_1', 'new_author_pct_wgt_cur', 'avg_author_size_start', 'avg_author_size_preceding_1',  'avg_author_size_cur', 
         'avg_author_pub_num_start', 'avg_author_pub_num_preceding_1', 'avg_author_pub_num_cur', 
         'avg_author_ruse_start', 
         'avg_author_ruse_preceding_1', 
         'author_female_start', 'author_female_preceding_1', 'author_female_cur', 
         'avg_neighbor_pop_start', 'avg_neighbor_pop_preceding_1', 'avg_neighbor_pop_cur', 
         'neighbor_ruse_start', 'neighbor_ruse_preceding_1', 
         'avg_physics_pct_start',  'avg_physics_pct_preceding_1', 'avg_physics_pct_cur', 
         'avg_engineering_pct_start', 'avg_engineering_pct_preceding_1', 'avg_engineering_pct_cur',
         'avg_agriculture_pct_start', 'avg_agriculture_pct_preceding_1','avg_agriculture_pct_cur', 
         'avg_bio_pct_start', 'avg_bio_pct_preceding_1', 'avg_bio_pct_cur', 
         'avg_ss_pct_start', 'avg_ss_pct_preceding_1', 'avg_ss_pct_cur', 'concept_age', 
          'author_density_start', 'author_density_preceding_1', 'author_density_cur',
         'ecological_start', 'ecological_preceding_1', 'ecological_cur',
         'positivity_start',  'positivity_preceding_1', 'positivity_cur',
         'negativity_start',  'negativity_preceding_1', 'negativity_cur',
         'readability_start', 'readability_preceding_1',  'readability_cur',
         'wordcnt_start', 'wordcnt_preceding_1', 'wordcnt_cur', 
         'venue_abstract_history_start', 'venue_abstract_history_preceding_1',
         'venue_abstract_history_cur', 'new_venue_pct_start', 
         'new_venue_pct_preceding_1',  'new_venue_pct_cur', 'lagging_venue_abstract_his_start',
         'lagging_venue_abstract_his_preceding_1', 
         'lagging_venue_abstract_his_cur', 'age','df_cum', 'df', 'term_start_year', 'df_year', 'impact_factor_cur', 'inward_citation_cur', 'no_usage',  'pub_num']
len(selected_features)

mergeddata = data122[selected_features]


def get_word_length(row):
    num_words = len(str(row['id']).split())
    return num_words

mergeddata['word_length'] = mergeddata.apply(get_word_length, axis = 1)


def get_word_char_length(row):
    num_words = len(str(row['id']))
    return num_words

mergeddata['word_char_length'] = mergeddata.apply(get_word_char_length, axis = 1)

mergeddata = mergeddata.merge(son_concepts, on = ['id'], how ='left')


mergeddata['son_concept_num'] = mergeddata['son_concept_num'].fillna(0)

selected_variables = ["id",
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
                     "concept_age"]
mergeddata[selected_variables].to_csv(dir+"final_regression_table_loose_burnin1993_sharing.csv")


