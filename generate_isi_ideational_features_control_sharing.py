######################################################################
######################################################################
####Code for "Modeling the Growth of Scientific Concepts" ############
####Code for ideational controls #####################################
####Correspondence:###################################################
####Daniel A. McFarland###############################################
####e-mail: mcfarland@stanford.edu####################################
######################################################################
######################################################################


import sys, operator, math
import pandas as pd
import numpy as np
from collections import defaultdict
from math import *
import random
import pickle
import itertools
from multiprocessing import Process, Manager


dir = "/dfs/scratch0/hanchcao/data/isi_random/"
BURN_IN_YEAR = 1992
START_YEAR = 1992 
END_YEAR = 2016


file_paper_emotion_readability = '/dfs/scratch0/hanchcao/data/isi_random/all_paperid_reademo.txt'
file_isi_term_paperlist = dir + 'ISI_term_paperlist_full_corpus_lemmitized_loose_corrected_final.txt'

file_isi_paper_year = dir + 'ISI_paper_year.txt'
file_term_df_year = dir + 'ISI_df_year_full_corpus_lemmitized_loose_corrected_final.txt'
file_total_df_year = dir + 'ISI_paper_each_year.txt'


def load_paper2item(filename):
    paper2item = defaultdict(str)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if pair[1].strip() != '':
                paper2item[pair[0].strip()] = pair[1].strip()
    print('load # papers: %d'%len(paper2item))
    return paper2item

def load_paper2list2(filename):
    paper2list = defaultdict(list)
    with open(filename) as f:
        for line in f:
            line = line.strip().split('\t')
            paper2list[line[0].strip()] = map(lambda x: x, line[1:])
    print('load # papers: %d'%len(paper2list))
    return paper2list

def get_first_year(term_df_year):
    term_list = []
    term_FIRSTYEAR = []
    for term in term_df_year:
        term_list.append(term)
        sentinal = False
        for idx in xrange(0, len(term_df_year[term]), 1):    
            if term_df_year[term][idx] != 0: 
                term_FIRSTYEAR.append(idx+START_YEAR)
                sentinal = True
                break
        if not sentinal:
            term_FIRSTYEAR.append(START_YEAR)
    return term_list, term_FIRSTYEAR

def get_end_year(term_df_year):
    term_list = []
    term_ENDYEAR = []
    for term in term_df_year:
        term_list.append(term)
        sentinal = False
        for idx in xrange(len(term_df_year[term])-1, -1, -1):    
            if term_df_year[term][idx] != 0: 
                term_ENDYEAR.append(idx+START_YEAR)
                sentinal = True
                break
        if not sentinal:
            term_ENDYEAR.append(END_YEAR)
    return term_list, term_ENDYEAR

def load_df_burnin(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            previous = map(lambda x: float(x), line[1:][:BURN_IN_YEAR - 1900])
            if sum(previous) != 0.0:
                continue
            else:
                features[line[0].strip()] = map(lambda x: float(x), line[1:][START_YEAR - 1900:END_YEAR-1900+1])
    print ('load # terms: %d', len(features))
    return features

def load_total_df_year(filename):
    total_df_year = [0.0] * (END_YEAR - START_YEAR + 1)
    with open(filename) as f:
        for line in f:
            year, count = line.split('\t')
            if int(year) <= END_YEAR and int(year) >= START_YEAR:
                total_df_year[int(year) - START_YEAR] = float(count)
    return total_df_year


print("loading term_paper_list")
term_paper_list = load_paper2list2(file_isi_term_paperlist)
print("# terms:", len(term_paper_list))

print("loading paper_year")
paper_year = load_paper2item(file_isi_paper_year)
print("original paper year: ", len(paper_year))
paper_year = dict((k, int(v)) for k, v in paper_year.items() if int(v) >= START_YEAR and int(v) <= END_YEAR)
print("valid paper year: ", len(paper_year))

print("loading term_df_year")
term_df_year = load_df_burnin(file_term_df_year)
print("# terms:" ,len(term_df_year))
print("loading total_df_year")
total_df_year = load_total_df_year(file_total_df_year)
print("# years:", len(total_df_year))

print("loading term list, term end year")
term_list, term_ENDYEAR = get_end_year(term_df_year)
print("# terms:", len(term_list))
print("loading term list, term first year")
term_list, term_STARTYEAR = get_first_year(term_df_year)
print("# terms:", len(term_list))


print("loading paper emotionality readability")
paper_emotion_readability = load_paper2list2(file_paper_emotion_readability)
print("# terms:", len(term_paper_list))


paper_positivity = {}
paper_negativity = {}
paper_readability = {}
paper_word_cnt = {}
cnt = 0


for paper in paper_emotion_readability:
    cnt += 1
    values = paper_emotion_readability[paper]
    abstract = values[0]
    title = values[1]
    readability = float(values[2])
    vaden = float(values[3])
    word_cnt = len(abstract.split())
    paper_readability[paper] = readability
    paper_word_cnt[paper] = word_cnt 
    if vaden > 0:
        paper_positivity[paper] = vaden
    else:
        paper_negativity[paper] = abs(vaden)   
    if cnt%100000==0:
        print(cnt)


def get_term_citation_feature(term_citation_year, term_start_year, term_end_year, term_current_year):
    if term_current_year < term_start_year or term_current_year > term_end_year:
        start = -1.0
        total = -1.0
        avg = -1.0
        current = -1.0
    else:
        start = term_citation_year[term_start_year-START_YEAR]
        total = sum(term_citation_year[term_start_year-START_YEAR:term_current_year-START_YEAR+1])
        avg = total*1.0/(term_current_year-term_start_year+1)
        current = term_citation_year[term_current_year-START_YEAR]
    return start, total, avg, current


def get_term_citation_preceeding_feature(term_citation_year, term_current_year, k):
    if term_current_year != START_YEAR:
        first_diff = term_citation_year[term_current_year-START_YEAR] - term_citation_year[term_current_year-START_YEAR-1]
    else:
        first_diff = 0.0
    if term_current_year-k < START_YEAR:
        preceding = 0.0
        diff_cr = 0.0
    else:
        preceding = term_citation_year[term_current_year-k-START_YEAR]
        diff = term_citation_year[term_current_year-START_YEAR]-term_citation_year[term_current_year-k-START_YEAR]
        diff_cr = diff*1.0/k
    return preceding, diff_cr, first_diff


def get_term_yearly_paperlist(paper_list, paper_year):
    term_paper_year_list = defaultdict(list)
    for paper in paper_list:
        paper = paper.strip()
        if paper in paper_year:
            term_paper_year_list[int(paper_year[paper] - START_YEAR)].append(paper)
    return term_paper_year_list

def get_emotionality(paper_list, paper_year, paper_emotionality):
    term_paper_year_list = get_term_yearly_paperlist(paper_list, paper_year)
    term_yearly_emotionality = [0.0]*(END_YEAR - START_YEAR + 1)
    for year in term_paper_year_list:
        #print(year)
        yearly_paper_list = term_paper_year_list[year]
        emotionality = 0
        cnt = 0
        for paper in yearly_paper_list:
            if paper in paper_emotionality and paper_emotionality[paper] != -1234 :
                cnt += 1
                emotionality += paper_emotionality[paper]
        if cnt != 0:
            result = emotionality*1.0/cnt
        else:
            result = -1234
        term_yearly_emotionality[year] = result
    return term_yearly_emotionality


def feature_worker(terms, term_startyears, term_endyears, paper_year, term_paper_list, paper_positivity, paper_negativity, paper_readability, paper_word_cnt, term_feature_all): 
    term_feature = []
    for i in range(len(terms)):
        term = terms[i]
        all_features = []
        term_end_year = term_endyears[i]
        term_start_year = term_startyears[i]
        positivity = get_emotionality(term_paper_list[term], paper_year, paper_positivity)
        negativity = get_emotionality(term_paper_list[term], paper_year, paper_negativity)
        readability = get_emotionality(term_paper_list[term], paper_year, paper_readability)
        word_cnt = get_emotionality(term_paper_list[term], paper_year, paper_word_cnt)
        
        for j in xrange(END_YEAR - START_YEAR + 1):
            term_current_year = START_YEAR + j
            features = defaultdict(float)
            positivity_start, positivity_sum, positivity_avg, positivity_cur = get_term_citation_feature(positivity, term_start_year, term_end_year, term_current_year)
            positivity_preceding_1, positivity_cr_1, positivity_firstdif = get_term_citation_preceeding_feature(positivity, term_current_year, 1)

            features['positivity_start'] = positivity_start 
            features['positivity_sum'] = positivity_sum 
            features['positivity_preceding_1'] = positivity_preceding_1 
            features['positivity_firstdif'] = positivity_firstdif
            features['positivity_cur'] = positivity_cur

            negativity_start, negativity_sum, negativity_avg, negativity_cur = get_term_citation_feature(negativity, term_start_year, term_end_year, term_current_year)
            negativity_preceding_1, negativity_cr_1, negativity_firstdif = get_term_citation_preceeding_feature(negativity, term_current_year, 1)
            features['negativity_start'] = negativity_start 
            features['negativity_sum'] = negativity_sum 
            features['negativity_preceding_1'] = negativity_preceding_1 
            features['negativity_firstdif'] = negativity_firstdif
            features['negativity_cur'] = negativity_cur

            readability_start, readability_sum, readability_avg, readability_cur = get_term_citation_feature(readability, term_start_year, term_end_year, term_current_year)
            readability_preceding_1, readability_cr_1, readability_firstdif = get_term_citation_preceeding_feature(readability, term_current_year, 1)
            features['readability_start'] = readability_start 
            features['readability_sum'] = readability_sum 
            features['readability_preceding_1'] = readability_preceding_1 
            features['readability_firstdif'] = readability_firstdif
            features['readability_cur'] = readability_cur

            wordcnt_start, wordcnt_sum, wordcnt_avg, wordcnt_cur = get_term_citation_feature(word_cnt, term_start_year, term_end_year, term_current_year)
            wordcnt_preceding_1, wordcnt_cr_1, wordcnt_firstdif = get_term_citation_preceeding_feature(word_cnt, term_current_year, 1)
            features['wordcnt_start'] = wordcnt_start 
            features['wordcnt_sum'] = wordcnt_sum 
            features['wordcnt_preceding_1'] = wordcnt_preceding_1 
            features['wordcnt_firstdif'] = wordcnt_firstdif
            features['wordcnt_cur'] = wordcnt_cur
            all_features.append(features)
            
        term_feature.append((term, all_features))
        if i % 1000 == 0:
             print(i)
    term_feature_all.extend(term_feature)   


numOfProcesses = 10

termsPerProc = int(math.floor(len(term_list) * 1.0/numOfProcesses))
mgr = Manager()
term_feature_all = mgr.list()
jobs = []
for i in range(numOfProcesses):
    if i == numOfProcesses - 1:
        p = Process(target=feature_worker, args=(term_list[i*termsPerProc:], term_STARTYEAR[i*termsPerProc:], term_ENDYEAR[i*termsPerProc:], paper_year, term_paper_list, paper_positivity, paper_negativity, paper_readability, paper_word_cnt, term_feature_all))
    else:
        p = Process(target=feature_worker, args=(term_list[i*termsPerProc:(i+1)*termsPerProc], term_STARTYEAR[i*termsPerProc:(i+1)*termsPerProc], term_ENDYEAR[i*termsPerProc:(i+1)*termsPerProc], paper_year, term_paper_list, paper_positivity, paper_negativity, paper_readability, paper_word_cnt, term_feature_all))
    p.start()
    jobs.append(p)
for proc in jobs:
    proc.join()
print('done')


f_out_term_feature = dir + 'isi_ideational_features_loose.txt'

### dump to files
print("begin dumping")
print("# terms:", len(term_feature_all))
with open(f_out_term_feature, 'w') as f:
    for pair in term_feature_all:
        term = pair[0]
        for features in pair[1]:
            f.write(term + '\t' +
                str(features['positivity_start']) + ' ' + str(features['positivity_sum']) + ' ' + \
                str(features['positivity_preceding_1']) + ' ' +  str(features['positivity_firstdif']) + ' ' + str(features['positivity_cur']) + ' ' + \
                str(features['negativity_start']) + ' ' + str(features['negativity_sum']) + ' ' + \
                str(features['negativity_preceding_1']) + ' ' + str(features['negativity_firstdif']) + ' ' + str(features['negativity_cur']) + ' ' + \
                str(features['readability_start']) + ' ' + str(features['readability_sum']) + ' ' + \
                str(features['readability_preceding_1']) + ' ' + str(features['readability_firstdif']) + ' ' + str(features['readability_cur']) + ' ' +\
                str(features['wordcnt_start']) + ' ' + str(features['wordcnt_sum']) + ' ' + \
                str(features['wordcnt_preceding_1']) + ' ' + str(features['wordcnt_firstdif']) + ' ' + str(features['wordcnt_cur']) + '\n')
print("done")

