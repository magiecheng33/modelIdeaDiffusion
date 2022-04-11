######################################################################
######################################################################
####code for "Modeling the Growth of Scientific Concepts" ############
####code for generating venue controls ###############################
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


dir = '/dfs/scratch0/hanchcao/data/isi_random/'
file_isi_term_paperlist = dir + 'ISI_term_paperlist_full_corpus_lemmitized_loose_corrected_final.txt'


file_isi_paper_year = dir + 'ISI_paper_year.txt'
file_term_df_year = dir + 'ISI_df_year_full_corpus_lemmitized_loose_corrected_final.txt'
file_total_df_year = dir + 'ISI_paper_each_year.txt'
file_isi_paper_venue = dir + 'ISI_paper_venue.txt'
file_isi_venue_start_year = dir + 'ISI_venue_start_year.txt'
file_isi_venue_abstract_start_year = dir + 'ISI_venue_abstract_start_year.txt'


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


def load_paper2list_reverse(filename):
    item2paperlist = defaultdict(list)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if len(pair) == 2:
                item2paperlist[pair[1][:-1].strip()].append(pair[0].strip())
    print('load # items: %d'%len(item2paperlist))
    return item2paperlist


print("loading term_paper_list")
term_paper_list = load_paper2list2(file_isi_term_paperlist)
print("# terms:", len(term_paper_list))

print("loading paper_year")
paper_year = load_paper2item(file_isi_paper_year)
print("original paper year: ", len(paper_year))

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

print("loading paper_venue")
paper_venue = load_paper2item(file_isi_paper_venue)
print(len(paper_venue))

print("loading venue_paper_list")
venue_paper_list = load_paper2list_reverse(file_isi_paper_venue)
print("# venue:", len(venue_paper_list))

print("loading venue start year")
venue_start_year = load_paper2item(file_isi_venue_start_year)
print(len(venue_start_year))

print("loading venue abstract start year")
venue_abstract_start_year = load_paper2item(file_isi_venue_abstract_start_year)
print(len(venue_abstract_start_year))


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


print("loading venue start year")
venue_start_year = load_paper2item(file_isi_venue_start_year)
print(len(venue_start_year))

print("loading venue abstract start year")
venue_abstract_start_year = load_paper2item(file_isi_venue_abstract_start_year)
print(len(venue_abstract_start_year))

venue_start_year.pop('', None)
venue_abstract_start_year.pop('', None)


def get_term_startpaperlist(paper_list, term_start_year, paper_year):
    startpaperlist = []  
    for paper in paper_list:
        year = int(paper_year[paper])
        if term_start_year == year:
            startpaperlist.append(paper)
    return startpaperlist


def gen_venue_abstract_history(paper_list, paper_year, paper_venue, term_start_year, venue_abstract_start_year):
    start_paper_list = get_term_startpaperlist(paper_list, term_start_year, paper_year)
    abstract_venue_num_list = []
    for paper in start_paper_list:
        year = int(paper_year[paper])
        venue = paper_venue[paper]
        if venue != '':
            abstract_start_year = int(venue_abstract_start_year[venue])
            if abstract_start_year != -1234 and year - abstract_start_year < 2018-1900+1: 
                venue_abstract_age = year - abstract_start_year
                abstract_venue_num_list.append(venue_abstract_age) 
                
    if len(abstract_venue_num_list) != 0:
        venue_abstract_history = sum(abstract_venue_num_list)*1.0/len(abstract_venue_num_list)
    else: 
        venue_abstract_history = -1234
    venue_abstract_history_list = [venue_abstract_history]*(END_YEAR - START_YEAR + 1) 
    return venue_abstract_history_list


def gen_new_venue_pct(paper_list, paper_year, paper_venue, term_start_year, venue_start_year):
    start_paper_list = get_term_startpaperlist(paper_list, term_start_year, paper_year)
    new_venue_cnt = 0
    for paper in start_paper_list:
        year = int(paper_year[paper])
        venue = paper_venue[paper]
        if venue != '':
            start_year_venue = int(venue_start_year[venue])
            if start_year_venue != -1234 and year == start_year_venue:
                new_venue_cnt += 1
    if len(start_paper_list) != 0:
        new_venue_pct = new_venue_cnt * 1.0 / len(start_paper_list)
    else:
        new_venue_pct = -1234
    new_venue_pct_list = [new_venue_pct] *(END_YEAR - START_YEAR + 1) 
    return new_venue_pct_list


def get_lagging_venue_abstract_history(paper_list, paper_year, paper_venue, term_start_year, venue_start_year, venue_abstract_start_year):
    start_paper_list = get_term_startpaperlist(paper_list, term_start_year, paper_year)
    abstract_venue_num_list = []
    for paper in start_paper_list:
        year = int(paper_year[paper])
        venue = paper_venue[paper]
        if venue != '':
            abstract_start_year = int(venue_abstract_start_year[venue])
            start_year_venue = int(venue_start_year[venue])
            if abstract_start_year != -1234 and abstract_start_year - start_year_venue < 2018-1900+1: 
                abstract_venue_num_list.append(abstract_start_year - start_year_venue) 
    if len(abstract_venue_num_list) != 0:
        venue_abstract_lagging_history = sum(abstract_venue_num_list)*1.0/len(abstract_venue_num_list)
    else: 
        venue_abstract_lagging_history = -1234
    venue_abstract_lagging_history_list = [venue_abstract_lagging_history]*(END_YEAR - START_YEAR + 1) 
    return venue_abstract_lagging_history_list


def feature_worker(terms, term_startyears, term_endyears, paper_year, term_paper_list, venue_abstract_start_year, venue_start_year, term_feature_all): 
    term_feature = []
    for i in range(len(terms)):
        term = terms[i]
        all_features = []
        term_end_year = term_endyears[i]
        term_start_year = term_startyears[i]
        venue_abstract_history = gen_venue_abstract_history(term_paper_list[term], paper_year, paper_venue, term_start_year, venue_abstract_start_year)
        new_venue_pct = gen_new_venue_pct(term_paper_list[term], paper_year, paper_venue, term_start_year, venue_start_year)
        lagging_venue_abstract_his = get_lagging_venue_abstract_history(term_paper_list[term], paper_year, paper_venue, term_start_year, venue_start_year, venue_abstract_start_year)
        
        for j in xrange(END_YEAR - START_YEAR + 1):
            term_current_year = START_YEAR + j
            features = defaultdict(float)
            venue_abstract_history_start, venue_abstract_history_sum, venue_abstract_history_avg, venue_abstract_history_cur = get_term_citation_feature(venue_abstract_history, term_start_year, term_end_year, term_current_year)
            venue_abstract_history_preceding_1, venue_abstract_history_cr_1, venue_abstract_history_firstdif = get_term_citation_preceeding_feature(venue_abstract_history, term_current_year, 1)

            features['venue_abstract_history_start'] = venue_abstract_history_start 
            features['venue_abstract_history_sum'] = venue_abstract_history_sum 
            features['venue_abstract_history_preceding_1'] = venue_abstract_history_preceding_1 
            features['venue_abstract_history_firstdif'] = venue_abstract_history_firstdif
            features['venue_abstract_history_cur'] = venue_abstract_history_cur

            new_venue_pct_start, new_venue_pct_sum, new_venue_pct_avg, new_venue_pct_cur = get_term_citation_feature(new_venue_pct, term_start_year, term_end_year, term_current_year)
            new_venue_pct_preceding_1, new_venue_pct_cr_1, new_venue_pct_firstdif = get_term_citation_preceeding_feature(new_venue_pct, term_current_year, 1)
            features['new_venue_pct_start'] = new_venue_pct_start 
            features['new_venue_pct_sum'] = new_venue_pct_sum 
            features['new_venue_pct_preceding_1'] = new_venue_pct_preceding_1 
            features['new_venue_pct_firstdif'] = new_venue_pct_firstdif
            features['new_venue_pct_cur'] = new_venue_pct_cur

            lagging_venue_abstract_his_start, lagging_venue_abstract_his_sum, lagging_venue_abstract_his_avg, lagging_venue_abstract_his_cur = get_term_citation_feature(lagging_venue_abstract_his, term_start_year, term_end_year, term_current_year)
            lagging_venue_abstract_his_preceding_1, lagging_venue_abstract_his_cr_1, lagging_venue_abstract_his_firstdif = get_term_citation_preceeding_feature(lagging_venue_abstract_his, term_current_year, 1)
            features['lagging_venue_abstract_his_start'] = lagging_venue_abstract_his_start 
            features['lagging_venue_abstract_his_sum'] = lagging_venue_abstract_his_sum 
            features['lagging_venue_abstract_his_preceding_1'] = lagging_venue_abstract_his_preceding_1 
            features['lagging_venue_abstract_his_firstdif'] = lagging_venue_abstract_his_firstdif
            features['lagging_venue_abstract_his_cur'] = lagging_venue_abstract_his_cur

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
        p = Process(target=feature_worker, args=(term_list[i*termsPerProc:], term_STARTYEAR[i*termsPerProc:], term_ENDYEAR[i*termsPerProc:], paper_year, term_paper_list, venue_abstract_start_year, venue_start_year, term_feature_all))
    else:
        p = Process(target=feature_worker, args=(term_list[i*termsPerProc:(i+1)*termsPerProc], term_STARTYEAR[i*termsPerProc:(i+1)*termsPerProc], term_ENDYEAR[i*termsPerProc:(i+1)*termsPerProc], paper_year, term_paper_list, venue_abstract_start_year, venue_start_year, term_feature_all))
    p.start()
    jobs.append(p)
for proc in jobs:
    proc.join()
print('done')


f_out_term_feature = dir + 'isi_venue_control_loose.txt'

### dump to files
print("begin dumping")
print("# terms:", len(term_feature_all))
with open(f_out_term_feature, 'w') as f:
    for pair in term_feature_all:
        term = pair[0]
        for features in pair[1]:
            f.write(term + '\t' +
                str(features['venue_abstract_history_start']) + ' ' + str(features['venue_abstract_history_sum']) + ' ' + \
                str(features['venue_abstract_history_preceding_1']) + ' ' +  str(features['venue_abstract_history_firstdif']) + ' ' + str(features['venue_abstract_history_cur']) + ' ' + \
                str(features['new_venue_pct_start']) + ' ' + str(features['new_venue_pct_sum']) + ' ' + \
                str(features['new_venue_pct_preceding_1']) + ' ' + str(features['new_venue_pct_firstdif']) + ' ' + str(features['new_venue_pct_cur']) + ' ' + \
                str(features['lagging_venue_abstract_his_start']) + ' ' + str(features['lagging_venue_abstract_his_sum']) + ' ' + \
                str(features['lagging_venue_abstract_his_preceding_1']) + ' ' + str(features['lagging_venue_abstract_his_firstdif']) + ' ' + str(features['lagging_venue_abstract_his_cur']) + '\n')
print("done")







