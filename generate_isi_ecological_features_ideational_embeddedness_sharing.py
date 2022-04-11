######################################################################
######################################################################
####code for "Modeling the Growth of Scientific Concepts" ############
####code for generating ideational embeddedness ######################
####Correspondence:###################################################
####Daniel A. McFarland###############################################
####e-mail: mcfarland@stanford.edu####################################
######################################################################
######################################################################


#!/usr/bin/env python
# coding: utf-8


from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from collections import defaultdict
import pickle
import itertools
import math
from multiprocessing import Process, Manager
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import random


dir = '/dfs/scratch0/hanchcao/data/'
f_term_word2vec = dir + 'isi_random/wordvecs_loose_preceding10/ISI_all_loose_word2vec_%d.bin'
file_term_df_year = dir + 'isi_random/ISI_df_year_full_corpus_lemmitized_loose_corrected_final.txt'
file_total_df_year = dir + 'isi_random/ISI_paper_each_year.txt'
f_term_semnet_features = dir + 'isi_random/ISI_loose_ecological_features_word2vec_v3/'
file_isi_term_paperlist = dir + 'isi_random/ISI_term_paperlist_full_corpus_lemmitized_loose_corrected_final.txt'


BURN_IN_YEAR = 1992
START_YEAR = 1992 
END_YEAR = 2016 


def load_term_df_year(filename):
    term_df_year = defaultdict(list)
    with open(filename) as f:
        for line in f:
            entries = line.split('\t')
            df_raw = list(map(float, entries[1:]))
            df_raw = df_raw[START_YEAR-1900:END_YEAR-1900+1]
            term_df_year[entries[0]] = df_raw
    print ('load # terms: %d', len(term_df_year))
    return term_df_year


def load_df_burnin(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            previous = list(map(lambda x: float(x), line[1:][:BURN_IN_YEAR - 1900]))
            if sum(previous) != 0.0:
                continue
            else:
                features[line[0]] = list(map(lambda x: float(x), line[1:][START_YEAR - 1900:END_YEAR-1900+1]))
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


term_df_year = load_df_burnin(file_term_df_year)
total_df_year = load_total_df_year(file_total_df_year)



term_list = list(term_df_year.keys())  


wvs = [KeyedVectors.load_word2vec_format(f_term_word2vec%year, binary=True) for year in range(START_YEAR, END_YEAR+1, 1)]
print ('load Word2Vec done')


term_avg_sim = [-1234] * (len(term_list) * (END_YEAR-START_YEAR+1))
wordvecs_list = [0.0] * (len(term_list) * (END_YEAR-START_YEAR+1))
count_missing = 0
count_pair_missing = 0


def load_paper2list2(filename):
    paper2list = defaultdict(list)
    with open(filename) as f:
        for line in f:
            line = line.strip().split('\t')
            paper2list[line[0].strip()] = list(map(lambda x: x, line[1:]))
    print('load # papers: %d'%len(paper2list))
    return paper2list

print("loading term_paper_list")
term_paper_list = load_paper2list2(file_isi_term_paperlist)
print("# terms:", len(term_paper_list))


file_isi_paper_year = dir + 'isi_random/ISI_paper_year.txt'

def load_paper2item(filename):
    paper2item = defaultdict(str)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if pair[1].strip() != '':
                paper2item[pair[0].strip()] = pair[1].strip()
    print('load # papers: %d'%len(paper2item))
    return paper2item

print("loading paper_year")
paper_year = load_paper2item(file_isi_paper_year)
print("original paper year2: ", len(paper_year))
paper_year2 = dict((k, int(v)) for k, v in paper_year.items())
print("original paper year: ", len(paper_year))
paper_year = dict((k, int(v)) for k, v in paper_year.items() if int(v) >= START_YEAR and int(v) <= END_YEAR)
print("valid paper year: ", len(paper_year))


paper_terms = {}
for term in term_paper_list:
    paper_list = term_paper_list[term]
    test_term = '_'.join(term.split())
    for paper in paper_list:
        if paper in paper_terms:
            paper_terms[paper].append(test_term)
        else:
            paper_terms[paper] = [test_term]
print("loading paper terms")
print("paper num", len(paper_terms))  


# new approach, saving while running
term_avg_sim = [-1234] * (len(term_list) * (END_YEAR-START_YEAR+1))
wordvecs_list = [0.0] * (len(term_list) * (END_YEAR-START_YEAR+1))
count_missing = 0
count_pair_missing = 0


dir = '/dfs/scratch0/hanchcao/data/isi_random/'
BURN_IN_YEAR = 1992
START_YEAR = 1992 
END_YEAR = 2016
file_isi_term_paperlist = dir + 'ISI_term_paperlist_full_corpus_lemmitized_loose_corrected_final.txt'


def load_paper2list2(filename):
    paper2list = defaultdict(list)
    with open(filename) as f:
        for line in f:
            line = line.strip().split('\t')
            paper2list[line[0].strip()] = list(map(lambda x: x, line[1:]))
    print('load # papers: %d'%len(paper2list))
    return paper2list


def load_paperlist2paperneighbors(term_paper_list):
    paper2list = {}
    for term in term_paper_list:
        paper_list = term_paper_list[term]
        for paper in paper_list:
            if paper in paper2list:
                paper2list[paper].append(term)
            else:
                paper2list[paper] = [term]
    return paper2list


print("loading term_paper_list")
term_paper_list = load_paper2list2(file_isi_term_paperlist)
print("# terms:", len(term_paper_list))
print("loading paper_neighbor_list")
paper_neighbor_list = load_paperlist2paperneighbors(term_paper_list)


def process(term_list, start_idx, end_idx, count_pair_missing, file, paper_neighbor_list):
    for idx in range(start_idx, end_idx, 1):
        ### 1. Compute avg closeness to all terms 
        test_term = term_list[idx]
        paper_list = term_paper_list[test_term]
        test_term =  '_'.join(test_term.split())
        term_avg_sim_temp = [-1234] * ((END_YEAR-START_YEAR+1))
        neighbor_freq_list = [{}]*(END_YEAR - START_YEAR + 1)
        term_neighbor_year_list = defaultdict(list)
        for paper in paper_list:
            paper = paper.strip()
            if paper in paper_year and paper in paper_neighbor_list:
                neighbors = paper_neighbor_list[paper]
                term_neighbor_year_list[int(paper_year[paper]) - START_YEAR].extend(neighbors)
        
        for year in term_neighbor_year_list:
            neighbor_freq_list[year] = {}
            for neighbor in term_neighbor_year_list[year]:
                if neighbor not in neighbor_freq_list[year]:
                    neighbor_freq_list[year][neighbor] = 1
                else:
                    neighbor_freq_list[year][neighbor] += 1
            count_all = 0.0
            sim_2_all = 0.0
            sim_2_all_list = []
            ninty_pct_neighbors = len(term_neighbor_year_list[year])*0.9  
            year_terms = set(wvs[year].wv.vocab.keys())
            year_neighbors_dict = neighbor_freq_list[year] 
            sort_orders = sorted(year_neighbors_dict.items(), key=lambda x: x[1], reverse=True)
            year_neighbor_list = []
            cnt = 0
            for i in sort_orders:
                cnt += i[1]
                if cnt <= ninty_pct_neighbors:
                    year_neighbor_list.append(i[0])
            year_neighbor_list_connect = []
            for neighbor in year_neighbor_list:
                term = '_'.join(neighbor.split())
                year_neighbor_list_connect.append(term)
            year_neighbors = set(year_neighbor_list_connect)
            random_neighbors = list(year_terms.intersection(year_neighbors))                     

            for i in range(len(random_neighbors)):
                for j in range(i+1, len(random_neighbors)):
                    term1 = random_neighbors[i]
                    term2 = random_neighbors[j]
                    try:
                        sim_score = wvs[year].similarity(term1, term2)
                        sim_2_all += float(sim_score)
                        count_all += 1.0
                        sim_2_all_list.append(sim_score)
                    except:
                        count_pair_missing += 1.0
            try:
                term_avg_sim_temp[year] = np.median(sim_2_all_list)
            except:
                print ('error')
                pass                

        try:
            f = open(file,'a')
            f.write(term_list[idx] + '\t')
            for year in range(START_YEAR, END_YEAR+1, 1):
                f.write(str(term_avg_sim_temp[year-START_YEAR])+' ')
            f.write('\n')
            f.close()
        except:
            pass
        print('processing #idx:%d, %s'%(idx, test_term))


numOfProcesses =20
termsPerProc = int(math.floor(len(term_list) * 1.0/numOfProcesses))
mgr = Manager()
        
output_file = []
for i in range(numOfProcesses):
    output_file.append(f_term_semnet_features+str(i)+'.txt')
        
jobs = []
for i in range(numOfProcesses):
    if i == numOfProcesses - 1:
        p = Process(target=process,
                    args=(term_list,i*termsPerProc,len(term_list),count_pair_missing,output_file[i],paper_neighbor_list))
    else:
        p = Process(target=process, 
                   args=(term_list, i*termsPerProc, (i+1)*termsPerProc, count_pair_missing, output_file[i],paper_neighbor_list))
    jobs.append(p)
    p.start()
    
for proc in jobs:
    proc.join()


dir = '/dfs/scratch0/hanchcao/'
fname = dir +'Autophrase_update/AutoPhrase/models/ISI_Lemmitized/segmentation_strict_corrected/segmentation_strict_more_corrected/segmentation_strict_more_corrected_freq_threshold/segmentation_strict_final'
fnamw = dir
+'Autophrase_update/AutoPhrase/models/ISI_Lemmitized/segmentation_loose_corrected/segmentation_loose_more_corrected/segmentation_loose_more_corrected_freq_threshold/segmentation_loose_final'

print("# terms:", len(term_feature_all))
print("done!")

