######################################################################
######################################################################
####code for "Modeling the Growth of Scientific Concepts" ############
####code for generating social conditions + controls #################
####Correspondence:###################################################
####Daniel A. McFarland###############################################
####e-mail: mcfarland@stanford.edu####################################
######################################################################
######################################################################


#!/usr/bin/env python
# coding: utf-8

#loading library
import sys, operator, math
from operator import add
from collections import defaultdict
from math import *
import random
import pickle
import itertools
from multiprocessing import Process, Manager
from numpy import median
from scipy import spatial
from scipy.stats.stats import pearsonr
import numpy as np


#define helper function
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


def load_paper2item(filename):
    paper2item = defaultdict(str)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if pair[1].strip() != '':
                paper2item[pair[0].strip()] = pair[1].strip()
    print('load # papers: %d'%len(paper2item))
    return paper2item


def load_paper2list(filename):
    paper2list = defaultdict(list)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if len(pair) == 2:
                paper2list[pair[0].strip()] = pair[1].strip().split(' ')
    print('load # papers: %d'%len(paper2list))
    return paper2list 

def load_paper2list_sub(filename):
    paper2list = defaultdict(list)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if len(pair) == 2:
                paper2list[pair[0].strip()] = [pair[1].strip()]
    print('load # papers: %d'%len(paper2list))
    return paper2list 


def load_paper2list2(filename):
    paper2list = defaultdict(list)
    with open(filename) as f:
        for line in f:
            line = line.strip().split('\t')
            paper2list[line[0].strip()] = list(map(lambda x: x, line[1:]))
    print('load # papers: %d'%len(paper2list))
    return paper2list

def load_paper2list3(filename):
    paper2list = defaultdict(list)
    with open(filename) as f:
        for line in f:
            line = line.strip().split('\t')
            paper2list[line[0].strip()] = list(map(lambda x: float(x), line[1:]))
    print('load # papers: %d'%len(paper2list))
    return paper2list


def load_paper2list_reverse(filename):
    item2paperlist = defaultdict(list)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if len(pair) == 2:
                item2paperlist[pair[1][:-1].strip()].append(pair[0].strip())
    print('load # items: %d'%len(item2paperlist))
    return item2paperlist


def load_paper2list_reverse2(filename):
    item2paperlist = defaultdict(list)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if len(pair) == 2:
                for item in pair[1].strip().split(' '):
                    item2paperlist[item.strip()].append(pair[0].strip())
    print('load # items: %d'%len(item2paperlist))
    return item2paperlist 


def load_total_df_year(filename):
    total_df_year = [0.0] * (END_YEAR - START_YEAR + 1)
    with open(filename) as f:
        for line in f:
            year, count = line.split('\t')
            if int(year) <= END_YEAR and int(year) >= START_YEAR:
                total_df_year[int(year) - START_YEAR] = float(count)
    return total_df_year


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


def load_coauthor_burnin(fname):
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


def load_author_burnin(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2: 
                pub_allyears = line[1].split(',')
                previous = list(map(lambda x: float(x), pub_allyears[:BURN_IN_YEAR - 1900]))
                if sum(previous) != 0.0:
                    continue
                else:
                    features[line[0].strip()] = list(map(lambda x: float(x), pub_allyears[START_YEAR - 1900:END_YEAR-1900+1]))
    print ('load # terms: %d', len(features))
    return features


def load_term_df_year(filename):
    term_df_year = defaultdict(list)
    with open(filename) as f:
        for line in f:
            entries = line.split('\t')
            df_raw = list(map(float, entries[1:]))
            df_raw = df_raw[START_YEAR-1913:END_YEAR-1913+1]
            term_df_year[entries[0].strip()] = df_raw
    print('load # terms: %d'%len(term_df_year))
    return term_df_year


def get_tf_per_doc(paper_term_dict):
    paper_tf_dict = {}
    for paper in paper_term_dict.keys():
        terms = paper_term_dict[paper]
        paper_tf_dict[paper] = defaultdict(float)
        for term in terms:
            paper_tf_dict[paper][term] += 1.0/len(terms)
    return paper_tf_dict


def get_first_year(term_df_year):
    term_list = []
    term_FIRSTYEAR = []
    term_firstyear_dict = {}
    for term in term_df_year:
        term_list.append(term)
        sentinal = False
        for idx in range(0, len(term_df_year[term]), 1):    
            if term_df_year[term][idx] != 0: 
                term_FIRSTYEAR.append(idx+START_YEAR)
                term_firstyear_dict[term] = idx+START_YEAR
                sentinal = True
                break
        if not sentinal:
            term_FIRSTYEAR.append(START_YEAR)
            term_firstyear_dict[term] = START_YEAR
    return term_list, term_FIRSTYEAR, term_firstyear_dict


def get_end_year(term_df_year):
    term_list = []
    term_ENDYEAR = []
    for term in term_df_year:
        term_list.append(term)
        sentinal = False
        for idx in range(len(term_df_year[term])-1, -1, -1):    
            if term_df_year[term][idx] != 0: 
                term_ENDYEAR.append(idx+START_YEAR)
                sentinal = True
                break
        if not sentinal:
            term_ENDYEAR.append(END_YEAR)
    return term_list, term_ENDYEAR


def get_term_subject_percent(paper_list, paper_year, paper_subject):
    term_interdisciplinary_year_count = [defaultdict(int) for i in range((END_YEAR - START_YEAR + 1))]
    physics = [0.0]*(END_YEAR - START_YEAR + 1)
    engineering = [0.0]*(END_YEAR - START_YEAR + 1)
    agriculture = [0.0]*(END_YEAR - START_YEAR + 1)
    bio = [0.0]*(END_YEAR - START_YEAR + 1)
    social_science = [0.0]*(END_YEAR - START_YEAR + 1)
    paper_year_cnt = [0.0]*(END_YEAR - START_YEAR + 1)
    physics_set = set(['Physical and Mathematical Sciences'])
    agr_set = set(['Agricultural Sciences'])
    engineering_set = set(['Engineering'])
    bio_set = set(['Biological and Health Sciences'])
    social_science_set = set(['Education', 'Humanities', 'Law', 'Social and Behavioral Sciences', 'Business'])
    for paper in paper_list:
        if paper in paper_subject and paper in paper_year and int(paper_year[paper]) >= START_YEAR and int(paper_year[paper]) <= END_YEAR:
            subjects = paper_subject[paper]
            subject =  subjects[0]
            paper_year_cnt[int(paper_year[paper])-START_YEAR] += 1.0
            if subject in physics_set:
                physics[int(paper_year[paper])-START_YEAR] += 1.0
            elif subject in agr_set:
                agriculture[int(paper_year[paper])-START_YEAR] += 1.0
            elif subject in engineering_set:
                engineering[int(paper_year[paper])-START_YEAR] += 1.0
            elif subject in bio_set:
                bio[int(paper_year[paper])-START_YEAR] += 1.0
            elif subject in social_science_set:
                social_science[int(paper_year[paper])-START_YEAR] += 1.0
    pct_physics = [i*1.0 / (j + 1e-10) for i, j in zip(physics, paper_year_cnt)] 
    pct_engineering = [i*1.0 / (j + 1e-10) for i, j in zip(engineering, paper_year_cnt)] 
    pct_agriculture = [i*1.0 / (j + 1e-10) for i, j in zip(agriculture, paper_year_cnt)] 
    pct_bio = [i*1.0 / (j + 1e-10) for i, j in zip(bio, paper_year_cnt)] 
    pct_social_science = [i*1.0 / (j + 1e-10) for i, j in zip(social_science, paper_year_cnt)]  
    return pct_physics, pct_engineering, pct_agriculture, pct_bio, pct_social_science


def get_venue_cum_citations(paper_venue, paper_citations_by_year):
    venue_citations_by_year = defaultdict(list)
    venue_paperList = defaultdict(list)
    for paper in paper_venue:
        venue_paperList[paper_venue[paper]].append(paper)
    for venue in venue_paperList:
        venue_citations_by_year[venue] = [0.0] * (END_YEAR-START_YEAR+1)
        for paper in venue_paperList[venue]:
            if paper in paper_citations_by_year:
                venue_citations_by_year[venue] = [sum(x) for x in zip(venue_citations_by_year[venue], paper_citations_by_year[paper])]
    return venue_citations_by_year


def get_term_interdisciplinary(paper_list, paper_year, paper_subject):
    term_interdisciplinary_year_count = [defaultdict(int) for i in range((END_YEAR - START_YEAR + 1))]
    entropy = [0.0]*(END_YEAR - START_YEAR + 1)
    for paper in paper_list:
        if paper in paper_subject and paper in paper_year and int(paper_year[paper]) >= START_YEAR and int(paper_year[paper]) <= END_YEAR:
            subjects = paper_subject[paper]
            sub_number = len(subjects)
            for subject in subjects:
                term_interdisciplinary_year_count[int(paper_year[paper])-START_YEAR][subject] += 1.0/sub_number
                entropy[int(paper_year[paper])-START_YEAR] += 1.0/sub_number
    for i in range(len(entropy)):
        e = 0.0
        c = entropy[i] + 1e-8
        for subject in term_interdisciplinary_year_count[i]:
            e -= term_interdisciplinary_year_count[i][subject]/c * log(term_interdisciplinary_year_count[i][subject]/c)
        entropy[i] = e
    return entropy


def get_term_female_percent(paper_list, paper_author_list, paper_year, term_start_year, term_end_year, author_gender):
    term_author_year_list = defaultdict(list)
    author_female_list = [0.0]*(END_YEAR - START_YEAR + 1)
    for paper in paper_list:
        paper = paper.strip()
        if paper in paper_year and paper in paper_author_list:
            for author in paper_author_list[paper]:
                term_author_year_list[int(paper_year[paper] - START_YEAR)].append(author)
    for i in range(0, term_end_year - term_start_year + 1, 1):
        females = 0
        males = 0
        total = 0
        for author in term_author_year_list[i + term_start_year - START_YEAR]:
            if author_gender[author] == 'female':
                females = females + 1
                total = total + 1
            if author_gender[author] == 'male':
                males = males + 1
                total = total + 1
        if total != 0:
            author_female_list[i + term_start_year - START_YEAR] = females*1.0 / total
        else:
            author_female_list[i + term_start_year - START_YEAR] = 0
    return author_female_list


def get_author_age(author_paper_list, paper_year):
    author_age = {}
    first_year = {}
    for author in author_paper_list:        
        first_year[author] = 10000
        for paper in author_paper_list[author]:
            if paper in paper_year:
                if paper_year[paper] < first_year[author]:
                    first_year[author] = paper_year[paper]
                                        
    for author in author_paper_list:
        year_age = [0.0] * (END_YEAR-START_YEAR+1)
        for paper in author_paper_list[author]:
            if paper in paper_year and paper_year[paper] >= START_YEAR and paper_year[paper] <= END_YEAR:
                year_age[paper_year[paper] - START_YEAR] = paper_year[paper] - first_year[author] + 1       
        author_age[author] = year_age
    return author_age


def get_term_authorlist2(paper_list, paper_author_list, paper_year, start_year, end_year):
    ### input: a term's list of papers
    term_author_year_list = defaultdict(list)
    for paper in paper_list:
        paper = paper.strip()
        if paper in paper_year and paper in paper_author_list:
            for author in paper_author_list[paper]:
                term_author_year_list[int(paper_year[paper] - START_YEAR)].append(author)
    return term_author_year_list


def get_term_authorsizelist(paper_list, paper_author_list, paper_year):
    ### input: a term's list of papers
    author_size_list = [0.0]*(END_YEAR - START_YEAR + 1)
    author_size_set = [0.0]*(END_YEAR - START_YEAR + 1)
    term_author_year_list = defaultdict(list)
    for paper in paper_list:
        paper = paper.strip()
        if paper in paper_year and paper in paper_author_list:
            for author in paper_author_list[paper]:
                term_author_year_list[int(paper_year[paper] - START_YEAR)].append(author)
    for year in term_author_year_list:
        author_size_list[year] = len(term_author_year_list[year])
        author_size_set[year] = len(set(term_author_year_list[year]))
    return author_size_set, author_size_list


def get_term_avg_author_pub(paper_list, paper_author_list, paper_year, author_yearly_pub, cooccur_cutoff = 10):
    ### input: a term's list of papers
    author_freq_list = [{}]*(END_YEAR - START_YEAR + 1)
    author_avg_pub_list = [0.0]*(END_YEAR - START_YEAR + 1)
    author_sum_pub_list = [0.0]*(END_YEAR - START_YEAR + 1)
    author_mid_pub_list = [0.0]*(END_YEAR - START_YEAR + 1)
    term_author_year_list = defaultdict(list)
    for paper in paper_list:
        paper = paper.strip()
        if paper in paper_year and paper in paper_author_list:
            for author in paper_author_list[paper]:
                term_author_year_list[int(paper_year[paper] - START_YEAR)].append(author)
    for year in term_author_year_list:
        author_freq_list[year] = {}
        for author in term_author_year_list[year]:
            if author not in author_freq_list[year]:
                author_freq_list[year][author] = 1
            else:
                author_freq_list[year][author] += 1
        
        author_set = set(term_author_year_list[year])
        total_freq = 0
        all_weightfreq = 0
        author_weight_list = []
        for author in author_set:
            if author in author_yearly_pub and year in author_yearly_pub[author] and author in author_freq_list[year]:
                weight = author_freq_list[year][author]
                freq = author_yearly_pub[author][year]
                author_weight_list.extend([weight]*int(freq))
                total_freq += freq
                all_weightfreq += weight*freq
                
        if total_freq != 0:
            author_avg_pub_list[year] = all_weightfreq*1.0/total_freq
            author_mid_pub_list[year] = median(author_weight_list)
            author_sum_pub_list[year] = all_weightfreq
    return author_freq_list, author_avg_pub_list, author_sum_pub_list, author_mid_pub_list


def get_term_new_neighbor_wgt_pct_year(paper_list, paper_neighbor_list, paper_year, term_firstyear_dict):
    ### input: a term's list of papers:
    neighbor_freq_list = [{}]*(END_YEAR - START_YEAR + 1)
    term_new_neighbor_pct_year = [0.0]*(END_YEAR - START_YEAR + 1)
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
        neighbor_set = set(term_neighbor_year_list[year])
        wgt = 0 
        for neighbor in neighbor_set:
            if neighbor in term_firstyear_dict:
                neighbor_startyear = int(term_firstyear_dict[neighbor])
                real_year = year + START_YEAR
                if neighbor_startyear == real_year:
                    weight = neighbor_freq_list[year][neighbor]
                    wgt = wgt + weight
        total_neighbor = len(term_neighbor_year_list[year])                                    
        if total_neighbor != 0:
            term_new_neighbor_pct_year[year] = wgt*1.0/total_neighbor
    return term_new_neighbor_pct_year


def get_term_new_neighbor_avg_pct_year(paper_list, paper_neighbor_list, paper_year, term_firstyear_dict):
    ### input: a term's list of papers:
    term_new_neighbor_pct_year = [0.0]*(END_YEAR - START_YEAR + 1)
    term_neighbor_year_list = defaultdict(list)
    for paper in paper_list:
        paper = paper.strip()
        if paper in paper_year and paper in paper_neighbor_list:
            neighbors = paper_neighbor_list[paper]
            term_neighbor_year_list[int(paper_year[paper]) - START_YEAR].extend(neighbors)
    for year in term_neighbor_year_list:
        neighbor_set = set(term_neighbor_year_list[year])
        cnt = 0 
        for neighbor in neighbor_set:
            if neighbor in term_firstyear_dict:
                neighbor_startyear = int(term_firstyear_dict[neighbor])
                real_year = year + START_YEAR
                if neighbor_startyear == real_year:
                    cnt += 1
        total_neighbor = len(neighbor_set)                                    
        if total_neighbor != 0:
            term_new_neighbor_pct_year[year] = cnt*1.0/total_neighbor
    return term_new_neighbor_pct_year


def get_term_avg_pop_neighbor(paper_list, paper_neighbor_list, paper_year, term_df_year, cooccur_cutoff = 30):
    ### input: a term's list of papers:
    neighbor_freq_list = [{}]*(END_YEAR - START_YEAR + 1)
    neighbor_avg_pop_list = [0.0]*(END_YEAR - START_YEAR + 1)
    neighbor_sum_pop_list = [0.0]*(END_YEAR - START_YEAR + 1)
    neighbor_mid_pop_list = [0.0]*(END_YEAR - START_YEAR + 1)
    
    #neighbor_size_set = [{}]*(END_YEAR - START_YEAR + 1)
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
        neighbor_set = set(term_neighbor_year_list[year])
        total_freq = 0
        all_weightfreq = 0
        neighbor_weight_list = []
        for neighbor in neighbor_set:
            if  neighbor_freq_list[year][neighbor] > cooccur_cutoff:
                neighbor_freq_list[year].pop(neighbor)
            if neighbor in term_df_year and neighbor in neighbor_freq_list[year]:
                weight = neighbor_freq_list[year][neighbor]
                freq = term_df_year[neighbor][year]
                neighbor_weight_list.extend([weight]*int(freq))
                total_freq += freq
                all_weightfreq += weight*freq
                              
        if total_freq != 0:
            neighbor_avg_pop_list[year] = all_weightfreq*1.0/total_freq
            neighbor_mid_pop_list[year] = median(neighbor_weight_list)
            neighbor_sum_pop_list[year] = all_weightfreq

    return neighbor_freq_list, neighbor_avg_pop_list, neighbor_sum_pop_list, neighbor_mid_pop_list

       
def get_term_repeated_use_year_v2(paper_list, paper_author_list, paper_year, term_start_year, term_end_year):
    term_author_year_list = defaultdict(list)
    repeated_use_start = [-1234]*(END_YEAR - START_YEAR + 1)
    repeated_use_preceding_1 = [-1234]*(END_YEAR - START_YEAR + 1)
    term_paper_year_list = defaultdict(list)
    
    for paper in paper_list:
        paper = paper.strip()
        if paper in paper_year and paper in paper_author_list:
            for author in paper_author_list[paper]:
                term_author_year_list[int(paper_year[paper] - START_YEAR)].append(author)
    
    for i in range(0, term_end_year - term_start_year + 1, 1):
        current_year_author_freq = defaultdict(int)
        for author in term_author_year_list[i + term_start_year - START_YEAR]:
            current_year_author_freq[author] += 1
        start_year_author_freq = defaultdict(int)
        for author in term_author_year_list[term_start_year - START_YEAR]:
            start_year_author_freq[author] += 1
        start_list = []
        current_list = []
        for key in start_year_author_freq:
            start_list.append(start_year_author_freq[key])
            if key in current_year_author_freq:
                current_list.append(current_year_author_freq[key])
            else:
                current_list.append(0)
        
        if len(current_list) != 0 and sum(start_list) != 0 and i != 0:
            current_list_array = np.array(current_list)
            start_list_array = np.array(start_list)
            array_length = len(current_list_array)
            sum_sq =np.sum(np.square(current_list_array - start_list_array)*1.0/array_length)
            repeated_use_start[i + term_start_year - START_YEAR] = 1 - spatial.distance.cosine(current_list, start_list)
        else:
            repeated_use_start[i + term_start_year - START_YEAR] = 0
    
    for i in range(0, term_end_year - term_start_year + 1, 1):
        current_year_author_freq = defaultdict(int)
        for author in term_author_year_list[i + term_start_year - START_YEAR]:
            current_year_author_freq[author] += 1
        preceding_year_author_freq = defaultdict(int)
        for author in term_author_year_list[i - 1 + term_start_year - START_YEAR]:
            preceding_year_author_freq[author] += 1
        preceding_list = []
        current_list = []
        for key in preceding_year_author_freq:
            preceding_list.append(preceding_year_author_freq[key])
            if key in current_year_author_freq:
                current_list.append(current_year_author_freq[key])
            else:
                current_list.append(0)

        if len(current_list) != 0 and sum(preceding_list) != 0 and i != 0:
            current_list_array = np.array(current_list)
            preceding_list_array = np.array(preceding_list)
            array_length = len(current_list_array)
            sum_sq =np.sum(np.square(current_list_array - preceding_list_array)*1.0/array_length)
            repeated_use_preceding_1[i + term_start_year - START_YEAR] = 1 - spatial.distance.cosine(current_list, preceding_list)
        else:
            repeated_use_preceding_1[i + term_start_year - START_YEAR] = 0
    return repeated_use_start, repeated_use_preceding_1

    
def get_term_repeated_neighbor_year_v2(paper_list, paper_neighbor_list, paper_year, term_start_year, term_end_year):
    term_neighbor_year_list = defaultdict(list)
    repeated_use_start = [-1234]*(END_YEAR - START_YEAR + 1)
    repeated_use_preceding_1 = [-1234]*(END_YEAR - START_YEAR + 1)
    term_paper_year_list = defaultdict(list)
    
    for paper in paper_list:
        paper = paper.strip()
        if paper in paper_year and paper in paper_neighbor_list:
            neighbors = paper_neighbor_list[paper]
            term_neighbor_year_list[int(paper_year[paper] - START_YEAR)].extend(neighbors)
    
    for i in range(0, term_end_year - term_start_year + 1, 1):
        current_year_neighbor_freq = defaultdict(int)
        for neighbor in term_neighbor_year_list[i + term_start_year - START_YEAR]:
            current_year_neighbor_freq[neighbor] += 1
        start_year_neighbor_freq = defaultdict(int)
        for neighbor in term_neighbor_year_list[term_start_year - START_YEAR]:
            start_year_neighbor_freq[neighbor] += 1
        
        start_list = []
        current_list = []
        for key in start_year_neighbor_freq:
            start_list.append(start_year_neighbor_freq[key])
            if key in current_year_neighbor_freq:
                current_list.append(current_year_neighbor_freq[key])
            else:
                current_list.append(0)
                
        if len(current_list) != 0 and sum(start_list) != 0 and i != 0:
            current_list_array = np.array(current_list)
            start_list_array = np.array(start_list)
            array_length = len(current_list_array)
            sum_sq =np.sum(np.square(current_list_array - start_list_array)*1.0/array_length)
            repeated_use_start[i + term_start_year - START_YEAR] = 1 - spatial.distance.cosine(current_list, start_list)
        else:
            repeated_use_start[i + term_start_year - START_YEAR] = 0
    
    for i in range(0, term_end_year - term_start_year + 1, 1):
        current_year_neighbor_freq = defaultdict(int)
        for neighbor in term_neighbor_year_list[i + term_start_year - START_YEAR]:
            current_year_neighbor_freq[neighbor] += 1
        preceding_year_neighbor_freq = defaultdict(int)
        for neighbor in term_neighbor_year_list[i - 1 + term_start_year - START_YEAR]:
            preceding_year_neighbor_freq[neighbor] += 1
        preceding_list = []
        current_list = []
        for key in preceding_year_neighbor_freq:
            preceding_list.append(preceding_year_neighbor_freq[key])
            if key in current_year_neighbor_freq:
                current_list.append(current_year_neighbor_freq[key])
            else:
                current_list.append(0)
        
        if len(current_list) != 0 and sum(preceding_list) != 0 and i != 0:
            current_list_array = np.array(current_list)
            preceding_list_array = np.array(preceding_list)
            array_length = len(current_list_array)
            sum_sq =np.sum(np.square(current_list_array - preceding_list_array)*1.0/array_length)
            repeated_use_preceding_1[i + term_start_year - START_YEAR] = 1 - spatial.distance.cosine(current_list, preceding_list)
        else:
            repeated_use_preceding_1[i + term_start_year - START_YEAR] = 0
    return repeated_use_start, repeated_use_preceding_1


def get_term_new_author_pct_year(paper_list, paper_author_list, author_age, paper_year, start_year, end_year, new_cutoff = 1):
    term_new_author_year_count = [0.0]*(END_YEAR - START_YEAR + 1)
    ### this func is wgted avg version
    term_author_year_list = get_term_authorlist2(paper_list, paper_author_list, paper_year, start_year, end_year)    
    author_size_set, author_size_list = get_term_authorsizelist(paper_list, paper_author_list, paper_year)
    term_new_author_pct_year = [0.0]*(END_YEAR - START_YEAR + 1)
    for year in term_author_year_list:
        author_list = term_author_year_list[year]
        for author in author_list:
            if author in author_age:
                if author_age[author][year] <= new_cutoff:
                    term_new_author_year_count[year] +=1
    for year in term_author_year_list:
        if author_size_list[year] != 0:
            term_new_author_pct_year[year] = term_new_author_year_count[year]*1.0/author_size_list[year]
        else:
            term_new_author_pct_year[year] = term_new_author_year_count[year]
    return term_new_author_pct_year


def get_term_new_author_pct_year_avg(paper_list, paper_author_list, author_age, paper_year, start_year, end_year, new_cutoff = 1):
    term_new_author_year_count = [0.0]*(END_YEAR - START_YEAR + 1)
    ### this func is avg version, repeated author is only cnted once
    term_author_year_list = get_term_authorlist2(paper_list, paper_author_list, paper_year, start_year, end_year)    
    author_size_set, author_size_list = get_term_authorsizelist(paper_list, paper_author_list, paper_year)
    term_new_author_pct_year = [0.0]*(END_YEAR - START_YEAR + 1)
    for year in term_author_year_list:
        author_list = list(set(term_author_year_list[year]))
        for author in author_list:
            if author in author_age:
                if author_age[author][year] <= new_cutoff:
                    term_new_author_year_count[year] +=1
    for year in term_author_year_list:
        if author_size_set[year] != 0:
            term_new_author_pct_year[year] = term_new_author_year_count[year]*1.0/author_size_set[year]
        else:
            term_new_author_pct_year[year] = term_new_author_year_count[year]
    return term_new_author_pct_year


def get_term_author_avg_pub_year(paper_list, paper_author_list, paper_year, author_yearly_pub):
    author_freq_list, author_avg_pub_list, author_sum_pub_list, author_mid_pub_list = get_term_avg_author_pub(paper_list, paper_author_list, paper_year, author_yearly_pub)
    return author_avg_pub_list, author_sum_pub_list, author_mid_pub_list

def get_term_neighbor_avg_pop_year(paper_list, paper_neighbor_list, paper_year, term_df_year):
    term_neighbor_size_year_count = [0.0]*(END_YEAR - START_YEAR + 1)
    neighbor_freq_list, neighbor_avg_pop_list, neighbor_sum_pop_list, neighbor_mid_pop_list = get_term_avg_pop_neighbor(paper_list, paper_neighbor_list, paper_year, term_df_year, 30)
    return neighbor_avg_pop_list, neighbor_sum_pop_list, neighbor_mid_pop_list 

 
### compute term's features of different modes    
def get_term_citation_year(paper_list, citation_year_count, paper_year, start_year, end_year):
    term_citation_year_count = [0.0]*(END_YEAR - START_YEAR + 1)
    for paper in paper_list:
        if paper in citation_year_count:
            term_citation_year_count = list(map(add, term_citation_year_count, citation_year_count[paper]))
    return term_citation_year_count
    
def get_term_venue_citation_year(paper_list, paper_venue, venue_year_citation, paper_year, start_year, end_year):
    venue_useCnt_by_year = defaultdict(list)
    for paper in paper_list:
        venue = paper_venue[paper]
        if paper in paper_year and paper_year[paper] and paper in paper_venue:            
            if int(paper_year[paper]) < START_YEAR:
                add_on = [0.0] * (END_YEAR-START_YEAR+1)
            elif int(paper_year[paper]) > END_YEAR:
                add_on = [0.0] * (END_YEAR-START_YEAR+1)
            else:
                add_on = [0.0] * (int(paper_year[paper])-START_YEAR) + [1.0] * 1 + [0.0] * (END_YEAR-int(paper_year[paper]))
            if venue not in venue_useCnt_by_year:
                venue_useCnt_by_year[venue] = [0.0] * (END_YEAR-START_YEAR+1)
            venue_useCnt_by_year[venue] = [sum(x) for x in zip(venue_useCnt_by_year[venue], add_on)]
    
    venueCitations = [0.0] * (END_YEAR-START_YEAR+1)
    for i in range(END_YEAR-START_YEAR+1):
        total_cnt = 0.0
        for venue in venue_useCnt_by_year:
            if venue not in venue_year_citation or venue not in venue_useCnt_by_year:
                continue
            venueCitations[i] += venue_year_citation[venue][i] * venue_useCnt_by_year[venue][i]
            total_cnt += venue_useCnt_by_year[venue][i]
        venueCitations[i] = venueCitations[i] / (total_cnt+1e-8)
    return venueCitations

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

def get_term_author_size_year(paper_list, paper_author_list, paper_year):
    term_author_size_year_count = [0.0]*(END_YEAR - START_YEAR + 1)
    author_size_set, author_size_list = get_term_authorsizelist(paper_list, paper_author_list, paper_year)
    return author_size_set


def feature_worker(terms, term_df_year, total_df_year, term_startyears, term_endyears, paper_year, paper_subject, paper_subject2, term_paper_list, paper_venue, citation_year_count, venue_year_count,  author_age, paper_author_list, author_paper_list, author_gender, term_coauthor_year, author_yearly_pub, term_firstyear_dict, term_feature_all, file): 
    term_feature = []
    for i in range(len(terms)):
        term = terms[i]
        all_features = []
        term_end_year = 2016
        term_start_year = term_startyears[i]
        term_citation_year = get_term_citation_year(term_paper_list[term], citation_year_count, paper_year, term_start_year, term_end_year)
        term_venue_citation_year = get_term_venue_citation_year(term_paper_list[term], paper_venue, venue_year_count, paper_year, term_start_year, term_end_year)
        entropy = get_term_interdisciplinary(term_paper_list[term], paper_year, paper_subject)
        entropy2 = get_term_interdisciplinary(term_paper_list[term], paper_year, paper_subject2)
        term_new_author_pct = get_term_new_author_pct_year(term_paper_list[term], paper_author_list, author_age, paper_year, term_start_year, term_end_year)
        term_new_author_pct_avg = get_term_new_author_pct_year_avg(term_paper_list[term], paper_author_list, author_age, paper_year, term_start_year, term_end_year)
        term_new_neighbor_wgt_pct = get_term_new_neighbor_wgt_pct_year(term_paper_list[term], paper_neighbor_list, paper_year, term_firstyear_dict)
        term_new_neighbor_avg_pct = get_term_new_neighbor_avg_pct_year(term_paper_list[term], paper_neighbor_list, paper_year, term_firstyear_dict)
       
        term_author_size = get_term_author_size_year(term_paper_list[term], paper_author_list, paper_year)
        term_author_avg_pub, term_author_sum_pub, term_author_mid_pub = get_term_author_avg_pub_year(term_paper_list[term], paper_author_list, paper_year, author_yearly_pub)
        repeated_use_start, repeated_use_preceding_1 = get_term_repeated_use_year_v2(term_paper_list[term], paper_author_list, paper_year, term_start_year, term_end_year)
        
        term_neighbor_avg_pop, term_neighbor_sum_pop, term_neighbor_mid_pop = get_term_neighbor_avg_pop_year(term_paper_list[term], paper_neighbor_list, paper_year, term_df_year)
        neighbor_repeated_use_start, neighbor_repeated_use_preceding_1 = get_term_repeated_neighbor_year_v2(term_paper_list[term], paper_neighbor_list, paper_year, term_start_year, term_end_year)

        term_author_female = get_term_female_percent(term_paper_list[term], paper_author_list, paper_year, term_start_year, term_end_year,author_gender)
        
        term_pct_physics, term_pct_engineering, term_pct_agriculture, term_pct_bio, term_pct_social_science = get_term_subject_percent(term_paper_list[term], paper_year, paper_subject) 
        for j in range(0, END_YEAR - START_YEAR + 1):
            term_current_year = START_YEAR + j
            features = defaultdict(float)
            
            avg_venuecitation_start, avg_venuecitation_sum, avg_venuecitation_avg, avg_venuecitation_cur  = get_term_citation_feature(term_venue_citation_year, term_start_year, term_end_year, term_current_year)
            avg_venuecitation_preceding_1, avg_venuecitation_cr_1, avg_venuecitation_firstdif = get_term_citation_preceeding_feature(term_venue_citation_year, term_current_year, 1)
            features['avg_venue_citation_start'] = avg_venuecitation_start 
            features['avg_venue_citation_total'] = avg_venuecitation_sum 
            features['avg_venue_citation_preceding_1'] = avg_venuecitation_preceding_1 
            features['avg_venue_citation_first_diff'] = avg_venuecitation_firstdif
            features['avg_venue_citation_cur'] = avg_venuecitation_cur
            
            entropy_start, entropy_sum, entropy_avg, entropy_cur = get_term_citation_feature(entropy, term_start_year, term_end_year, term_current_year)
            entropy_preceding_1, entropy_cr_1, entropy_firstdif = get_term_citation_preceeding_feature(entropy, term_current_year, 1)
            features['avg_entropy_start'] = entropy_start
            features['avg_entropy_total'] = entropy_sum
            features['avg_entropy_preceding_1'] = entropy_preceding_1 
            features['avg_entropy_first_diff'] = entropy_firstdif
            features['avg_entropy_cur'] = entropy_cur
            
            entropy_s_start, entropy_s_sum, entropy_s_avg, entropy_s_cur = get_term_citation_feature(entropy2, term_start_year, term_end_year, term_current_year)
            entropy_s_preceding_1, entropy_s_cr_1, entropy_s_firstdif = get_term_citation_preceeding_feature(entropy2, term_current_year, 1)
            features['avg_entropy_s_start'] = entropy_s_start
            features['avg_entropy_s_total'] = entropy_s_sum
            features['avg_entropy_s_preceding_1'] = entropy_s_preceding_1 
            features['avg_entropy_s_first_diff'] = entropy_s_firstdif
            features['avg_entropy_s_cur'] =  entropy_s_cur
                        
            age_start, age_sum, age_avg, age_cur = get_term_citation_feature(term_new_author_pct, term_start_year, term_end_year, term_current_year)
            age_preceding_1, age_cr_1, age_firstdif = get_term_citation_preceeding_feature(term_new_author_pct, term_current_year, 1)        
            features['new_author_pct_start'] = age_start 
            features['new_author_pct_total'] = age_sum 
            features['new_author_pct_preceding_1'] = age_preceding_1 
            features['new_author_pct_first_diff'] = age_firstdif
            features['new_author_pct_cur'] = age_cur
            
            age_avg_start, age_avg_sum, age_avg_avg, age_avg_cur = get_term_citation_feature(term_new_author_pct_avg, term_start_year, term_end_year, term_current_year)
            age_avg_preceding_1, age_avg_cr_1, age_avg_firstdif = get_term_citation_preceeding_feature(term_new_author_pct_avg, term_current_year, 1)        
            features['new_author_pct_avg_start'] = age_avg_start 
            features['new_author_pct_avg_total'] = age_avg_sum 
            features['new_author_pct_avg_preceding_1'] = age_avg_preceding_1 
            features['new_author_pct_avg_first_diff'] = age_avg_firstdif
            features['new_author_pct_avg_cur'] = age_avg_cur
            
            newneighbor_start, newneighbor_sum, newneighbor_avg, newneighbor_cur = get_term_citation_feature(term_new_neighbor_wgt_pct, term_start_year, term_end_year, term_current_year)
            newneighbor_preceding_1, newneighbor_cr_1, newneighbor_firstdif = get_term_citation_preceeding_feature(term_new_neighbor_wgt_pct, term_current_year, 1)        
            features['new_neighbor_wgt_pct_start'] = newneighbor_start 
            features['new_neighbor_wgt_pct_total'] = newneighbor_sum 
            features['new_neighbor_wgt_pct_preceding_1'] = newneighbor_preceding_1 
            features['new_neighbor_wgt_pct_first_diff'] = newneighbor_firstdif
            features['new_neighbor_wgt_pct_cur'] = newneighbor_cur


            newneighbor_avg_start, newneighbor_avg_sum, newneighbor_avg_avg, newneighbor_avg_cur = get_term_citation_feature(term_new_neighbor_avg_pct, term_start_year, term_end_year, term_current_year)
            newneighbor_avg_preceding_1, newneighbor_avg_cr_1, newneighbor_avg_firstdif = get_term_citation_preceeding_feature(term_new_neighbor_avg_pct, term_current_year, 1)        
            features['new_neighbor_avg_pct_start'] = newneighbor_avg_start 
            features['new_neighbor_avg_pct_total'] = newneighbor_avg_sum 
            features['new_neighbor_avg_pct_preceding_1'] = newneighbor_avg_preceding_1 
            features['new_neighbor_avg_pct_first_diff'] = newneighbor_avg_firstdif
            features['new_neighbor_avg_pct_cur'] = newneighbor_avg_cur            
            
            
            size_start, size_sum, size_avg, size_cur = get_term_citation_feature(term_author_size, term_start_year, term_end_year, term_current_year)
            size_preceding_1, size_cr_1, size_firstdif = get_term_citation_preceeding_feature(term_author_size, term_current_year, 1)        
            features['avg_author_size_start'] = size_start 
            features['avg_author_size_total'] = size_sum 
            features['avg_author_size_preceding_1'] = size_preceding_1 
            features['avg_author_size_first_diff'] = size_firstdif 
            features['avg_author_size_cur'] = size_cur 
            
            avg_size_start, avg_size_sum, avg_size_avg, avg_size_cur = get_term_citation_feature(term_author_avg_pub, term_start_year, term_end_year, term_current_year)
            avg_size_preceding_1, avg_size_cr_1, avg_size_firstdif = get_term_citation_preceeding_feature(term_author_avg_pub, term_current_year, 1)        
            features['avg_author_pub_num_start'] = avg_size_start 
            features['avg_author_pub_num_total'] = avg_size_sum 
            features['avg_author_pub_num_preceding_1'] = avg_size_preceding_1 
            features['avg_author_pub_num_first_diff'] = avg_size_firstdif 
            features['avg_author_pub_num_cur'] = avg_size_cur 
                        
            features['avg_author_ruse_start'] = repeated_use_start[j] 
            features['avg_author_ruse_preceding_1'] = repeated_use_preceding_1[j] 

            gender_start, gender_sum, gender_avg, gender_cur = get_term_citation_feature(term_author_female, term_start_year, term_end_year, term_current_year)
            gender_preceding_1, gender_cr_1, gender_firstdif = get_term_citation_preceeding_feature(term_author_female, term_current_year, 1)
            features['author_female_start'] = gender_start 
            features['author_female_total'] = gender_sum 
            features['author_female_preceding_1'] = gender_preceding_1 
            features['author_female_first_diff'] = gender_firstdif 
            features['author_female_cur'] = gender_cur 
            
            neighbor_size_start, neighbor_size_sum, neighbor_size_avg, neighbor_size_cur = get_term_citation_feature(term_neighbor_avg_pop, term_start_year, term_end_year, term_current_year)
            neighbor_size_preceding_1, neighbor_size_cr_1, neighbor_size_firstdif = get_term_citation_preceeding_feature(term_neighbor_avg_pop, term_current_year, 1)        
            features['avg_neighbor_pop_start'] = neighbor_size_start 
            features['avg_neighbor_pop_total'] = neighbor_size_sum 
            features['avg_neighbor_pop_preceding_1'] = neighbor_size_preceding_1 
            features['avg_neighbor_pop_first_diff'] = neighbor_size_firstdif 
            features['avg_neighbor_pop_cur'] = neighbor_size_cur

            sum_neighbor_size_start, sum_neighbor_size_sum, sum_neighbor_size_avg, sum_neighbor_size_cur = get_term_citation_feature(term_neighbor_sum_pop, term_start_year, term_end_year, term_current_year)
            sum_neighbor_size_preceding_1, sum_neighbor_size_cr_1, sum_neighbor_size_firstdif = get_term_citation_preceeding_feature(term_neighbor_sum_pop, term_current_year, 1)        
            features['sum_neighbor_pop_start'] = sum_neighbor_size_start 
            features['sum_neighbor_pop_total'] = sum_neighbor_size_sum 
            features['sum_neighbor_pop_preceding_1'] = sum_neighbor_size_preceding_1 
            features['sum_neighbor_pop_first_diff'] = sum_neighbor_size_firstdif 
            features['sum_neighbor_pop_cur'] = sum_neighbor_size_cur
 
            mid_neighbor_size_start, mid_neighbor_size_sum, mid_neighbor_size_avg, mid_neighbor_size_cur = get_term_citation_feature(term_neighbor_mid_pop, term_start_year, term_end_year, term_current_year)
            mid_neighbor_size_preceding_1, mid_neighbor_size_cr_1, mid_neighbor_size_firstdif = get_term_citation_preceeding_feature(term_neighbor_mid_pop, term_current_year, 1)        
            features['mid_neighbor_pop_start'] = mid_neighbor_size_start 
            features['mid_neighbor_pop_total'] = mid_neighbor_size_sum 
            features['mid_neighbor_pop_preceding_1'] = mid_neighbor_size_preceding_1 
            features['mid_neighbor_pop_first_diff'] = mid_neighbor_size_firstdif 
            features['mid_neighbor_pop_cur'] = mid_neighbor_size_cur
            
            features['neighbor_ruse_start'] = neighbor_repeated_use_start[j] 
            features['neighbor_ruse_preceding_1'] = neighbor_repeated_use_preceding_1[j] 

            physics_pct_start, physics_pct_sum, physics_pct_avg, physics_pct_cur = get_term_citation_feature(term_pct_physics, term_start_year, term_end_year, term_current_year)
            physics_pct_preceding_1, physics_pct_cr_1, physics_pct_firstdif = get_term_citation_preceeding_feature(term_pct_physics, term_current_year, 1)        
            features['avg_physics_pct_start'] = physics_pct_start 
            features['avg_physics_pct_total'] = physics_pct_sum 
            features['avg_physics_pct_preceding_1'] = physics_pct_preceding_1 
            features['avg_physics_pct_first_diff'] = physics_pct_firstdif
            features['avg_physics_pct_cur'] = physics_pct_cur

            engineering_pct_start, engineering_pct_sum, engineering_pct_avg, engineering_pct_cur = get_term_citation_feature(term_pct_engineering, term_start_year, term_end_year, term_current_year)
            engineering_pct_preceding_1, engineering_pct_cr_1, engineering_pct_firstdif = get_term_citation_preceeding_feature(term_pct_engineering, term_current_year, 1)        
            features['avg_engineering_pct_start'] = engineering_pct_start 
            features['avg_engineering_pct_total'] = engineering_pct_sum 
            features['avg_engineering_pct_preceding_1'] = engineering_pct_preceding_1 
            features['avg_engineering_pct_first_diff'] = engineering_pct_firstdif
            features['avg_engineering_pct_cur'] = engineering_pct_cur
            
            agriculture_pct_start, agriculture_pct_sum, agriculture_pct_avg, agriculture_pct_cur = get_term_citation_feature(term_pct_agriculture, term_start_year, term_end_year, term_current_year)
            agriculture_pct_preceding_1, agriculture_pct_cr_1, agriculture_pct_firstdif = get_term_citation_preceeding_feature(term_pct_agriculture, term_current_year, 1)        
            features['avg_agriculture_pct_start'] = agriculture_pct_start 
            features['avg_agriculture_pct_total'] = agriculture_pct_sum 
            features['avg_agriculture_pct_preceding_1'] = agriculture_pct_preceding_1 
            features['avg_agriculture_pct_first_diff'] = agriculture_pct_firstdif
            features['avg_agriculture_pct_cur'] = agriculture_pct_cur
            
            bio_pct_start, bio_pct_sum, bio_pct_avg, bio_pct_cur = get_term_citation_feature(term_pct_bio, term_start_year, term_end_year, term_current_year)
            bio_pct_preceding_1, bio_pct_cr_1, bio_pct_firstdif = get_term_citation_preceeding_feature(term_pct_bio, term_current_year, 1)        
            features['avg_bio_pct_start'] = bio_pct_start 
            features['avg_bio_pct_total'] = bio_pct_sum 
            features['avg_bio_pct_preceding_1'] = bio_pct_preceding_1 
            features['avg_bio_pct_first_diff'] = bio_pct_firstdif
            features['avg_bio_pct_cur'] = bio_pct_cur
            
            ss_pct_start, ss_pct_sum, ss_pct_avg, ss_pct_cur = get_term_citation_feature(term_pct_social_science, term_start_year, term_end_year, term_current_year)
            ss_pct_preceding_1, ss_pct_cr_1, ss_pct_firstdif = get_term_citation_preceeding_feature(term_pct_social_science, term_current_year, 1)        
            features['avg_ss_pct_start'] = ss_pct_start 
            features['avg_ss_pct_total'] = ss_pct_sum 
            features['avg_ss_pct_preceding_1'] = ss_pct_preceding_1 
            features['avg_ss_pct_first_diff'] = ss_pct_firstdif
            features['avg_ss_pct_cur'] = ss_pct_cur
            
            features['concept_age'] = term_current_year - term_start_year
            all_features.append(features)
            
        term_feature.append((term, all_features))
        if i % 1000 == 0:
             print(i)
    
    try:
        f = open(file,'a')
        for pair in term_feature:
            term = pair[0]
            for features in pair[1]:
                f.write(term + '\t' +  str(features['avg_venue_citation_start']) + ' '+str(features['avg_venue_citation_total']) + ' ' + str(features['avg_venue_citation_preceding_1']) + ' ' + str(features['avg_venue_citation_first_diff']) + ' ' +str(features['avg_venue_citation_cur']) + ' ' + str(features['avg_entropy_start']) + ' ' + str(features['avg_entropy_total']) +' '+ str(features['avg_entropy_preceding_1']) + ' ' + str(features['avg_entropy_first_diff']) + ' ' + str(features['avg_entropy_cur']) + ' ' + str(features['avg_entropy_s_start']) + ' ' + str(features['avg_entropy_s_total']) + ' ' + str(features['avg_entropy_s_preceding_1']) + ' ' + str(features['avg_entropy_s_first_diff']) + ' ' + str(features['avg_entropy_s_cur']) + ' ' + str(features['new_neighbor_wgt_pct_start']) + ' ' + str(features['new_neighbor_wgt_pct_total']) + ' ' +  str(features['new_neighbor_wgt_pct_preceding_1']) + ' ' + str(features['new_neighbor_wgt_pct_first_diff']) + ' ' + str(features['new_neighbor_wgt_pct_cur']) + ' ' + str(features['new_neighbor_avg_pct_start']) + ' ' + str(features['new_neighbor_avg_pct_total']) + ' ' + str(features['new_neighbor_avg_pct_preceding_1']) + ' ' + str(features['new_neighbor_avg_pct_first_diff']) + ' ' + str(features['new_neighbor_avg_pct_cur']) + ' ' + str(features['new_author_pct_avg_start']) + ' ' + str(features['new_author_pct_avg_total']) + ' ' + str(features['new_author_pct_avg_preceding_1']) + ' ' + str(features['new_author_pct_avg_first_diff']) + ' ' + str(features['new_author_pct_avg_cur']) + ' ' + str(features['new_author_pct_start']) + ' ' + str(features['new_author_pct_total']) + ' ' + str(features['new_author_pct_preceding_1']) + ' ' + str(features['new_author_pct_first_diff']) + ' ' + str(features['new_author_pct_cur']) + ' ' +   str(features['avg_author_size_start']) + ' ' + str(features['avg_author_size_total']) + ' ' + str(features['avg_author_size_preceding_1']) + ' ' + str(features['avg_author_size_first_diff']) + ' ' + str(features['avg_author_size_cur']) + ' ' + str(features['avg_author_pub_num_start']) + ' ' + str(features['avg_author_pub_num_total']) + ' ' + str(features['avg_author_pub_num_preceding_1']) + ' ' + str(features['avg_author_pub_num_first_diff']) + ' ' + str(features['avg_author_pub_num_cur']) +  ' ' + str(features['avg_author_ruse_start']) + ' ' + str(features['avg_author_ruse_preceding_1']) + ' ' + str(features['author_female_start']) + ' '  + str(features['author_female_total']) + ' ' + str(features['author_female_preceding_1']) + ' ' + str(features['author_female_first_diff']) + ' ' + str(features['author_female_cur']) + ' ' + str(features['avg_neighbor_pop_start']) + ' ' + str(features['avg_neighbor_pop_total']) + ' ' + str(features['avg_neighbor_pop_preceding_1']) + ' ' + str(features['avg_neighbor_pop_first_diff']) + ' ' + str(features['avg_neighbor_pop_cur']) + ' ' + str(features['sum_neighbor_pop_start']) + ' ' +  str(features['sum_neighbor_pop_total']) + ' ' + str(features['sum_neighbor_pop_preceding_1']) + ' ' + str(features['sum_neighbor_pop_first_diff']) + ' ' + str(features['sum_neighbor_pop_cur']) + ' ' + str(features['mid_neighbor_pop_start']) + ' ' + str(features['mid_neighbor_pop_total']) + ' ' + str(features['mid_neighbor_pop_preceding_1']) + ' ' + str(features['mid_neighbor_pop_first_diff']) + ' ' + str(features['mid_neighbor_pop_cur']) + ' ' + str(features['neighbor_ruse_start']) + ' '+  str(features['neighbor_ruse_preceding_1']) + ' ' + str(features['avg_physics_pct_start']) + ' ' +  str(features['avg_physics_pct_total']) + ' ' + str(features['avg_physics_pct_preceding_1']) + ' ' + str(features['avg_physics_pct_first_diff']) + ' ' + str(features['avg_physics_pct_cur']) + ' ' +  str(features['avg_engineering_pct_start']) + ' ' + str(features['avg_engineering_pct_total']) + ' ' + str(features['avg_engineering_pct_preceding_1']) + ' ' + str(features['avg_engineering_pct_first_diff']) + ' ' + str(features['avg_engineering_pct_cur']) + ' ' + str(features['avg_agriculture_pct_start']) + ' ' + str(features['avg_agriculture_pct_total']) + ' ' + str(features['avg_agriculture_pct_preceding_1']) + ' ' + str(features['avg_agriculture_pct_first_diff']) + ' ' + str(features['avg_agriculture_pct_cur']) + ' ' + str(features['avg_bio_pct_start']) + ' ' + str(features['avg_bio_pct_total']) + ' ' + str(features['avg_bio_pct_preceding_1']) + ' ' + str(features['avg_bio_pct_first_diff']) + ' ' + str(features['avg_bio_pct_cur']) + ' ' + str(features['avg_ss_pct_start']) + ' ' + str(features['avg_ss_pct_total']) + ' ' + str(features['avg_ss_pct_preceding_1']) + ' ' + str(features['avg_ss_pct_first_diff']) + ' ' + str(features['avg_ss_pct_cur']) + ' ' +  str(features['concept_age']) + '\n')
        f.close()
    except:
        pass

    term_feature_all.extend(term_feature)  


dir = '/dfs/scratch0/hanchcao/data/isi_random/'
BURN_IN_YEAR = 1992
START_YEAR = 1992 
END_YEAR = 2016
file_isi_term_paperlist = dir + 'ISI_term_paperlist_full_corpus_lemmitized_loose_corrected_final.txt'
file_isi_paper_year = dir + 'ISI_paper_year.txt'
file_isi_paper_author = dir + 'ISI_paper_author.txt'
file_isi_paper_venue = dir + 'ISI_paper_venue.txt'
file_isi_citation_yearly = dir + 'ISI_paper_yearly_citation.txt'
file_paper_subject = dir + 'ISI_paper_subject.txt'
file_paper_subject2 = dir + 'ISI_paper_subject2.txt'
file_isi_author_gender = dir + 'ISI_author_gender.txt'
file_term_df_year = dir + 'ISI_df_year_full_corpus_lemmitized_loose_corrected_final.txt'
file_total_df_year = dir + 'ISI_paper_each_year.txt'


numOfProcesses = 5


### load meta data
print("loading paper_year")
paper_year = load_paper2item(file_isi_paper_year)
print("original paper year2: ", len(paper_year))
paper_year2 = dict((k, int(v)) for k, v in paper_year.items())
print("original paper year: ", len(paper_year))
paper_year = dict((k, int(v)) for k, v in paper_year.items() if int(v) >= START_YEAR and int(v) <= END_YEAR)
print("valid paper year: ", len(paper_year))
print("loading author gender")
author_gender = load_paper2item(file_isi_author_gender)
print("author gender: ", len(author_gender))
print("loading paper_venue")
paper_venue = load_paper2item(file_isi_paper_venue)
print(len(paper_venue))
print("loading term_paper_list")
term_paper_list = load_paper2list2(file_isi_term_paperlist)
print("# terms:", len(term_paper_list))
print("loading paper_subject")
paper_subject = load_paper2list_sub(file_paper_subject)
print("# papers:", len(paper_subject))
print("loading paper_subject2")
paper_subject2 = load_paper2list(file_paper_subject2)
print("# papers:", len(paper_subject2))

print("loading paper_neighbor_list")
paper_neighbor_list = load_paperlist2paperneighbors(term_paper_list)

print("loading paper_author_list")
paper_author_list = load_paper2list(file_isi_paper_author)
print(len(paper_author_list))
print("loading author_paper_list")
author_paper_list = load_paper2list_reverse2(file_isi_paper_author)
print(len(author_paper_list))

print("loading venue_paper_list")
venue_paper_list = load_paper2list_reverse(file_isi_paper_venue)
print("# venue:", len(venue_paper_list))
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
term_list, term_STARTYEAR, term_firstyear_dict = get_first_year(term_df_year)
print("# terms:", len(term_list))
print('preprocessing citation_year_count')
citation_year_count = load_paper2list3(file_isi_citation_yearly)
print("# citation:", len(citation_year_count))

print('preprocessing venue_year_citation')
venue_year_count = get_venue_cum_citations(paper_venue, citation_year_count)
print("# venue:" ,len(venue_year_count))

print('preprocessing author_age')
author_age = get_author_age(author_paper_list, paper_year2)

print("loading term_author_yearly_count")
file_author_yearly = '/dfs/scratch0/hanchcao/data/isi_author/author_yearly_pub_count.txt'
author_yearly_pub = load_author_burnin(file_author_yearly)
print("# terms:" ,len(author_yearly_pub))

file_author_yearly = '/dfs/scratch0/hanchcao/data/isi_author/author_coauthor_count_yearly_complete.txt'
print("loading term_coauthor_year")
term_coauthor_year = load_coauthor_burnin(file_author_yearly)
print("# terms:" ,len(term_coauthor_year))



####
BURN_IN_YEAR = 1992
START_YEAR = 1992 
END_YEAR = 2016
f_out_term_feature = dir + 'isi_resources_features_loose/'
termsPerProc = int(math.floor(len(term_list) * 1.0/numOfProcesses))
mgr = Manager()
term_feature_all = mgr.list()

output_file = []
for i in range(numOfProcesses):
    output_file.append(f_out_term_feature+str(i)+str(i)+'.txt')


jobs = []
for i in range(numOfProcesses):
    if i == numOfProcesses - 1:
        p = Process(target=feature_worker, args=(term_list[i*termsPerProc:], term_df_year, total_df_year, term_STARTYEAR[i*termsPerProc:], term_ENDYEAR[i*termsPerProc:], paper_year, paper_subject, paper_subject2, term_paper_list, paper_venue,  citation_year_count, venue_year_count, author_age, paper_author_list, author_paper_list, author_gender, term_coauthor_year, author_yearly_pub, term_firstyear_dict, term_feature_all, output_file[i]))
    else:
        p = Process(target=feature_worker, args=(term_list[i*termsPerProc:(i+1)*termsPerProc], term_df_year, total_df_year, term_STARTYEAR[i*termsPerProc:(i+1)*termsPerProc], term_ENDYEAR[i*termsPerProc:(i+1)*termsPerProc], paper_year, paper_subject, paper_subject2, term_paper_list, paper_venue,  citation_year_count, venue_year_count, author_age, paper_author_list, author_paper_list, author_gender, term_coauthor_year, author_yearly_pub, term_firstyear_dict, term_feature_all, output_file[i]))
    jobs.append(p)
    p.start()
    
for proc in jobs:
    proc.join()

print('done')


### dump to files
f_out_term_feature = dir + 'isi_resources_features_loose.txt'

print("begin dumping")
print("# terms:", len(term_feature_all))
with open(f_out_term_feature, 'w') as f:
    for pair in term_feature_all:
        term = pair[0]
        for features in pair[1]:
            f.write(term + '\t' +  str(features['avg_venue_citation_start']) + ' '+str(features['avg_venue_citation_total']) + ' ' + str(features['avg_venue_citation_preceding_1']) + ' ' + str(features['avg_venue_citation_first_diff']) + ' ' +str(features['avg_venue_citation_cur']) + ' ' + str(features['avg_entropy_start']) + ' ' + str(features['avg_entropy_total']) +' '+ str(features['avg_entropy_preceding_1']) + ' ' + str(features['avg_entropy_first_diff']) + ' ' + str(features['avg_entropy_cur']) + ' ' + str(features['avg_entropy_s_start']) + ' ' + str(features['avg_entropy_s_total']) + ' ' + str(features['avg_entropy_s_preceding_1']) + ' ' + str(features['avg_entropy_s_first_diff']) + ' ' + str(features['avg_entropy_s_cur']) + ' ' + str(features['new_neighbor_wgt_pct_start']) + ' ' + str(features['new_neighbor_wgt_pct_total']) + ' ' +  str(features['new_neighbor_wgt_pct_preceding_1']) + ' ' + str(features['new_neighbor_wgt_pct_first_diff']) + ' ' + str(features['new_neighbor_wgt_pct_cur']) + ' ' + str(features['new_neighbor_avg_pct_start']) + ' ' + str(features['new_neighbor_avg_pct_total']) + ' ' + str(features['new_neighbor_avg_pct_preceding_1']) + ' ' + str(features['new_neighbor_avg_pct_first_diff']) + ' ' + str(features['new_neighbor_avg_pct_cur']) + ' ' + str(features['new_author_pct_avg_start']) + ' ' + str(features['new_author_pct_avg_total']) + ' ' + str(features['new_author_pct_avg_preceding_1']) + ' ' + str(features['new_author_pct_avg_first_diff']) + ' ' + str(features['new_author_pct_avg_cur']) + ' ' + str(features['new_author_pct_start']) + ' ' + str(features['new_author_pct_total']) + ' ' + str(features['new_author_pct_preceding_1']) + ' ' + str(features['new_author_pct_first_diff']) + ' ' + str(features['new_author_pct_cur']) + ' ' +   str(features['avg_author_size_start']) + ' ' + str(features['avg_author_size_total']) + ' ' + str(features['avg_author_size_preceding_1']) + ' ' + str(features['avg_author_size_first_diff']) + ' ' + str(features['avg_author_size_cur']) + ' ' + str(features['avg_author_pub_num_start']) + ' ' + str(features['avg_author_pub_num_total']) + ' ' + str(features['avg_author_pub_num_preceding_1']) + ' ' + str(features['avg_author_pub_num_first_diff']) + ' ' + str(features['avg_author_pub_num_cur']) + ' ' + str(features['avg_author_ruse_start']) + ' ' + str(features['avg_author_ruse_preceding_1']) + ' ' + str(features['author_female_start']) + ' '  + str(features['author_female_total']) + ' ' + str(features['author_female_preceding_1']) + ' ' + str(features['author_female_first_diff']) + ' ' + str(features['author_female_cur']) + ' ' + str(features['avg_neighbor_pop_start']) + ' ' + str(features['avg_neighbor_pop_total']) + ' ' + str(features['avg_neighbor_pop_preceding_1']) + ' ' + str(features['avg_neighbor_pop_first_diff']) + ' ' + str(features['avg_neighbor_pop_cur']) + ' ' + str(features['sum_neighbor_pop_start']) + ' ' +  str(features['sum_neighbor_pop_total']) + ' ' + str(features['sum_neighbor_pop_preceding_1']) + ' ' + str(features['sum_neighbor_pop_first_diff']) + ' ' + str(features['sum_neighbor_pop_cur']) + ' ' + str(features['mid_neighbor_pop_start']) + ' ' + str(features['mid_neighbor_pop_total']) + ' ' + str(features['mid_neighbor_pop_preceding_1']) + ' ' + str(features['mid_neighbor_pop_first_diff']) + ' ' + str(features['mid_neighbor_pop_cur']) + ' ' + str(features['neighbor_ruse_start']) + ' '+  str(features['neighbor_ruse_preceding_1']) + ' ' + str(features['avg_physics_pct_start']) + ' ' +  str(features['avg_physics_pct_total']) + ' ' + str(features['avg_physics_pct_preceding_1']) + ' ' + str(features['avg_physics_pct_first_diff']) + ' ' + str(features['avg_physics_pct_cur']) + ' ' +  str(features['avg_engineering_pct_start']) + ' ' + str(features['avg_engineering_pct_total']) + ' ' + str(features['avg_engineering_pct_preceding_1']) + ' ' + str(features['avg_engineering_pct_first_diff']) + ' ' + str(features['avg_engineering_pct_cur']) + ' ' + str(features['avg_agriculture_pct_start']) + ' ' + str(features['avg_agriculture_pct_total']) + ' ' + str(features['avg_agriculture_pct_preceding_1']) + ' ' + str(features['avg_agriculture_pct_first_diff']) + ' ' + str(features['avg_agriculture_pct_cur']) + ' ' + str(features['avg_bio_pct_start']) + ' ' + str(features['avg_bio_pct_total']) + ' ' + str(features['avg_bio_pct_preceding_1']) + ' ' + str(features['avg_bio_pct_first_diff']) + ' ' + str(features['avg_bio_pct_cur']) + ' ' + str(features['avg_ss_pct_start']) + ' ' + str(features['avg_ss_pct_total']) + ' ' + str(features['avg_ss_pct_preceding_1']) + ' ' + str(features['avg_ss_pct_first_diff']) + ' ' + str(features['avg_ss_pct_cur']) + ' ' +  str(features['concept_age']) + '\n')
print("done")



