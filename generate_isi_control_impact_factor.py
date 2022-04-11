######################################################################
######################################################################
####code for "Modeling the Growth of Scientific Concepts" ############
####code for generating journal impact factor at the concept level ###
####Correspondence:###################################################
####Daniel A. McFarland###############################################
####e-mail: mcfarland@stanford.edu####################################
######################################################################
######################################################################


#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import pickle
import itertools
import math
from multiprocessing import Process, Manager
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import random
from scipy import spatial
import networkx as nx
from collections import Counter
import igraph as ig
from collections import OrderedDict
import warnings
import sys,os
import time


dir = '/dfs/scratch0/hanchcao/data/'
file_total_df_year = dir + 'isi_random/ISI_paper_each_year.txt'
file_isi_term_paperlist = dir + 'isi_random/ISI_term_paperlist_full_corpus_lemmitized_loose_corrected_final.txt'
file_term_df_year = dir + 'isi_random/ISI_df_year_full_corpus_lemmitized_loose_corrected_final.txt'
file_wos_article_journal = dir + 'isi_random/WoS_Publications.txt'
file_wos_journal_information = dir + 'isi_random/WoS_Journals.txt'


BURN_IN_YEAR = 1992
START_YEAR = 1992 #1990
END_YEAR = 2016 #2012
year_window = 1  # control how many years in advance do we aggregate for the metric


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
print ('# testing terms:', len(term_list))


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


def load_paper2list(filename):
    paper2list = defaultdict(list)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            if len(pair) == 2:
                paper2list[pair[0].strip()] = pair[1].strip().split(' ')
    print('load # papers: %d'%len(paper2list))
    return paper2list

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


# load paper journal
def load_paper3item(filename):
    paper2item = defaultdict(str)
    with open(filename) as f:
        for line in f:
            pair = line.split('         ')
            try:
                if pair[1].strip() != '':
                    paper2item[pair[0].strip()] = pair[1].strip()
            except:
                pass
    print('load # papers: %d'%len(paper2item))
    return paper2item
paper_journal = load_paper3item(file_wos_article_journal)


f = open(file_wos_journal_information)
journal_id_year_issn = {}
line = f.readline()
while line:
    try:
        journal_id = line.split()[0]
        issn = line.split()[1]
        year = line.split()[-1]
        if journal_id in journal_id_year_issn:
            journal_id_year_issn[journal_id][year] = issn
        else:
            journal_id_year_issn[journal_id] = {}
            journal_id_year_issn[journal_id][year] = issn
    except:
        pass
    line = f.readline()


# load journal issn impact factor file
impact_factor_dir = '/dfs/scratch0/hanchcao/data/isi_random/concept_journal_impact_factor/journal_impact/'
issn_year_impact_factor = {}

for year in range(1992,2018):
    f = open(impact_factor_dir+str(year)+'.csv')
    line = f.readline()
    while line:
        try:
            issn = line.split(',')[2]
            impact_factor = line.split(',')[-4]
            if issn in issn_year_impact_factor:
                issn_year_impact_factor[issn][year] = impact_factor
            else:
                issn_year_impact_factor[issn] = {}
                issn_year_impact_factor[issn][year] = impact_factor
        except:
            pass
        
        line = f.readline()
    f.close()


# get concept level impact factor over the years
g = open('concept_impact_factor_loose_term.txt','w')

count = 0
for term in term_list:
    g.write(term)
    g.write('\t')   
    
    paper_list = term_paper_list[term]
    term_year_if = {}
    for paper in paper_list:
        year = paper_year[paper]
        try:
            issn = journal_id_year_issn[paper_journal[paper]][year]
        
            journal_impact_factor = issn_year_impact_factor[issn][int(year)]
            if year in term_year_if:
                term_year_if[year].append(journal_impact_factor)
            else:
                term_year_if[year] = [journal_impact_factor]
        except:
            pass
            
    for year in range(START_YEAR, END_YEAR+1, 1):
        term_year_if_temp = []
        for y in range(year-year_window+1,year+1,1):
            if str(y) in term_year_if:
                term_year_if_temp.extend(term_year_if[str(y)])
        term_year_if_temp1 = filter(lambda a: a != 'Not Available', term_year_if_temp)
        if len(term_year_if_temp1)>0:
            g.write(str(np.nanmean(map(float,term_year_if_temp1)))+' ')
        else:
            g.write('-1234'+' ')
    g.write('\n')
            
    if count%1000 == 0:
        print (count)
    count += 1   
    
            
g.close()

