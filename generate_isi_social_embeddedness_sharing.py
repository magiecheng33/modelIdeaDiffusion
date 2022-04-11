######################################################################
######################################################################
####code for "Modeling the Growth of Scientific Concepts" ############
####code for generating social conditions + controls #################
####Correspondence:###################################################
####Daniel A. McFarland###############################################
####e-mail: mcfarland@stanford.edu####################################
######################################################################
######################################################################

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
import collections

dir = '/dfs/scratch0/hanchcao/data/'

file_term_df_year = dir + 'isi_random/ISI_df_year_full_corpus_lemmitized_loose_corrected_final.txt'

file_total_df_year = dir + 'isi_random/ISI_paper_each_year.txt'
f_term_semnet_features = dir + 'isi_random/ISI_loose_ecological_features_author_density_0808_2021/'
file_isi_paper_author = dir + 'isi_random/ISI_paper_author.txt'
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
print("loading paper_author_list")
paper_author_list = load_paper2list(file_isi_paper_author)
print(len(paper_author_list))

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


# load author paper list
print ('load author paper list')
f = open('/dfs/scratch0/hanchcao/data/isi_author/ISI_author.txt')
line = f.readline()
author_paper = {}
count = 0
while line:
    count +=1
    paper = line.split('\t')[0]
    author = line.split('\t')[1][0:-1]
    if author in author_paper:
        author_paper[author].append(paper)
    else:
        author_paper[author] = [paper]
    line = f.readline()
print ('load author paper list completed')


# get coauthor relationship
print ('loading coauthor relationship')
f = open('/dfs/scratch0/hanchcao/data/isi_author/author_coauthor_list_each_year.txt','r')
line = f.readline()
author_coauthor_year = {}
count = 0
while line:
    try:
        author = line.split('-')[0]
        if len(author)>0:
            author_coauthor = line.split('-')[1]
            author_coauthor_year[author] = {}
            for j in author_coauthor.split(' ')[0:-1]:
                year = int(j.split(':')[0])
                coauthor_list = j.split(':')[1]
                author_coauthor_year[author][year] = coauthor_list.split(',')[0:-1]
    except:
        pass

    line = f.readline()
    count = count+1
    if count%100000 ==0:
        print ('loading coauthor relationship')
        print (count)
f.close()
print ('loading coauthor relationship completed')


count_missing = 0
count_pair_missing = 0
year_window = 10  # control the length of years before *year* to construct author collaboration network
weight_threshold = 0.9 # total weight threshold used to shrink the collaboration network



def process(term_list, start_idx, end_idx, count_pair_missing, file):
    for idx in range(start_idx, end_idx, 1):
        ### 1. Compute avg closeness to all terms 
        test_term = term_list[idx]
        paper_list = term_paper_list[test_term]
        test_term =  '_'.join(test_term.split())
        
        f = open(file,'a')
        f.write(term_list[idx] + '\t')
        
        term_author_year = {}
                
        for paper in paper_list:
            year = paper_year2[paper]   # get the publication year of the paper
            try:
                authors = paper_author_list[paper]
                if year in term_author_year:
                    term_author_year[year].extend(authors) 
                else:
                    term_author_year[year] = authors
            except:
                pass
            
        for year in range(START_YEAR, END_YEAR+1, 1):
            # construct the network
            if year in term_author_year:                  
                counter=collections.Counter(term_author_year[year]).most_common()
                weight_cutoff = weight_threshold * len(term_author_year[year])
                authors = []
                weight_sum = 0
                for pair in counter:
                    if weight_sum > weight_cutoff:
                        break
                    else:
                        authors.append(pair[0])
                        weight_sum += pair[1]
            else:
                authors = set([])
            G = nx.Graph()
            try:
                for node in authors:
                    G.add_node(node)
                    # find the coauthor of author within 10 years before *year*
                    coauthor_node = []
                    for y in range(year-year_window+1,year+1):
                        if y in author_coauthor_year[node]:
                            coauthor_node.extend(author_coauthor_year[node][y])
                    coauthor_node = set(coauthor_node)

                    # find the intersection of coauthor of node, and those using the concept
                    coauthor_node_concept = coauthor_node.intersection(set(authors))
                    for coauthor in coauthor_node_concept:
                        G.add_edge(node,coauthor)
                if (len(authors)>0):
                    density = nx.classes.function.density(G)
                else:
                    density = -1234
                f.write(str(density)+' ')
            except:
                f.write('-1234'+' ')
                print ('error')
                #pass
        f.write('\n')  
        f.close()        
        print('processing #idx:%d, %s'%(idx, test_term))


numOfProcesses =10
termsPerProc = int(math.floor(len(term_list) * 1.0/numOfProcesses))
mgr = Manager()


output_file = []
for i in range(numOfProcesses):
    output_file.append(f_term_semnet_features+str(i)+'.txt')
        
        
jobs = []
for i in range(numOfProcesses):
    if i == numOfProcesses - 1:
        p = Process(target=process, args=(term_list, i*termsPerProc, len(term_list),
                                          count_pair_missing, output_file[i]))
    else:
        p = Process(target=process, args=(term_list, i*termsPerProc, (i+1)*termsPerProc,  count_pair_missing, output_file[i]))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()


    
print ('saving!')
with open(f_term_semnet_features, 'w') as f:
    for i in range(len(term_list)):
        term = term_list[i]
        f.write(term + '\t')
        for year in range(START_YEAR, END_YEAR+1, 1):
            f.write(str(density[i*(END_YEAR-START_YEAR+1)+year-START_YEAR])+' ')
        f.write('\n')