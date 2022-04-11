######################################################################
######################################################################
####code for "Modeling the Growth of Scientific Concepts" ############
####code for generating within WoS citation for each concept #########
####Correspondence:###################################################
####Daniel A. McFarland###############################################
####e-mail: mcfarland@stanford.edu####################################
######################################################################
######################################################################

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from collections import defaultdict
import numpy as np
from scipy import stats


f_citation_dir = '/dfs/scratch0/hanchcao/data/isi_random/WoS_Citations.txt'


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


g = open('Article_inside_WoS_Citation_percentage','w')
article_citation = {}
f_citation = open(f_citation_dir)
f_citation.readline()
f_citation.readline()
line = f_citation.readline()
count = 0
            
while line:
    try:
        id = line.split('      ')[0].strip()
        if  hasNumbers(line.split('      ')[1].strip()):
            id_cited = line.split('      ')[1].strip()
        else:
            id_cited = 'outside'

        if id in article_citation:
            article_citation[id].append(id_cited)
        else:
            article_citation[id]=[id_cited]

        if count%1000000 == 0:
            print (count)
        count += 1
    except:
        pass
    line = f_citation.readline()


# calculate the percentage and save to file
g = open('Article_inside_WoS_Citation_percentage','w')
count = 0
for id in article_citation:
    cited_list = article_citation[id]
    g.write(id)
    g.write('\t')
    try:
        percentage = 1- cited_list.count('outside')/float(len(cited_list))
        g.write(str(percentage))
    except:
        g.write('')
    g.write('\n')
        
    if count%1000000 == 0:
        print (count)
    count += 1    


dir = '/dfs/scratch0/hanchcao/data/'
file_isi_paper_year = dir + 'isi_random/ISI_paper_year.txt'
file_isi_in_WoS_citation_percentage = dir + 'isi_random/Article_inside_WoS_Citation_percentage'

def load_paper2item(filename):
    paper2item = defaultdict(str)
    with open(filename) as f:
        for line in f:
            pair = line.split('\t')
            try:
                if pair[1].strip() != '':
                    paper2item[pair[0].strip()] = pair[1].strip()
            except:
                pass
    print('load # papers: %d'%len(paper2item))
    return paper2item

print("loading paper_in web of science citation percentage")
paper_in_wos_citation_percent = load_paper2item(file_isi_in_WoS_citation_percentage)
print("loading paper_year")
paper_year = load_paper2item(file_isi_paper_year)

file_term_df_year = dir + 'isi_random/ISI_df_year_full_corpus_lemmitized_loose_corrected_final.txt'
file_isi_term_paperlist = dir + 'isi_random/ISI_term_paperlist_full_corpus_lemmitized_loose_corrected_final.txt'


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



BURN_IN_YEAR = 1992
START_YEAR = 1992 
END_YEAR = 2016 

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

term_df_year = load_df_burnin(file_term_df_year)
term_list = list(term_df_year.keys())


# calculate concept level in WoS citation percentage and save to file
START_YEAR = 1992 #1990
END_YEAR = 2016 #2012
year_window = 1  # control how many years in advance do we aggregate for the metric

count = 0

g = open('In_WoS_citation_percentage_loose_term.txt','w')

for term in term_list:
    g.write(term)
    g.write('\t')
    paper_list = term_paper_list[term]
    term_year_in_wos_percent = {}
    for paper in paper_list:
        year = paper_year[paper]
        if year in term_year_in_wos_percent:
            term_year_in_wos_percent[year].append(paper_in_wos_citation_percent[paper])
        else:
            term_year_in_wos_percent[year] = [paper_in_wos_citation_percent[paper]]
            
    for year in range(START_YEAR, END_YEAR+1, 1):
        cite_percentage_list_temp = []
        for y in range(year-year_window+1,year+1,1):
            if str(y) in term_year_in_wos_percent:
                cite_percentage_list_temp.extend(term_year_in_wos_percent[str(y)])
        cite_percentage_list_temp1 = filter(lambda a: a != '', cite_percentage_list_temp)
        if len(cite_percentage_list_temp1)>0:
            g.write(str(np.nanmean(map(float,cite_percentage_list_temp1)))+' ')
        else:
            g.write('-1234'+' ')
    g.write('\n')
            
    if count%1000 == 0:
        print (count)
    count += 1   
    
            
g.close()




