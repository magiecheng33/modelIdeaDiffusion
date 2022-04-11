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
import random
from collections import defaultdict
from flashtext import KeywordProcessor
from multiprocessing import Process, Manager
import math

# sample
from flashtext import KeywordProcessor
keyword_processor = KeywordProcessor()
keyword_processor.add_keyword('Big Apple')
keyword_processor.add_keyword('Bay Area')
keywords_found = keyword_processor.extract_keywords('I love big Apple and Bay Area.')
keywords_found

# load wos concepts
dir = '/dfs/scratch0/hanchcao/data/'
file_term_df_year = dir + 'isi_random/ISI_df_year_chunks_allrandom30.txt'

BURN_IN_YEAR = 1992
START_YEAR = 1992 #1990
END_YEAR = 2008 #2012

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


############# no burn in version
dir = '/dfs/scratch0/hanchcao/data/'
file_term_df_year = dir + 'isi_random/ISI_df_year_chunks_allrandom30.txt'

# without burn in
def load_df(fname):
    features = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            features[line[0]] = list(map(lambda x: float(x), line[1:][START_YEAR - 1900:END_YEAR-1900+1]))
    print ('load # terms: %d', len(features))
    return features

term_df_year_not_burn_in = load_df(file_term_df_year) 


Concept_set = term_df_year.keys()


Son_concepts = {}
count = 0
for Concept_focal in Concept_set[0:20]:
    count +=1
    if count%10 ==0:
        print (count)
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keyword(Concept_focal)
    for concept_temp in Concept_set:
        match = keyword_processor.extract_keywords(concept_temp)
        if len(match) == 1 and Concept_focal!=concept_temp: 
            if Concept_focal in Son_concepts:
                Son_concepts[Concept_focal].append(concept_temp) 
            else:
                Son_concepts[Concept_focal] = [concept_temp] 

######## multiple threads
def get_son_concept(Concept_set, start_idx, end_idx, file):
    Son_concepts = {}
    count = 0
    for idx in range(start_idx, end_idx, 1):
        print (idx)
        Concept_focal = Concept_set[idx]
        keyword_processor = KeywordProcessor()
        keyword_processor.add_keyword(Concept_focal)
        for concept_temp in Concept_set:
            match = keyword_processor.extract_keywords(concept_temp)
            if len(match) == 1 and Concept_focal!=concept_temp: 
                if Concept_focal in Son_concepts:
                    Son_concepts[Concept_focal].append(concept_temp) 
                else:
                    Son_concepts[Concept_focal] = [concept_temp] 
    f = open(file,'a')
    for concept in Son_concepts:
        output = []
        output.append(concept)
        for son in Son_concepts[concept]:
            output.append(son)
        f.write('\t'.join(output) + '\n')
    f.close()


numOfProcesses =50

dir = '/dfs/scratch0/hanchcao/data/isi/concept_substring_match/'
f_SonConcept = dir + 'allrandom30_SonConcept/'

termsPerProc = int(math.floor(len(Concept_set[0:500]) * 1.0/numOfProcesses))
mgr = Manager()

Son_concepts = {}

jobs = []

output_file = []
for i in range(numOfProcesses):
    output_file.append(f_SonConcept +str(i)+'.txt')
    
for i in range(numOfProcesses):
    if i == numOfProcesses - 1:
        p = Process(target=get_son_concept, args=(Concept_set, i*termsPerProc, len(Concept_set[0:500]), output_file[i]))
    else:
        p = Process(target=get_son_concept, args=(Concept_set, i*termsPerProc, (i+1)*termsPerProc, output_file[i]))
    jobs.append(p)
    p.start()
    
for proc in jobs:
    proc.join()

print("getting the father of each son")
f = open('/dfs/scratch0/hanchcao/data/isi/concept_substring_match/allrandom30_SonConcept/concept_son_allrandom30.txt')
dict_father = {}
count = 0
while line:
    line = line[0:-1].split('\t')
    count+=1
    if count%10000==0:
        print (count)
    for son in line[1:]:
        if son in dict_father:
            dict_father[son].append(line[0])
        else:
            dict_father[son] = [line[0]]
    line = f.readline()

f = open('/dfs/scratch0/hanchcao/data/isi/concept_substring_match/allfields2_SonConcept/concept_father_allfields2.txt','a')
for concept in dict_father:
    output = []
    output.append(concept)
    for son in dict_father[concept]:
        output.append(son)
    f.write('\t'.join(output) + '\n')
f.close()


print("count son number!")
file_input= '/dfs/scratch0/hanchcao/data/isi/concept_substring_match/loose_SonConcept/SonConcept_list_loose.txt'
file_output = '/dfs/scratch0/hanchcao/data/isi/concept_substring_match/loose_SonConcept/SonConcept_Num_loose.txt'
f = open(file_input)
g = open(file_output,'w')
line = f.readline()
while line:
    concept = line.split('\t')[0]
    number = len(line.split('\t')) - 1
    g.write(concept)
    g.write('\t')
    g.write(str(number))
    g.write('\n')
    line = f.readline()
f.close()
g.close()

print("done!")