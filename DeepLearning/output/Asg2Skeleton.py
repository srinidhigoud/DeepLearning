
# coding: utf-8

# Deep Learning Programming Assignment 2
# --------------------------------------
# Name: Srinidhi Goud
# Roll No.: 13EC10042
# 
# Submission Instructions:
# 1. Fill your name and roll no in the space provided above.
# 2. Name your folder in format <Roll No>_<First Name>.
#     For example 12CS10001_Rohan
# 3. Submit a zipped format of the file (.zip only).
# 4. Submit all your codes. But do not submit any of your datafiles
# 5. From output files submit only the following 3 files. simOutput.csv, simSummary.csv, analogySolution.csv
# 6. Place the three files in a folder "output", inside the zip.

# In[59]:

import tensorflow as tf
import gzip
import os
import csv
import numpy as np
from scipy import spatial
import pickle
from word_analogy_pos_neg_datasetprepro import mainFunc
from analogy_train import trian_analogy
from derived_words_prepro import derived_prepro 
from derived_train import derive_train
## paths to files. Do not change this
simInputFile = "Q1/word-similarity-dataset"
analogyInputFile = "Q1/word-analogy-dataset"
vectorgzipFile = "Q1/glove.6B.300d.txt.gz"
vectorTxtFile = "Q1/glove.6B.300d.txt"   # If you extract and use the gz file, use this.
analogyTrainPath = "Q1/wordRep/"
simOutputFile = "Q1/simOutput.csv"
simSummaryFile = "Q1/simSummary.csv"
anaSoln = "Q1/analogySolution.csv"
Q4List = "Q4/wordList.csv"




# In[ ]:

# Similarity Dataset
simDataset = [item.split(" | ") for item in open(simInputFile).read().splitlines()]
# Analogy dataset
analogyDataset = [[stuff.strip() for stuff in item.strip('\n').split('\n')] for item in open(analogyInputFile).read().split('\n\n')]

def vectorExtract(simD = simDataset, anaD = analogyDataset, vect = vectorgzipFile):
    simList = [stuff.strip(' ') for item in simD for stuff in item]
    analogyList = [thing for item in anaD for stuff in item[0:4] for thing in stuff.split()]
    simList.extend(analogyList)
    wordList = set(simList)
    print len(wordList)
    wordDict = dict()
    
    vectorFile = gzip.open(vect, 'r')
    for line in vectorFile:
        if line.split()[0].strip() in wordList:
            wordDict[line.split()[0].strip()] = line.split()[1:]
    
    
    vectorFile.close()
    print 'retrieved', len(wordDict.keys())
    return wordDict

# Extracting Vectors from Analogy and Similarity Dataset
#validateVectors = vectorExtract()

with open('vectors.pkl','rb') as f :
    #pickle.dump(validateVectors,f)
    validateVectors = pickle.load(f)
# In[ ]:

# Dictionary of training pairs for the analogy task
trainDict = dict()
for subDirs in os.listdir(analogyTrainPath):
    for files in os.listdir(analogyTrainPath+subDirs+'/'):
        f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
        trainDict[files] = f
print len(trainDict.keys())


# In[58]:


def similarityTask(inputDS = simDataset, outputFile = simOutputFile, summaryFile=simSummaryFile, vectors=validateVectors):
    print 'hello world'

    """
    Output simSummary.csv in the following format
    Distance Metric, Number of questions which are correct, Total questions evalauted, MRR
    C, 37, 40, 0.61
    """

    """
    Output a CSV file titled "simOutput.csv" with the following columns

    file_line-number, query word, option word i, distance metric(C/E/M), similarity score 

    For the line "rusty | corroded | black | dirty | painted", the outptut will be

    1,rusty,corroded,C,0.7654
    1,rusty,dirty,C,0.8764
    1,rusty,black,C,0.6543
    

    The order in which rows are entered does not matter and so do Row header names. Please follow the order of columns though.
    """
    j=1
    outputfile = open(outputFile,'wb')
    summaryfile = open(summaryFile,'wb')
    writer_output = csv.writer(outputfile,delimiter=',')
    writer_summary = csv.writer(summaryfile,delimiter=',')
    MRR_C = []
    MRR_E = []
    MRR_M = []
    corr_C = 0
    corr_E = 0
    corr_M = 0
    for item in inputDS:
        refword = item[0]
        refvect = np.array([float(i) for i in vectors[refword]])
        C = []
        E = []
        M = []
        tempC = 0
        for i in range(len(item)) :
            item[i] = item[i].strip(' ')
            if ' ' in item[i] :
                break
            tempC+=1
        if tempC<len(item) :
            continue        
        for word in item[1:]:
            wordvect = np.array([float(i) for i in vectors[word]])
            C.append(spatial.distance.cosine(refvect,wordvect))
            writer_output.writerow([str(j),refword,word,'C',str(1-C[-1])])
            E.append(spatial.distance.euclidean(refvect,wordvect))
            writer_output.writerow([str(j),refword,word,'E',str(1/(1+E[-1]))])
            M.append(np.sum(np.abs(refvect-wordvect)))
            writer_output.writerow([str(j),refword,word,'M',str(1/(1+M[-1]))])
        C = np.array(C)
        E = np.array(E)
        M = np.array(M)
        if np.argmin(C)==0:
            corr_C+=1
        MRR_C.append(C.argsort()[0]+1)
        if np.argmin(E)==0:
            corr_E+=1
        MRR_E.append(E.argsort()[0]+1)
        if np.argmin(M)==0:
            corr_M+=1
        MRR_M.append(M.argsort()[0]+1)
        j = j+1
    MRR_C1 = np.mean(np.reciprocal([float(i) for i in MRR_C]))
    MRR_E1 = np.mean(np.reciprocal([float(i) for i in MRR_E]))
    MRR_M1 = np.mean(np.reciprocal([float(i) for i in MRR_M]))

    writer_summary.writerow(['C',str(corr_C),str(j-1),str(MRR_C1)])
    writer_summary.writerow(['E',str(corr_E),str(j-1),str(MRR_E1)])
    writer_summary.writerow(['M',str(corr_M),str(j-1),str(MRR_M1)])
    outputfile.close()
    summaryfile.close()

# In[ ]:

def analogyTask(inputDS=analogyDataset,outputFile = anaSoln ): # add more arguments if required
    
    """
    Output a file, analogySolution.csv with the following entris
    Query word pair, Correct option, predicted option    
    """
    
    train_dataset,analogy_dataset = mainFunc()
    accuracy1,accuracy = train_analogy(train_dataset,analogy_dataset)
    return accuracy1
    # return accuracy #return the accuracy of your model after 5 fold cross validation



# In[60]:

def derivedWOrdTask(inputFile = Q4List):
    print 'hello world'
    
    """
    Output vectors of 3 files:
    1)AnsFastText.txt - fastText vectors of derived words in wordList.csv
    2)AnsLzaridou.txt - Lazaridou vectors of the derived words in wordList.csv
    3)AnsModel.txt - Vectors for derived words as provided by the model
    
    For all the three files, each line should contain a derived word and its vector, exactly like 
    the format followed in "glove.6B.300d.txt"
    
    word<space>dim1<space>dim2........<space>dimN
    charitably 256.238 0.875 ...... 1.234
    
    """
    
    """
    The function should return 2 values
    1) Averaged cosine similarity between the corresponding words from output files 1 and 3, as well as 2 and 3.
    
        - if there are 3 derived words in wordList.csv, say word1, word2, word3
        then find the cosine similiryt between word1 in AnsFastText.txt and word1 in AnsModel.txt.
        - Repeat the same for word2 and word3.
        - Average the 3 cosine similarity values
        - DO the same for word1 to word3 between the files AnsLzaridou.txt and AnsModel.txt 
        and average the cosine simialities for valuse so obtained
        
    """
    all_words_derived,derived_dict = derivedmain()
    cosVal1,cosVal2 = derive_train(all_words_derived,derived_dict)
    return cosVal1,cosVal2
    


# In[ ]:

def main():
    similarityTask()
    anaSim = analogyTask()
    derCos1,derCos2 = derivedWordTask()

if __name__ == '__main__':
    main()
