import gzip
import os
import numpy as np 
import pickle
import random 

def mainFunc() :
	# simInputFile = "Q1/word-similarity-dataset"
	# analogyInputFile = "Q1/word-analogy-dataset"
	# vectorgzipFile = "Q1/glove.6B.300d.txt.gz"
	# vectorTxtFile = "Q1/glove.6B.300d.txt"   # If you extract and use the gz file, use this.
	# analogyTrainPath = "Q1/wordRep/Pairs_from_WordNet"
	# simOutputFile = "Q1/simOutput.csv"
	# simSummaryFile = "Q1/simSummary.csv"
	# anaSoln = "Q1/analogySolution.csv"
	# Q4List = "Q4/wordList.csv"

	vectorFile = gzip.open(vectorgzipFile,'r')

	#1
	analogy_words = []
	for files in os.listdir(analogyTrainPath):
	    f = open(analogyTrainPath+'/'+files).read().splitlines()
	    g = [item.split('\t') for item in f]
	    for i in range(len(g)):
	    	if len(g[i]) < 2:
	    		g[i] = g[i][0].split(' ')
	    for item in g:
	    	for word in item:
	    		analogy_words.append(word)

	wordDict = dict()
	for line in vectorFile:
		if line.split()[0].strip() in analogy_words:
			wordDict[line.split()[0].strip()] = line.split()[1:]
	vectorFile.close()

	# with open('analogy_word_dict.pkl','wb') as f:
	# 	pickle.dump(wordDict,f)

	#2
	# with open('analogy_word_dict.pkl','rb') as f:
	# 	wordDict = pickle.load(f)
	keys = wordDict.keys()
	allfiles_dict = dict()
	for files in os.listdir(analogyTrainPath):
		allfiles_dict[files] = []
		f = open(analogyTrainPath+'/'+files).read().splitlines()
		g = [item.split('\t') for item in f]
		for i in range(len(g)):
			if len(g[i])<2:
				g[i] = g[i][0].split(' ')
		for item in g:
			temp = dict()
			if item[0] in keys and item[1] in keys:
				temp['pair'] = item
				temp['vector'] = np.array([float(i) for i in wordDict[item[0]]]) - np.array([float(i) for i in wordDict[item[1]]])
				allfiles_dict[files].append(temp)

	# with open('allfiles_dict.pkl','wb') as f:
	# 	pickle.dump(allfiles_dict,f)

	#3
	# with open('allfiles_dict.pkl','rb') as f:
	# 	allfiles_dict = pickle.load(f)

	files = allfiles_dict.keys()
	for i in range(len(files)):
		if len(allfiles_dict[files[i]])<2:
			del allfiles_dict[files[i]]

	files = allfiles_dict.keys()
	# dictindex = 1
	dataset = []
	for i in range(len(files)):
		for j in range(len(allfiles_dict[files[i]])):
			for rep in range(2):
				k = random.randint(0,len(files)-1) #select a file for negative example
				while k==i:
					k = random.randint(0,len(files)-1)
				p = random.randint(0,len(allfiles_dict[files[i]])-1) #select for positive example from ith file
				n = random.randint(0,len(allfiles_dict[files[k]])-1) #select for negative example from kth file
				temp_pos = dict()
				temp_neg = dict()
				temp_pos['pair1'] = allfiles_dict[files[i]][j]['pair']
				temp_pos['pair2'] = allfiles_dict[files[i]][p]['pair']
				temp_pos['vector'] = np.concatenate((allfiles_dict[files[i]][j]['vector'],allfiles_dict[files[i]][p]['vector']),0)
				temp_neg['pair1'] = allfiles_dict[files[i]][j]['pair']
				temp_neg['pair2'] = allfiles_dict[files[k]][n]['pair']
				temp_neg['vector'] = np.concatenate((allfiles_dict[files[i]][j]['vector'],allfiles_dict[files[k]][n]['vector']),0)
				temp_dict = dict()
				temp_dict['pos'] = temp_pos
				temp_dict['neg'] = temp_neg
				dataset.append(temp_dict)
				
				# dictindex += 1

	# with open('dataset_pos_neg.pkl','wb') as f:
	# 	pickle.dump(dataset,f)

	#4

	vectorFile = gzip.open(vectorgzipFile,'r')
	analogyDataset = [[stuff.strip() for stuff in item.strip('\n').split('\n')] for item in open(analogyInputFile).read().split('\n\n')]

	analogyList = []
	for i in range(len(analogyDataset)):
		for j in range(len(analogyDataset[i])-1):
			for item in analogyDataset[i][j].split(' '):
				analogyList.append(item)

	analogy_dict = dict()
	for line in vectorFile:
	    if line.split()[0].strip() in analogyList:
	        analogy_dict[line.split()[0].strip()] = line.split()[1:]

	vectorFile.close()


	# with open('analogy_final_dict.pkl','wb') as f:
	# 	pickle.dump(analogy_final_dict,f)


	# 5
	# with open('analogy_final_dict.pkl','rb') as f:
	# 	analogy_dict = pickle.load(f)

	keys = analogy_dict.keys()
	analogyDataset_test = []
	flag = 0
	for i in range(len(analogyDataset)):
		for j in range(len(analogyDataset[i])-1):
			for item in analogyDataset[i][j].split(' '):
				if item not in keys:
					flag = 1
					break
			if flag==1:
				break
		if flag==1:
			flag=0
		else:
			analogyDataset_test.append(analogyDataset[i])

	dataset_analogy_test_final = dict()
	dataset_analogy_test = np.zeros((97,5,600))
	ground_truth_analogy_test = np.zeros(97)
	tempdict = dict()
	tempdict['a'] = 0
	tempdict['b'] = 1
	tempdict['c'] = 2
	tempdict['d'] = 3
	tempdict['e'] = 4
	for i in range(len(analogyDataset_test)):
		item = analogyDataset_test[i][j].split(' ')
		vect_ref = np.array([float(k) for k in analogy_dict[item[0]]])-np.array([float(k) for k in analogy_dict[item[1]]])
		for j in range(1,len(analogyDataset_test[i])-1):
			item = analogyDataset_test[i][j].split(' ')
			vect = np.array([float(k) for k in analogy_dict[item[0]]])-np.array([float(k) for k in analogy_dict[item[1]]])
			dataset_analogy_test[i][j-1] = np.concatenate((vect_ref,vect),0)
		ground_truth_analogy_test[i] = tempdict[analogyDataset[i][6]]
	dataset_analogy_test_final['dataset'] = dataset_analogy_test
	dataset_analogy_test_final['ground_truth'] = ground_truth_analogy_test
	dataset_analogy_test_final['words'] = analogyDataset_test

	# with open('analogy_dataset_questions.pkl','wb') as f:
	# 	pickle.dump(dataset_analogy_test_final,f)
	return dataset,dataset_analogy_test_final

























