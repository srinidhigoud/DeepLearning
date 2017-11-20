import csv
import numpy as np 
import pickle
import os

def derived_prepro():
	fastextFile = "Q4/AnsFastText.txt" 
	lazaridouFile = "Q4/AnsLzaridou.txt"
	# AnsModel.txt

	fastText_vectors = open('Q4/fastText_vectors.txt')
	fastText_dict = dict()
	for line in fastText_vectors.read().splitlines():
		fastText_dict[line.split()[0].strip()] = line.split()[1:]

	vector_lazaridou = open('Q4/vector_lazaridou.txt')
	lazaridou_dict = dict()
	for line in vector_lazaridou.read().splitlines():
		lazaridou_dict[line.split()[0].strip()] = line.split()[1:]


	f = open('Q4/wordList.csv','rb')
	reader = csv.reader(f)

	derived_dict = dict()
	all_words_derived = dict()
	for row in reader:
		if row[1] not in derived_dict.keys():
			derived_dict[row[1]] = dict()
			all_words_derived[row[1]] = []
			fastText_data = dict()
			fastText_data['source'] = []
			fastText_data['derived'] = []
			derived_dict[row[1]]['fastText'] = fastText_data
			lazaridou_data = dict()
			lazaridou_data['source'] = []
			lazaridou_data['derived'] = []
			derived_dict[row[1]]['lazaridou'] = lazaridou_data
		if row[3] in fastText_dict.keys() and row[2] in fastText_dict.keys():
			derived_dict[row[1]]['fastText']['source'].append(np.array([float(i) for i in fastText_dict[row[3]]]))
			derived_dict[row[1]]['fastText']['derived'].append(np.array([float(i) for i in fastText_dict[row[2]]]))
			derived_dict[row[1]]['lazaridou']['source'].append(np.array([float(i.strip(',').strip('[').strip(']')) for i in lazaridou_dict[row[3]]]))
			derived_dict[row[1]]['lazaridou']['derived'].append(np.array([float(i.strip(',').strip('[').strip(']')) for i in lazaridou_dict[row[2]]]))
			all_words_derived[row[1]].append(row[2])

	f.close()

	for key in derived_dict.keys():
		try :
			for item in derived_dict[key].keys():
				for origin in derived_dict[key][item].keys():
					
					derived_dict[key][item][origin] = np.array(derived_dict[key][item][origin])
					if len(derived_dict[key][item][origin])==0:
						del(derived_dict[key])
						del(all_words_derived[key])
		except :
			continue		

	# with open('derived_words_dict.pkl','wb') as f:
	# 	pickle.dump(derived_dict,f)

	# with open('all_words_derived.pkl','wb') as g:
	# 	pickle.dump(all_words_derived,g)

	file = open(fastextFile,'wb')
	for item in all_words_derived:
		count=0
		for words in all_words_derived[item]:
			str1 = ' '.join(str(e) for e in derived_dict[item]['fastText']['derived'][count])
			count += 1
			print >> file,words+' '+str1
			# print words+' '+str1

	file = open(lazaridouFile,'wb')
	for item in all_words_derived:
		count=0
		for words in all_words_derived[item]:
			str1 = ' '.join(str(e) for e in derived_dict[item]['lazaridou']['derived'][count])
			count += 1
			print >> file,words+' '+str1
			# print words+' '+str1
	return all_words_derived,derived_dict









