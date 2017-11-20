from pprint import pprint
import glob
import re, math
from collections import Counter

def text_to_vector(text):
	WORD = re.compile(r'\w+')
	words = WORD.findall(text)
	return Counter(words)

def get_cosine(vec1, vec2):
	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum([vec1[x] * vec2[x] for x in intersection])
	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	denominator = math.sqrt(sum1) * math.sqrt(sum2)
	if not denominator:
		return 0.0
	else:
		return float(numerator) / denominator

def findCosineSimilarity(id1, id2):
	dirPath = '../Data_TXT/'
	
	f1Name = dirPath + str(id1) + '.txt'
	f2Name = dirPath + str(id2) + '.txt'

	doc1 = open(f1Name, 'r').read()
	doc2 = open(f2Name, 'r').read()

	vec1 = text_to_vector(doc1)
	vec2 = text_to_vector(doc2)

	cosineSim = get_cosine(vec1, vec2)
	return cosineSim
