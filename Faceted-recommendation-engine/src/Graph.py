import networkx
import glob
import numpy
from itertools import combinations, permutations
# from Cosine import *

def create_subgraph(G,query_id,fac_type):
	
	H = networkx.Graph()
	H.add_node(query_id)
	for neighbor in G.neighbors(query_id):
		if G.edge[query_id][neighbor]['fac_type'] == fac_type:
			H.add_edge(query_id,neighbor,weight=1.0)
			for node in G.neighbors(neighbor):
				if G.edge[neighbor][node]['fac_type'] == fac_type:
					H.add_edge(neighbor,node,weight=1.0)
		else:
			for node in G.neighbors(neighbor):
				if G.edge[neighbor][node]['fac_type'] == fac_type:
					H.add_edge(neighbor,node,weight=0.3)	
	return H

def find_relevant_docs_in_subgraph(subG,query_id):
	
	n = subG.number_of_nodes()
	if n == 1:
		return
	A = networkx.to_numpy_matrix(subG,weight='weight')
	nodes_list = subG.nodes()
	##R = numpy.random.random((n,1))
	R = numpy.zeros((n,1))
	R.fill(1.0/n)
	sum_array = numpy.sum(A,axis=1)
	for i in range(len(A)):
		if sum_array[i] != 0.0:
			for j in range(len(A[i])):
				A[i][j] = A[i][j]/sum_array[i]
	E = numpy.zeros((n,1))
	index = nodes_list.index(query_id)
	E[index][0] = 1.0/n
	##print('==========Before===========')
	##print(R)
	for i in range(500):
		##print(R)
		R = 0.6 * numpy.dot(A,R) + 0.4 * E
	##print('===========After =========')
	##print(R)
	num_found_docs = 4
	if n < 5:
		num_found_docs = n-1
	max_id = R.argmax()
	document_ids = [None] * num_found_docs
	for i in range(num_found_docs):
		R[max_id][0] = 0.0
		max_id = R.argmax()
		document_ids[i] = nodes_list[max_id]
	return document_ids

def kendalltau_dist(rank_a, rank_b):
    tau = 0
    n_candidates = len(rank_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += (numpy.sign(rank_a[i] - rank_a[j]) == -numpy.sign(rank_b[i] - rank_b[j]))
    return tau

def rank_aggregate(ranks):
    min_dist = numpy.inf
    best_rank = None
    n_voters, n_candidates = ranks.shape
    for candidate_rank in permutations(range(n_candidates)):
        dist = numpy.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
        if dist < min_dist:
            min_dist = dist
            best_rank = candidate_rank
    return min_dist, best_rank

# def getCosineSimilarity(qid, graph):
# 	cosine_list = list()

# 	for node in graph.nodes():
# 		cosine_list.append(findCosineSimilarity(qid, node))

# 	cosine_list.sort(reverse=True)
# 	return cosine_list

# if __name__ == "__main__":
	
# 	G = networkx.Graph()
# 	for filePath in glob.glob('./Data/*.txt'):
# 		with open(filePath, 'r') as f:
# 			for line in f:
# 				citer, cited, cited_sentence, fac_type = line.split('\t|\t')
# 				G.add_edge(citer, cited)
# 				fac_type = fac_type.lower()
# 				##print(fac_type)
# 				if fac_type.find('introduction') != -1 or fac_type.find('background') != -1:
# 					fac_type = 1
# 				elif fac_type.find('alternative') != -1 or fac_type.find('approach') != -1:
# 					fac_type = 2
# 				elif fac_type.find('discussion') != -1 or fac_type.find('conclusion') != -1 or fac_type.find('result') != -1:
# 					fac_type = 4
# 				else:
# 					fac_type = 3
# 				##print(fac_type)
# 				G.edge[citer][cited]['fac_type'] = fac_type
	
# 	# print(G.edges())

	
# 	query_id = '20810945'
# 	G1 = create_subgraph(G,query_id,1)
# 	cosine_list1 = getCosineSimilarity(query_id, G1)
# 	G2 = create_subgraph(G,query_id,2)
# 	cosine_list1 = getCosineSimilarity(query_id, G1)
# 	G3 = create_subgraph(G,query_id,3)
# 	cosine_list1 = getCosineSimilarity(query_id, G1)
# 	G4 = create_subgraph(G,query_id,4)
# 	cosine_list1 = getCosineSimilarity(query_id, G1)

# 	print('G1 ----------->')
# 	print(G1.edges())
# 	print('G2 ----------->')
# 	print(G2.edges())
# 	print('G3 ----------->')
# 	print(G3.edges())
# 	print('G4 ----------->')
# 	print(G4.edges())
# 	print('G1 ----------->')
# 	print(find_relevant_docs_in_subgraph(G1,query_id))
# 	print('G2 ----------->')
# 	print(find_relevant_docs_in_subgraph(G2,query_id))
# 	print('G3 ----------->')
# 	print(find_relevant_docs_in_subgraph(G3,query_id))
# 	print('G4 ----------->')
# 	print(find_relevant_docs_in_subgraph(G4,query_id))
	
# 	relevant_docs = find_relevant_docs_in_subgraph(G2,query_id)
# 	random_walk_rankings = [0,1,2,3]
# 	cosine_rankings = [0,2,1,3]
# 	ranks  = numpy.vstack((random_walk_rankings, cosine_rankings))
# 	dist, aggr = rank_aggregate(ranks)
# 	print("A Kemeny-Young aggregation with score {} is: {}".format(dist,", ".join(relevant_docs[i] for i in numpy.argsort(aggr))))