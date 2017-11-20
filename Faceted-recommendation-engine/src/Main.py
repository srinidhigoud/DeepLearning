from SearchEngine import *
from Graph import *


if __name__ == '__main__':

	# Preparing Graph

	print 'Creating Graph...'
	G = networkx.Graph()
	for filePath in glob.glob('../Data_Meta/**/*.txt'):
		with open(filePath, 'r') as f:
			for line in f:
				# print line
				
				tup = line.split('\t|\t')

				if len(tup) == 4:
					citer = tup[0]
					cited = tup[1]
					cited_sentence = tup[2]
					fac_type = tup[3]
				else:
					continue

				G.add_edge(citer, cited)
				fac_type = fac_type.lower()
				##print(fac_type)
				if fac_type.find('introduction') != -1 or fac_type.find('background') != -1:
					fac_type = 1
				elif fac_type.find('alternative') != -1 or fac_type.find('approach') != -1:
					fac_type = 2
				elif fac_type.find('discussion') != -1 or fac_type.find('conclusion') != -1 or fac_type.find('result') != -1:
					fac_type = 4
				else:
					fac_type = 3
				##print(fac_type)
				G.edge[citer][cited]['fac_type'] = fac_type

	print 'Graph Creation Successful'
	#print G.edges()

	print 'Indexing...'
	indexDB()
	isIndexed = True
	print 'Indexing Successful'
	# Indexing
	while True:
	
		queryStr = raw_input('Enter Query: ')
		docIds = searchDB(queryStr)

		i = 0
		for id in docIds:
			print(str(i) + ' -> ' + docIds[i])
			i += 1

		usrChoice = int(raw_input('Which Document do you Want to Read >> '))

		query_id = docIds[usrChoice]

		G.add_node(query_id)
		G1 = create_subgraph(G,query_id,1)
		# cosine_list1 = getCosineSimilarity(query_id, G1)
		G2 = create_subgraph(G,query_id,2)
		# cosine_list2 = getCosineSimilarity(query_id, G2)
		G3 = create_subgraph(G,query_id,3)
		# cosine_list3 = getCosineSimilarity(query_id, G3)
		G4 = create_subgraph(G,query_id,4)
		# cosine_list4 = getCosineSimilarity(query_id, G4)

		# print('G1 ----------->')
		# print(G1.edges())
		# print('G2 ----------->')
		# print(G2.edges())
		# print('G3 ----------->')
		# print(G3.edges())
		# print('G4 ----------->')
		# print(G4.edges())
		# print('G1 ----------->')
		# print(find_relevant_docs_in_subgraph(G1,query_id))
		# print('G2 ----------->')
		# print(find_relevant_docs_in_subgraph(G2,query_id))
		# print('G3 ----------->')
		# print(find_relevant_docs_in_subgraph(G3,query_id))
		# print('G4 ----------->')
		# print(find_relevant_docs_in_subgraph(G4,query_id))
		
		usrChoice = 0
		while usrChoice != 5:
			print 'Select the Facet:'
			print '1 -> Introduction'
			print '2 -> Alternative Aproaches'
			print '3 -> Methods'
			print '4 -> Conclusion'
			print '5 -> New Query'

			usrChoice = int(raw_input('Enter your Choice >> '))
			print
			if usrChoice == 1:
				relevant_docs = find_relevant_docs_in_subgraph(G1,query_id)
			elif usrChoice == 2:
				relevant_docs = find_relevant_docs_in_subgraph(G2,query_id)
			elif usrChoice == 3:
				relevant_docs = find_relevant_docs_in_subgraph(G3,query_id)
			elif usrChoice == 4:
				relevant_docs = find_relevant_docs_in_subgraph(G4,query_id)
			elif usrChoice == 5:
				break
			else:
				print 'Error: Invalid Choice!!'


			if relevant_docs == None:
				print 'No Facets Found'
			else:
				numDocs = len(relevant_docs)

				if numDocs == 0:
					print 'No Facets Found'
				else:
					numFacets = min(4, numDocs)
					random_walk_rankings = [0,1,2,3]
					cosine_rankings = [0,2,1,3]
					
					random_walk_rankings = random_walk_rankings[0:numFacets]
					cosine_rankings = cosine_rankings[0:numFacets]


					ranks  = numpy.vstack((random_walk_rankings, cosine_rankings))
					dist, aggr = rank_aggregate(ranks)
					print("Final top ranked faceted recommendations are {}".format(", ".join(relevant_docs[i] for i in numpy.argsort(aggr))))