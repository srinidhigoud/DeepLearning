import os
import xapian
import glob
import random
import shutil
import pprint
from TextMachine import *

database = None
isIndexed = False
queryStr = None

databasePath = os.path.abspath('../xapian-database')

def indexDB():
	global database
	global isIndexed

	if isIndexed == True:
		return

	if os.path.exists(databasePath):
		database = xapian.WritableDatabase(databasePath, xapian.DB_OPEN)
	else:
		database = xapian.WritableDatabase(databasePath, xapian.DB_CREATE)

		indexer = xapian.TermGenerator()
		indexer.set_stemmer(xapian.Stem('english'))

		xapian_file_name = 0

		for filePath in glob.glob('../Data_TXT/*.txt'):
	    
		    content = open(filePath).read()
		    
		    document = xapian.Document()
		    document.set_data(content)
		    
		    fileName = os.path.basename(filePath)
		    pathArr = fileName.split('.')
		    docuId = pathArr[len(pathArr)-2]
		    document.add_value(xapian_file_name, docuId)
		    
		    indexer.set_document(document)
		    indexer.index_text(content)
		    
		    database.add_document(document)
		    
		# Save Changes
		database.flush()
		isIndexed = True

def searchDB(queryStr, withContent=False, extractLength = 32):
    # Parse Query
    queryParser = xapian.QueryParser()
    queryParser.set_stemmer(xapian.Stem('english'))
    queryParser.set_database(database)
    queryParser.set_stemming_strategy(xapian.QueryParser.STEM_SOME)
    
    query = queryParser.parse_query(queryStr)
    
    offset, limit = 0, 5
    
    # Start Query Session
    enquire = xapian.Enquire(database)
    enquire.set_query(query)
    
    docIds = list()

    # Display Matches
    matches = enquire.get_mset(offset, limit)
    print ('*' * 50)
    for match in matches:
        print ('-' * 50)

        pmId = match.document.get_value(0)
        docIds.append(match.document.get_value(0))
        print 'Rank/ID: %s, docID: %s' %(match.rank, pmId)
        print ('-' * 50)
        # Process
        content = match.document.get_data()
        extract = TextMachine(extractLength, '*%s*').process(queryStr, content)
        print extract.replace('\n', ' ')
        print ('-' * 50)

    print ('*' * 50)
    print 'No. of Docs matching Query: %s' % matches.get_matches_estimated()
    print 'No. of Docs Returned: %s' % matches.size()

    return docIds

# if __name__ == '__main__':
# 	print 'Indexing...'
# 	indexDB()
# 	isIndexed = True
# 	print 'Indexing Successful'
# 	queryStr = raw_input('Enter Query: ')
# 	docIds = searchDB()

# 	i = 0
# 	for id in docIds:
# 		print(str(i) + ' -> ' + docIds[i])
# 		i += 1

# 	usrChoice = int(raw_input('Which Document do you Want to Read >> '))

# 	documentId = docIds[usrChoice]

# 	print documentId
