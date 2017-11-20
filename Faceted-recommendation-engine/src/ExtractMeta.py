import sys
import re
import io
import os
import os.path
import nltk
import nltk.data
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktParameters
from nltk.tokenize.punkt import PunktSentenceTokenizer

# Lists to print in metadata file 
"""citerID_list = []
citedID_list = []
SectionNamelist = []
sentence_list = []"""



def filter_list(sentences):
	new_list=[]
	count = 0
	for sent in sentences:
		match= re.findall(r'\(cit\)',sent)
		
		if len(match) > 0:
			count = len(match)
			for i in range(count):
				new_list.append(sent)
	return new_list

def getSentences(paragraph):

	
	unicode_data= paragraph.decode("utf-8")
	data= "".join([i if ord(i) < 128  else "" for i in unicode_data])
	
	##tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	punkt_params = PunktParameters()
	punkt_params.abbrev_types = set(['al',"inc","mr","dr","mrs","prof", "etal"])
	splitter = PunktSentenceTokenizer(punkt_params)

	sentences=splitter.tokenize(data)
	
	sentences1=filter_list(sentences)
	##print sentences1,"\n----------------------------------------------------------------------------"
	return sentences1

def find_pmid(refer_ID):
	
	ref = soup.find("ref",{"id":refer_ID})
	if ref is None :
		return "None" 
	if ref.find("pub-id",{"pub-id-type":"pmid"}) is not None:
		return ref.find("pub-id",{"pub-id-type":"pmid"}).text
	else :
		return "None"
	
						
def SectionParse(section_list, citedID_list, sentence_list, SectionNamelist):

	for section in section_list:
		try :
			for paragraph in section.find_all("p"):
				if paragraph.find("xref",{"ref-type":"bibr"}) is not None:
					
					for xref_tag in paragraph.find_all("xref",{"ref-type":"bibr"}):

						## Extract the PMID's
						refer_ID = xref_tag['rid']
						
						pmid = find_pmid(refer_ID)    ## Get the PMID of the referred papers

						citedID_list.append(pmid)      
						SectionNamelist.append(section.title.text)       ## Get the name of the sections 
						try :
							xref_tag.string = "(cit)"
						except TypeError :
							xref_tag.string = "(cit)"		

					sentence_list.extend(getSentences(paragraph.get_text().encode("utf-8",errors="ignore")))
					## Get the list of eligible sentences throgh getSentences method	
		except AttributeError :
			pass    ## Obtaining NavigableString 

def createmetadata(soup,new_file) :

	Citation_ID = soup.find("article-id",{"pub-id-type":"pmid"}) ## get the Citer Id for the paper 

	if Citation_ID is None:
		return

	section_list = [] ## List to iterate over all the sections 
	SectionNamelist = []
	citedID_list = []
	sentence_list = []
	citerID_list = []
	section1 = soup.find("sec")
	section_list.append(section1)

	if soup.sec.next_siblings is not None:
		for siblings in soup.sec.next_siblings:
			section_list.append(siblings)
	
	SectionParse(section_list, citedID_list, sentence_list, SectionNamelist)     ## method to parse the whole document 

	
	i=0
	try :
		while i<=len(citedID_list):
			citerID_list.append(Citation_ID.text)
			i += 1	
	except :
		return 

	##  Writing the particular file 
	
	##with io.open(datafile, 'a') as new_file:

	
	for i in range(len(citedID_list)):
		new_file.write(citerID_list[i]+"\t|\t"+citedID_list[i].encode("utf-8")+"\t|\t"+sentence_list[i].replace("\n","").replace("\r","")+"\t|\t"+SectionNamelist[i]+"\n")
		


""""handler = open("IDCases_2013_Oct_18_1(1)_1.nxml").read()
soup = BeautifulSoup(handler,"xml")
createmetadata(soup)"""

## Loop to read all the files in the directory one by one 
true =0
false =0

for dir in os.listdir("/media/tusharg/LENOVO_USB_HDD/O-Z"):           ## List all the directories
	
	print "Current Directory",dir
	datafile = dir+".txt"
	print datafile,"\n"

	for filename in os.listdir(os.path.join("/media/tusharg/LENOVO_USB_HDD/O-Z",dir)):     ## Listing all the files in the diectory
		
		filename1 = os.path.join("/media/tusharg/LENOVO_USB_HDD/O-Z",dir)
		filename1 = os.path.join(filename1,filename)    ## Obtain the final filename 


		print filename1
		handler = open(filename1).read()
		soup = BeautifulSoup(handler,"xml")
		new_file = io.open(datafile,'a') # Open a datafile for the directory 

		try :
			createmetadata(soup,new_file)
			print " 1"
			true = true+1
		except Exception as ex:
			false = false+1
			with open("errorsO-Z.txt","a") as f:
				f.write("{}\t,{}\t,{}\t,{}\n".format(dir,filename1,type(ex).__name__,ex.args))
			
			print " 0"
			
			f.close()
	new_file.close()
print "\nSuccess :",true,"\n"

print "Fail :",false,"\n" 