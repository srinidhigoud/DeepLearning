from bs4 import BeautifulSoup
import glob

def convertDB():
	xmlDirPath = '../Data_XML/**/**/*.nxml'
	txtDirPath = '../Data_TXT/'
	for filePath in glob.glob(xmlDirPath, recursive=True):
		text = open(filePath, 'r').read()
		soup = BeautifulSoup(text, "lxml")

		docId = soup.find("article-id",{"pub-id-type":"pmid"})
		if docId is None:
			continue
		else:
			docId = docId.get_text()
		content = str(soup.get_text())

		fileName = txtDirPath + docId + '.txt'
		wrt = open(fileName, 'w')
		wrt.write(content)

if __name__ == '__main__':
	convertDB()
