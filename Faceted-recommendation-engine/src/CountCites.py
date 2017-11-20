import glob
import pprint

Citations = [0, 0, 0, 0]

dataBasePath = '../metaData/*.txt'
for filePath in glob.glob(dataBasePath):
	with open(filePath, 'r') as f:
		for line in f:
			tups = line.split('\t')
			if len(tups) != 3:
				continue
			z = tups[2]
			Citations[int(z)-1] += 1
			
pprint.pprint(Citations)
