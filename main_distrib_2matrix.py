import csv
import io
import nltk
import utils
import sys
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import math
from pyspark import SparkContext, SparkConf
from scipy.sparse import csr_matrix

# globals
cEN = utils.contractionsEN()
swEN = utils.stopWordsEN()
firstLine = "id\tconf\ttitle\tauthor\tyear\tabstract\tconf_short\ttheme"
lemmatizer = WordNetLemmatizer()
punctuationCharacters = ['.', '<', '>', '-', '{', '}', '[', ']', '/', ':', ';', '\\', '@', '!', '#', '$', '%', '^', '&',
                         '*', '(', ')']
nrDocsPerTerm = {}
avgLen = 0
totalDocs = 0

def get_wordnet_pos(w):
    if w.startswith('J'):
    	return 'a'
        #return wordnet.ADJ
    elif w.startswith('V'):
    	return 'v'
        #return wordnet.VERB
    elif w.startswith('N'):
    	return 'n'
        #return wordnet.NOUN
    elif w.startswith('R'):
    	return 'r'
        #return wordnet.ADV
    return None

def isAWord(word):
    if len(word) < 2:
        return False
    for punc in punctuationCharacters:
        if word.startswith(punc) or word.endswith(punc):
            return False
    return True

def mapToWordsFreq(line):
	s = io.StringIO(firstLine + "\n" + line)
	reader = csv.DictReader(s, delimiter='\t')
	lineR = next(reader)
	phrases = sent_tokenize(lineR['abstract'])
	wordsResult = {}
	for phrase in phrases:
		# expand and remove digits
		for contr, expand in cEN.items():
			phrase = phrase.replace(contr, expand)
		phrase = ''.join(c if not c.isdigit() else "" for c in phrase)
		# get words
		wordsWithoutPOS = word_tokenize(phrase)
		# remove stop words
		wordsWithoutPOS = [w for w in wordsWithoutPOS if w not in swEN]
		# get part of speech for each word
		wordsWithPOS = nltk.pos_tag([s.lower() for s in wordsWithoutPOS])
		# get relevant words
		finalWords = [lemmatizer.lemmatize(w[0], get_wordnet_pos(w[1])) for w in wordsWithPOS if (get_wordnet_pos(w[1]) is not None) and isAWord(w[0])]
		for word in finalWords:
			if wordsResult.get(word) is None:
				wordsResult[word] = 1
			else:
				wordsResult[word] = wordsResult[word] + 1
	# we save document's id
	workerResult = {}
	workerResult[int(lineR['id'])] = wordsResult
	return workerResult

def getOkapiMap(matrLine):
	maxTerm = float(max(matrLine))
	docuSize = float(np.sum(matrLine))
	for j in range(len(matrLine)):
		matrLineJ = matrLine[j]
		if matrLineJ == 0:
			okapi = 0
		else:
			tf = 0.5 + 0.5 * matrLine[j] / maxTerm
			idf = 1 + math.log(totalDocs / float(nrDocsPerTerm[j]))
			okapi = idf * tf * 2.6 / float(tf + 1.6 * (0.25 + 0.75 * docuSize / float(avgLen)))
		matrLine[j] = okapi
	return matrLine

if __name__ == '__main__':

	sc = SparkContext(appName = "Okapi BM 25 - top-k-words and top-k-docs")
	linesRDD = sc.textFile("hdfs://hadoop-master:9000/user/hadoop/mediumlowbooks.csv")
	noHeaderRDD = linesRDD.zipWithIndex().filter(lambda rowindex: rowindex[1] > 0).keys()
	wordFreqs = noHeaderRDD.map(mapToWordsFreq).collect()

	wordsToId = {}
	idToWords = {}
	lastWordId = 0
	
	for workerResultDict in wordFreqs:
		for docId, wordsMap in workerResultDict.items():
			for word, freq in wordsMap.items():
				# words to id by-map
				curWordId = wordsToId.get(word)
				if curWordId is None:
					curWordId = lastWordId
					wordsToId[word] = lastWordId
					idToWords[lastWordId] = word
					lastWordId = lastWordId + 1
				# number of documents that contain each term
				if nrDocsPerTerm.get(curWordId) is None:
					nrDocsPerTerm[curWordId] = 1
				else:
					nrDocsPerTerm[curWordId] += 1
				avgLen = avgLen + freq

	# matriSize = (len(wordFreqs), len(wordsToId)) # old code
	matri1Size = (int(len(wordFreqs) / 2), len(wordsToId))
	matri2Size = (len(wordFreqs) - int(len(wordFreqs) / 2), len(wordsToId))
	totalDocs = len(wordFreqs)
	avgLen = avgLen / float(totalDocs)

	# mainMatrix = np.zeros(matriSize) # old code
	matri1 = np.zeros(matri1Size)
	matri2 = np.zeros(matri2Size)

	# for i in range(matriSize[0]):
	# 	resultLine = []
	# 	for j in range(matriSize[1]):
	# 		freq = wordFreqs[i][i + 1].get(idToWords[j])
	# 		resultLine.append(0 if freq is None else freq)
	# 	mainMatrix[i, :] = resultLine

	for i in range(matri1Size[0]):
		resultLine1 = []
		resultLine2 = []
		for j in range(matri1Size[1]):
			freq1 = wordFreqs[i][i + 1].get(idToWords[j])
			freq2 = wordFreqs[i + matri1Size[0]][i + matri1Size[0] + 1].get(idToWords[j])
			resultLine1.append(0 if freq1 is None else freq1)
			resultLine2.append(0 if freq2 is None else freq2)
		matri1[i, :] = resultLine1
		matri2[i, :] = resultLine2

	for i in range(2 * matri1Size[0], len(wordFreqs)):
		resultLine = []
		for j in range(matri1Size[1]):
			freq = wordFreqs[i][i + 1].get(idToWords[j])
			resultLine.append(0 if freq is None else freq)
		matri2[i - matri1Size[0], :] = resultLine

	# get okapi matrix
	# okapiMatrix = sc.parallelize(mainMatrix).map(getOkapiMap).collect()
	# mainMatrix = np.matrix(okapiMatrix)
	okapi1Matrix = sc.parallelize(matri1).map(getOkapiMap).collect()
	okapi2Matrix = sc.parallelize(matri2).map(getOkapiMap).collect()

	matri1 = np.matrix(okapi1Matrix)
	matri2 = np.matrix(okapi2Matrix)

	# top-k-docs
	queries = [['object-oriented'], ['object-oriented', 'deductive'], ['object-oriented', 'deductive', 'database'], ['object-oriented', 'deductive', 'database', 'perform'], ['object-oriented', 'deductive', 'database', 'perform', 'language']]
	debugFile = open('/home/hadoop/resultEmanuel.txt', 'w')
	for query in queries:
		topkdoc = []
		for i in range(matri1Size[0]):
			topkval = 0
			for j in range(len(query)):
				wordId = wordsToId.get(query[j])
				if wordId is None:
					continue
				topkval += matri1[i, wordId]
				#topkval += np.sum(mainMatrix[i, wordId])
			topkdoc.append((i + 1, topkval))
		for i in range(matri2Size[0]):
			topkval = 0
			for j in range(len(query)):
				wordId = wordsToId.get(query[j])
				if wordId is None:
					continue
				topkval += matri2[i, wordId]
			topkdoc.append((i + matri1Size[0] + 1, topkval))
		topkdoc = sorted(topkdoc, key=lambda x: x[1], reverse=True)
		debugFile.write(str(topkdoc[:10]) + '\n')

	# top-k-words
	topkwords = []
	for k in range(matri1Size[1]):
		topkwords.append((idToWords[k], np.sum(matri1[:, k]) + np.sum(matri2[:, k])))
		#topkwords.append((idToWords[k], np.sum(mainMatrix[:, k])))
	topkwords = sorted(topkwords, key=lambda x: x[1], reverse=True)
	debugFile.write(str(topkwords[:10]) + "\n")
	debugFile.close()

	sc.stop()
'''
[(1249, 7.1839180582408249), (1281, 7.1839180582408249), (1363, 7.1839180582408249), (2473, 7.1839180582408249), (2483, 7.1839180582408249), (28, 7.0170983568917285), (52, 7.0170983568917285), (71, 7.0170983568917285), (122, 7.0170983568917285), (178, 7.0170983568917285)]
[(1, 16.119521990347589), (1181, 15.76181885893925), (2536, 15.76181885893925), (1369, 13.909810107426237), (755, 13.628582660452572), (1871, 9.8563833288658351), (2520, 9.7020329338262528), (759, 9.4767394068986466), (762, 9.4767394068986466), (829, 9.4767394068986466)]
[(1, 20.40915482858826), (2536, 19.956261833625064), (1369, 17.611407353686822), (1181, 15.76181885893925), (2520, 14.195623371797524), (759, 13.865982967752128), (762, 13.865982967752128), (829, 13.865982967752128), (871, 13.865982967752128), (1853, 13.865982967752128)]
[(1, 20.40915482858826), (2536, 19.956261833625064), (1369, 17.611407353686822), (1181, 15.76181885893925), (2520, 14.195623371797524), (759, 13.865982967752128), (762, 13.865982967752128), (829, 13.865982967752128), (871, 13.865982967752128), (1853, 13.865982967752128)]
[(1, 27.110795039278898), (2536, 26.509188100417642), (1369, 23.394366848069982), (762, 20.723243775647944), (755, 19.294622734513148), (775, 16.711366140484991), (1181, 15.76181885893925), (2520, 14.195623371797524), (811, 14.0152481381134), (3254, 13.874359164787545)]
[('database', 2521.2491137660854), ('system', 2133.5622971266912), ('data', 2097.4551469668904), ('algorithm', 1856.6721165530757), ('problem', 1576.7908022352676), ('model', 1522.6885789785988), ('abstract', 1344.9426863276319), ('query', 1343.5637367218869), ('extend', 1331.260483102913), ('object', 1225.4555566073661)]
'''