import csv
import io
import nltk
import utils
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import ast
import numpy as np
import math

# globals
cEN = utils.contractionsEN()
swEN = utils.stopWordsEN()
lemmatizer = WordNetLemmatizer()
punctuationCharacters = ['.', '<', '>', '-', '{', '}', '[', ']', '/', ':', ';', '\\', '@', '!', '#', '$', '%', '^', '&',
                         '*', '(', ')']
firstLine = "id\tconf\ttitle\tauthor\tyear\tabstract\tconf_short\ttheme"

def get_wordnet_pos(w):
    if w.startswith('J'):
        return wordnet.ADJ
    elif w.startswith('V'):
        return wordnet.VERB
    elif w.startswith('N'):
        return wordnet.NOUN
    elif w.startswith('R'):
        return wordnet.ADV
    return None

def isAWord(word):
    if len(word) < 2:
        return False
    for punc in punctuationCharacters:
        if word.startswith(punc) or word.endswith(punc):
            return False
    return True

# offered by master
with open('test.txt', 'r') as f:
	first_line = f.readline()
	second_line = f.readline()

# start of worker code
s = io.StringIO(first_line + "\n" + second_line)
reader = csv.DictReader(s, delimiter='\t')
line = next(reader)
phrases = sent_tokenize(line['abstract'])
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
	#print(finalWords)
	for word in finalWords:
		if wordsResult.get(word) is None:
			wordsResult[word] = 1
		else:
			wordsResult[word] = wordsResult[word] + 1
# we save document's id
workerResult = {}
workerResult[int(line['id'])] = wordsResult
#print(wordsResult) # return wordsResult
# end of worker code


wordsToId = {}
idToWords = {}
lastWordId = 0
# start of master code

docSize = {}
mostFreqValue = {}
nrDocsPerTerm = {}
# first we get the id-to-words bidirectional map
for docId, wordsMap in workerResult.items():
	dSize = 0
	mostFrequentValue = 0
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
			nrDocsPerTerm[curWordId] = nrDocsPerTerm[curWordId] + 1
		# document size
		dSize = dSize + freq
		# most frequent word's frequency from that document
		if mostFrequentValue < freq:
			mostFrequentValue = freq
	mostFreqValue[docId] = mostFrequentValue
	docSize[docId] = dSize

matriSize = (len(workerResult), len(wordsToId))

mainMatrix = np.zeros(matriSize)

totalDocs = 1
avgLen = nrDocsPerTerm[1]

for i in range(matriSize[0]):
	for j in range(matriSize[1]):
		mainMatrix[i, j] = 0 if workerResult[i + 1][idToWords[j]] is None else workerResult[i + 1][idToWords[j]]

# again map stage for worker
for i in range(matriSize[0]):
	for j in range(matriSize[1]):
		tf = 0.5 + 0.5 * mainMatrix[i, j] / mostFreqValue[i + 1]
		idf = 1 + math.log (totalDocs / float(nrDocsPerTerm[j]))
		okapi = idf * tf * 2.6 / float(tf + 1.6 * (0.25 + 0.75 * docSize[i + 1] / avgLen))
		mainMatrix[i, j] = okapi

query1 = ['object-oriented', 'deductive', 'useless', 'language']
query2 = ['language', 'deductive', 'database']

# top-k-documents - query1
topkdocq1 = []
for i in range(matriSize[0]):
	topkval = 0
	for j in range(len(query1)):
		wordId = wordsToId.get(query1[j])
		if wordId is None:
			continue
		topkval = topkval + np.sum(mainMatrix[i, wordId])
	topkdocq1.append((i + 1, topkval))
topkdocq1 = sorted(topkdocq1, key=lambda x: x[1], reverse=True)

# top-k-documents - query2
topkdocq2 = []
for i in range(matriSize[0]):
	topkval = 0
	for j in range(len(query2)):
		wordId = wordsToId.get(query2[j])
		if wordId is None:
			continue
		topkval = topkval + np.sum(mainMatrix[i, wordId])
	topkdocq2.append((i + 1, topkval))
topkdocq2 = sorted(topkdocq2, key=lambda x: x[1], reverse=True)
print(str(topkdocq1[:5]) + "\n\n" + str(topkdocq2[:5]))
# top-k-words TODO