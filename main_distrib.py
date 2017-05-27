import csv
import io
import nltk
import utils
import sys
import time
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

wordsToId = {}
idToWords = {}
lastWordId = 0

def get_wordnet_pos(w):
    if w.startswith('J'):
        return 'a'
    elif w.startswith('V'):
        return 'v'
    elif w.startswith('N'):
        return 'n'
    elif w.startswith('R'):
        return 'r'
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

def getOkapiMapDictToList(docu):
    result = [0] * lastWordId
    docu_id = next(iter(docu))
    docu_dict = docu[docu_id]
    if len(docu_dict.values()) == 0:
        return result
    maxTerm = float(max(docu_dict.values()))
    docuSize = float(sum(docu_dict.values()))

    for word_freq in docu_dict.items():
        result[wordsToId[word_freq[0]]] = word_freq[1]

    for j in range(len(result)):
        resultJ = result[j]
        if resultJ == 0:
            continue
        tf = 0.5 + 0.5 * result[j] / maxTerm
        idf = 1 + math.log(totalDocs / float(nrDocsPerTerm[j]))
        okapi = idf * tf * 2.6 / float(tf + 1.6 * (0.25 + 0.75 * docuSize / float(avgLen)))
        result[j] = okapi
    return result

if __name__ == '__main__':
    for execu in range(0, 1):

        nrDocsPerTerm = {}
        avgLen = 0
        totalDocs = 0

        wordsToId = {}
        idToWords = {}
        lastWordId = 0

        program_start_time = time.time()
        sc = SparkContext(appName = "Okapi BM 25 - top-k-words and top-k-docs")
        linesRDD = sc.textFile("hdfs://hadoop-master:9000/user/hadoop/books.csv")
        noHeaderRDD = linesRDD.zipWithIndex().filter(lambda rowindex: rowindex[1] > 0).keys()
        wordFreqs = noHeaderRDD.map(mapToWordsFreq).collect()
        
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

        matriSize = (len(wordFreqs), len(wordsToId))
        totalDocs = len(wordFreqs)
        avgLen = avgLen / float(totalDocs)

        # get okapi matrix
        okapiMatrix = sc.parallelize(wordFreqs).map(getOkapiMapDictToList).collect()
        mainMatrix = np.matrix(okapiMatrix)

        # top-k-docs
        queries = [['object-oriented'], ['object-oriented', 'deductive'], ['object-oriented', 'deductive', 'database'], ['object-oriented', 'deductive', 'database', 'perform'], ['object-oriented', 'deductive', 'database', 'perform', 'language']]
        resultFile = open('/home/hadoop/resultEmanuel.txt', 'a')
        # resultFile.write('Execution ' + str(execu) + "\n")
        for query in queries:
            query_start_time = time.time()
            topkdoc = []
            for i in range(matriSize[0]):
                topkval = 0
                for j in range(len(query)):
                    wordId = wordsToId.get(query[j])
                    if wordId is None:
                        continue
                    topkval += mainMatrix[i, wordId]
                topkdoc.append((i + 1, topkval))
            topkdoc = sorted(topkdoc, key=lambda x: x[1], reverse=True)
            resultFile.write(str(topkdoc[:10]) + '\n')
            query_end_time = time.time()
            resultFile.write('Total time for this query: ' + str(query_end_time - query_start_time) + ' seconds.\n')

        # top-k-words
        topkwords_start_time = time.time()
        topkwords = []
        for k in range(matriSize[1]):
            topkwords.append((idToWords[k], np.sum(mainMatrix[:, k])))
        topkwords = sorted(topkwords, key=lambda x: x[1], reverse=True)
        topkwords_end_time = time.time()
        resultFile.write(str(topkwords[:10]) + "\n")
        program_end_time = time.time()
        resultFile.write('Top-k-words time: ' + str(topkwords_end_time - topkwords_start_time) + ' seconds\n')
        resultFile.write('Total time: ' + str(program_end_time - program_start_time) + ' seconds.\n\n\n')
        resultFile.close()

        sc.stop()