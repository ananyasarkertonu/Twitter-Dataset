#Random Forest
import pandas as pd 
from nltk.tokenize import WordPunctTokenizer
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify import SklearnClassifier
from nltk.corpus import names
import numpy as np
import time
import matplotlib.pyplot as plt
from pandas import *


start = time.time()
num=0
df = read_csv("./traindata.csv")
df1 = df['tag']
df2 = df['content']
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))


def clean(s):
    def tweet_cleaner(text):
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        stripped = re.sub(combined_pat, '', souped)
        try:
            clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            clean = stripped
        letters_only = re.sub("[^a-zA-Z]", " ", clean)
        lower_case = letters_only.lower()
        words = tok.tokenize(lower_case)
        return (" ".join(words)).strip()
    
    stopWords = set(stopwords.words('english'))
    ps = PorterStemmer()
    li=[]
    test_result = []
    wordsFiltered = []
    b=tweet_cleaner(s)
    test_result.append(b)
    words = word_tokenize(tweet_cleaner(s))
    #print(words)
    wordsFiltered=[]
    
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    for word in wordsFiltered:
        l=ps.stem(word)
        li.append(l)
    
    #print(li)
    return li


p=[]
s=[]
n=[]
c=[]
for t in range(0,len(df2)):
    if df1[t]=='politics':
        sen=df2[t]
        p=p+clean(sen)
    elif df1[t]=='sports':
        sen=df2[t]
        s=s+clean(sen)
    elif df1[t]=='natural':
        sen=df2[t]
        n=n+clean(sen)
    elif df1[t]=='crime':
        sen=df2[t]
        c=c+clean(sen)
        
        
#print(p)
#print(s)
#print(n)
#print(c)

normal_vocab=[]
def word_feats(words):
    return dict([(words, True)]) 

for a in p:
    if a in p and a in s and a in n and a in c:
        normal_vocab=normal_vocab+list(a);
        p.remove(a)
        s.remove(a)
        n.remove(a)
        c.remove(a)
    if a in p and a in s:
        normal_vocab=normal_vocab+list(a);
        p.remove(a)
        s.remove(a)
    if a in s and a in n:
        normal_vocab=normal_vocab+list(a);
        n.remove(a)
        s.remove(a)
    if a in n and a in c:
        normal_vocab=normal_vocab+list(a);
        n.remove(a)
        c.remove(a)
    if a in p and a in c:
        normal_vocab=normal_vocab+list(a);
        p.remove(a)
        c.remove(a)
    if a in s and a in c:
        normal_vocab=normal_vocab+list(a);
        c.remove(a)
        s.remove(a)
    if a in p and a in n:
        normal_vocab=normal_vocab+list(a);
        p.remove(a)
        n.remove(a)
    if a in p and a in s and a in n:
        normal_vocab=normal_vocab+list(a);
        p.remove(a)
        s.remove(a)
        n.remove(a)
    if a in s and a in n and a in c:
        normal_vocab=normal_vocab+list(a);
        c.remove(a)
        s.remove(a)
        n.remove(a)
    if a in p and a in n and a in c:
        normal_vocab=normal_vocab+list(a);
        p.remove(a)
        c.remove(a)
        n.remove(a)
    if a in p and a in s and a in c:
        normal_vocab=normal_vocab+list(a);
        p.remove(a)
        s.remove(a)
        c.remove(a)
        
politics_vocab=p
#print(politics_vocab)
sports_vocab=s
natural_vocab=n
crime_vocab=c
 
politics_features = [(word_feats(pol), 'pol') for pol in politics_vocab]
sports_features = [(word_feats(spo), 'spo') for spo in sports_vocab]
natural_features = [(word_feats(nat), 'nat') for nat in natural_vocab]
crime_features = [(word_feats(cri), 'cri') for cri in crime_vocab]
normal_features = [(word_feats(nor), 'nor') for nor in normal_vocab]
train_set = politics_features + sports_features + natural_features+crime_features
#print(train_set)
classifier = SklearnClassifier(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, criterion="entropy"), sparse=False).train(train_set) 
cols = ['text']
dt = pd.read_csv("./test.csv",header=None, names=cols)
dtm = read_csv("./testmatch2.csv")
dtm1 = dtm['tag']
dtm2 = dtm['content']
#print(classifier)





# Predict
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))


def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()


testing = dt.text[:]
stopWords = set(stopwords.words('english'))
ps = PorterStemmer()
li=[]
test_result = []
wordsFiltered = []

c=0
n=0
po=0
sp=0
cr=0
na=0
poli=0
spor=0
natu=0
crim=0
o=0

for t in testing:
    pol=0
    cri=0
    nat=0
    spo=0
    pol2=0
    cri2=0
    nat2=0
    spo2=0
    #print(t)
    
    #print(wd)
    b=tweet_cleaner(t)
    #print(b)
    wd = word_tokenize(b)
    test_result.append(b)
    words = word_tokenize(tweet_cleaner(t))
    #print(words)
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    #print(wordsFiltered)
    for word in wordsFiltered:
        l=ps.stem(word)
        li.append(l)
    #print(li)
    sentence = li
    #print(sentence)
    #sentence = sentence.lower()
    #words = sentence.split(' ')
    w=sentence
    wordsFiltered=[]
    for word in w:
        classResult = classifier.classify( word_feats(word))
        if classResult == 'pol':
            pol = pol + 1
        if classResult == 'spo':
            spo = spo + 1
        if classResult == 'nat':
            nat = nat + 1
        if classResult == 'cri':
            cri = cri + 1
    #print(neg)
    #print(pos)
    #print(li)
    #print(len(words))

    pol2=float(pol)/(len(wd))
    spo2=float(spo)/(len(wd))
    nat2=float(nat)/(len(wd))
    cri2=float(cri)/(len(wd))

    #print('Politics: ' + str(pol2))
    #print('Sports: ' + str(spo2))
    #print('Natural: ' + str(nat2))
    #print('Crime: ' + str(cri2))
    if pol2>spo2 and pol2>nat2 and pol2>cri2:
        po=po+1
        if dtm1[o]=='politics':
            poli=poli+1
            #print('politics')
    elif spo2>pol2 and spo2>nat2 and spo2>cri2:
        sp=sp+1
        if dtm1[o]=='sports':
            spor=spor+1
            #print('sports')
    elif nat2>spo2 and nat2>pol2 and nat2>cri2:
        na=na+1
        if dtm1[o]=='natural':
            natu=natu+1
            #print('natural')
    elif cri2>spo2 and cri2>nat2 and cri2>pol2:
        cr=cr+1
        if dtm1[o]=='crime':
            crim=crim+1
            #print('crime')
    
    #print(li)
    while len(li)>0:
        li.pop()
    
    c=c+float(pol2+spo2+nat2+cri2)
    n=n+1
    o=o+1



politics_count=0
sports_count=0
natural_count=0
crime_count=0
for i in range(0, len(testing)):
    if dtm1[i]=='politics':
        politics_count=politics_count+1
    elif dtm1[i]=='sports':
        sports_count=sports_count+1
    elif dtm1[i]=='natural':
        natural_count=natural_count+1
    elif dtm1[i]=='crime':
        crime_count=crime_count+1
print('Politics Count: ' + str(politics_count))
print('Sports Count: ' + str(sports_count))
print('Natural Count: ' + str(natural_count))
print('Crime Count: ' + str(crime_count))
print()



ac=poli+spor+crim+natu
print('Politics Predicted: ' + str(po))
print('Sports Predicted: ' + str(sp))
print('Natural Predicted: ' + str(na))
print('Crime Predicted: ' + str(cr))

print()
print('Politics Correct: ' + str(poli))
print('Sports Correct: ' + str(spor))
print('Natural Correct: ' + str(natu))
print('Crime Correct: ' + str(crim))
print()

print('Test Data: ' + str(n))
print('Accuracy: ' + str(ac/n))
end = time.time()
print('Run Time: ' +str(end - start))# -*- coding: utf-8 -*-

