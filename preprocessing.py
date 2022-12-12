import os
import re
import random

def fix(a):
    if a.startswith('"') and a.endswith('"'):
        a = a[1:-1] #removes extraneous quotation marks
    if not a.endswith((".", "!", "?")):
        a = a + "." #adds periods to sentences without ending punctuation
    instances = re.compile(r"COVID-19", re.IGNORECASE)
    a = instances.sub("coronavirus", a) #changes instances of COVID-19 to coronavirus (see methods section)
    return a

def parse():
    with open('data.txt') as f:
        lines = f.readlines()
    random.shuffle(lines) #Shuffles data, which prevents the same set of statements from being in training and testing set
    data = [[line.split(',',2)[1],fix(line.split(',',2)[2][:-3])] for line in lines] #implements the fix() method, as outlined above
    train_data = data[:int(0.75*len(data))]
    print(len(train_data))
    test_data = [x for x in data if x not in train_data]
    print(len(test_data))
    vocabulary, vocab_size = {}, 2 #initializing COVID-19 and UNK tokens
    vocabulary['covid-19'] = 0
    vocabulary['coronavirus'] = 0 #additional check, covid-19 and coronavirus given the same token
    vocabulary['UNK'] = 1 #UNK token used for words which appear in the testing set but do not appear in the training set
    claims = []
    max = 0
    for x in range(0,len(test_data)):
        claim = test_data[x][1]
        tokens = claim.split()
        if len(tokens)>max:
            max = len(tokens) #gets max length sentence to pad other sentences to same length
    for x in range(0,len(train_data)):
        claim = train_data[x][1]
        claims.append(claim)
        tokens = claim.split()
        if len(tokens)>max:
            max = len(tokens)
        for token in tokens:
            token = token.lower().replace('"', '').replace("'", '').replace('�?','') #additional cleaning procedures
            if token not in vocabulary: #tokenizing input words for LSTM model
                vocabulary[token] = vocab_size
                vocab_size += 1
    tokenize = lambda x: [vocabulary[key.lower().replace('"', '').replace("'", '').replace('�?','')] if key in vocabulary else vocabulary['UNK'] for key in x.split()]
    encode = lambda x: [1,0,0] if x == "F" else ([0,1,0] if x == "U" else [0,0,1])
    lstm_encode = lambda x: 0 if x == "F" else (1 if x == "U" else 2)
    lstm_training = [[pad(tokenize(claims[x]),max,len(vocabulary)) for x in range(0,len(claims))],[lstm_encode(train_data[x][0]) for x in range(0,len(claims))]]
    test_claims = [test_data[x][1] for x in range(0,len(test_data))]
    lstm_testing = [[pad(tokenize(test_claims[x]),max,len(vocabulary)) for x in range(0,len(test_claims))],[lstm_encode(test_data[x][0]) for x in range(0,len(test_claims))]]
    fax_training = [claims,[encode(train_data[x][0]) for x in range(0,len(claims))]]
    fax_testing = [test_claims,[encode(test_data[x][0]) for x in range(0,len(test_claims))]]
    vocabulary['**PAD**'] = vocab_size
    vocabulary['**STOP**'] = vocab_size + 1
    vocab_size += 2
    return [fax_training,lstm_training],[fax_testing,lstm_testing], vocabulary

def pad(a,max,size): #pads all token sequences to be the same length using pad tokens followed by a final stop token
    for x in range(0,max-len(a)):
        a.append(size)
    a.append(size+1) 
    return a
