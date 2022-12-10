import os

def parse():
    with open('data.txt') as f:
        lines = f.readlines()
    data = [line.split(',')[1:3] for line in lines]
    train_data = data[:int(0.75*len(data))]
    test_data = data[len(data)-len(train_data):]
    vocabulary, vocab_size = {}, 2 #initializing COVID-19 and UNK tokens
    vocabulary['covid-19'] = 0
    vocabulary['coronavirus'] = 0
    vocabulary['UNK'] = 1  
    preset = ['covid-19','coronavirus','UNK']
    claims = []
    frequency = {}
    UNK_list = [] 
    for x in range(0,len(train_data)):
        claim = train_data[x][1]
        claims.append(claim)
       # print('claim',claim)
        tokens = claim.split()
        #print('tokens')
        for token in tokens:
            token = token.lower().replace('"', '').replace("'", '').replace('�?','')
            if x == 34:
                print('token',token)
            if token not in vocabulary:
                vocabulary[token] = vocab_size
                vocab_size += 1
                frequency[token] = 1
                UNK_list.append(token)
            else:
                if token not in preset:
                    frequency[token] = frequency[token] + 1
                if token in UNK_list:
                    UNK_list.remove(token)
    #print('frequency',frequency)
    # for UNK in UNK_list:
    #     vocabulary[UNK] = 1
    tokenize = lambda x: [vocabulary[key.lower().replace('"', '').replace("'", '').replace('�?','')] if key in vocabulary else vocabulary['UNK'] for key in x.split()]
    #print('claimsZX',claims[0])
    #print('tokenize',tokenize(claims[0]))
    encode = lambda x: [1,0,0] if x is "F" else ([0,1,0] if x is "U" else [0,0,1])
    #print('train_data',train_data[0])
    lstm_training = [[tokenize(claims[x]) for x in range(0,len(claims))],[encode(train_data[x][0]) for x in range(0,len(claims))]]
    test_claims = [test_data[x][1] for x in range(0,len(test_data))]
    lstm_testing = [[tokenize(test_claims[x]) for x in range(0,len(test_claims))],[encode(test_data[x][0]) for x in range(0,len(test_claims))]]
    #test_data  = list(map(lambda x: vocabulary[x], test_data[:][1]))
    #print('train',train_data)
    #print('test',testing)
    fax_training = [claims,[encode(train_data[x][0]) for x in range(0,len(claims))]]
    fax_testing = [test_claims,[encode(test_data[x][0]) for x in range(0,len(test_claims))]]
    print('lstm_train',lstm_training)
    print('lstm_test',lstm_testing)
    return [fax_training,lstm_training],[fax_testing,lstm_testing], vocabulary