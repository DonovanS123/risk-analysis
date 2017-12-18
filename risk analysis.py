# Donovan Schroeder

######################################################################



import csv
import matplotlib.pyplot as plt
import numpy as np
from random import randint

# Fields from the input file, in same order as they appear in the file
# (ignoring respondent ID). Do not modify or change.
fields=['result','smoke','drink','gamble','skydive','speed','cheat','steak','cook','gender','age','income','education','location']
# Corresponding field values from the input file. Do not modify or change.
values=[('lottery a','lottery b'),('no','yes'),('no','yes'),('no','yes'),('no','yes'),('no','yes'),('no','yes'),('no','yes'),
        ('rare','medium rare','medium','medium well','well'),('male','female'),
        ('18-29','30-44','45-60','> 60'),
        ('$0 - $24,999','$25,000 - $49,999','$50,000 - $99,999','$100,000 - $149,999','$150,000+'),
        ('less than high school degree','high school degree','some college or associate degree','bachelor degree','graduate degree'),
        ('east north central','east south central','middle atlantic','mountain','new england','pacific',
         'south atlantic','west north central','west south central')]

######################################################################
# returns a dictionary of dictionaries
# outer dictionary has keys of responce # and value equal to a dictionary of their responces
# response dictionary has keys equal to fields and values equal to their choice
def readData(filename='steak-risk-survey.csv', fields=fields, values=values):
    dictList = []
    # copied to retain field to maintain field indexes
    fieldZ=['result','smoke','drink','gamble','skydive','speed','cheat','steak','cook','gender','age','income','education','location']
    def helper(row):
        helpDict={ fieldZ[i]:row[i+1].lower() for i in range(len(row)-1) if row[i+1] != '' } # create initial dictionary using field indexes from fieldZ
        return( { fields[i]:helpDict[fields[i]] for i in range(len(fields)) if fields[i] in helpDict } ) # limits dictionary to fields requested
    
    try:
        with open(filename, newline='') as csvfile: # open file
            fopen = csv.reader(csvfile, delimiter=',') # read file
            for row in fopen:
                if 'Lottery A' in row or 'Lottery B' in row: # only keep lines with a lottery choice
                    dictList.append(helper(row))
            return(dictList)
    except:
        print("File couldn't be opened: ", filename)
        exit()    

######################################################################
# prints a graph with bars displaying ratios of values per field
# and seperated by lottery preference
def showPlot(D, field, values):
    '''print a graph of ratios of value choice seperated by lottery choice
D is the result of readData
field is category of response
values options they can pick'''
    lottos = ('lottery a', 'lottery b') # stored lotto options
    # creates dictionary of counters e.g. {lottery a:{value1:0, value2:0}, lottery b:{value1:0, value2:0}}
    countDict = {lotto:{ value:0 for value in values } for lotto in lottos}
    for i in range(len(D)):
        if field in D[i]:
            countDict[D[i]['result']][D[i][field]] += 1 # add to count at {lottery:{value:count}}
    # data to be plot
    n_groups = len(values) # number of responses
    totalResp = sum([ sum([countDict[lotto][value] for value in values]) for lotto in lottos]) # totals results by adding count from each dictionary
    resultLotto = [[countDict[lotto][value]/totalResp for value in values] for lotto in lottos] # convert counters to proportions
    # plot settings
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1

    # bar settings
    rectsA = plt.bar(index, resultLotto[0], bar_width, alpha=opacity, color='b', label='Lottery A')
    rectsB = plt.bar(index + bar_width, resultLotto[1], bar_width, alpha=opacity, color='r', label='Lottery B')
    # labels
    plt.xlabel('Value')
    plt.ylabel('Percentage of population')
    plt.title('Lottery preference by '+"''"+field+"''")
    plt.xticks(index + (.5*bar_width), (values))
    plt.legend()

    plt.tight_layout()
    plt.show()
    
######################################################################
# returns a dictionary indexed by fields
# each value within that is another dictionary indexed by lottery choice
# within that is another dictionary indexed by their field response
# with the inner most value being the ratio
# of people within that field that chose that result
def train(D, fields=fields, values=values):
    '''returns a dictionary in the form {field1:{lotterya:{value1: %, value2: %}, lotteryb:{value1: %, value2: %}}, field2:{lotterya:{value1: %, value2: %}, lotteryb:{value1: %, value2: %}}}
including a special case {'result':{lotterya:%,lotteryb:%}}
D should be a list of dictionaries such as from readData
%s are the proportion of people in that category that chose that value'''
    lottos = ('lottery a', 'lottery b') # stored lotto options
    
    # creates dictionary of dictionaries of dictionaries of counters e.g. {field:{lottery a:{value1:0, value2:0}, lottery b:{value1:0, value2:0}, etc.}
    megaDict = { fields[i]:{ lotto:{ value:0 for value in values[i] } for lotto in lottos } for i in range(len(fields)) if fields[i] != 'result' } # initializes dictionary except for D['result']
    
    # adds keys with values of dictionaries of counters for the lottery choice e.g. {result:{lottery a:0, lottery b:0}}
    megaDict['result'] = {lotto:0 for lotto in lottos} # done seperately because of exceptions
    
    for i in range(len(D)): # iterate through respondants
        for field in fields:
            if field in megaDict and field in D[i] and field != 'result':
                megaDict[field][D[i]['result']][D[i][field]] += 1 # add to counter in field:lottery:value
            elif field == 'result':
                megaDict[field][D[i][field]] += 1 # add to counter in field:lottery
    for i in range(len(fields)):
        if fields[i] != 'result':
            for lotto in lottos:
                totalVal = sum( [ megaDict[fields[i]][lotto][v] for v in values[i]]) # total count responses by field:lottery
                for value in values[i]:
                    megaDict[fields[i]][lotto][value] = megaDict[fields[i]][lotto][value] / totalVal # replaces the stored counts to proportions
    # finally change D[result][lottery] counts to proportions
    megaDict['result']['lottery a'],megaDict['result']['lottery b'] = megaDict['result']['lottery a']/len(D),megaDict['result']['lottery b']/len(D)
    return(megaDict)

######################################################################
# return our prediction of which lottery someone with example answers would pick
# example is a respondant it is a dictionary of fields:values
# P is the proportions we get back from our train function
def predict(example, P, fields=fields, values=values):
    '''return a prediction of which lottery the respondant with example answers would pick
example is a dictionary of respondant information e.g. an element from readData
P is the result of train'''
    lottos = ('lottery a', 'lottery b')
    # dictionary of lotteries with value equal to the average chance of lottery a or b
    pDict = {lotto:P['result'][lotto] for lotto in lottos}
    for i in range(len(fields)):
        if fields[i] != 'result' and fields[i] in example:
            for lotto in lottos:
                # multiply by ratio by proportion for the value chosen by respondant
                pDict[lotto] = pDict[lotto]*P[fields[i]][lotto][example[fields[i]]]
    if pDict[lottos[0]] > pDict[lottos[1]]: # return whichever has higher chance
        return('lottery a')
    else:
        return('lottery b')
                        
            

######################################################################
# Predict by guessing. You're going to be about half right!
def guess(example, fields=fields, values=values):
    return(values[0][randint(0,1)]==example['result'])

######################################################################
# returns the ratio that our predict function got correct out of all tested
def test(D, P, fields=fields, values=values):
    '''returns the % our train/predict functions got correct vs the results
D is list of respondants dictionaries
P is result of train'''
    correct = 0
    for i in range(len(D)):
        if predict(D[i], P, fields, values) == D[i]['result']: # if predict(respondant) was correct
            correct += 1
    return(correct / len(D))
            

######################################################################
# Fisher-Yates-Knuth fair shuffle, modified to only shuffle the last k
# elements. S[-k:] will then be the test set and S[:-k] will be the
# training set.
def shuffle(D, k):
    # Work backwards, randomly selecting an element from the head of
    # the list and swapping it into the tail location. We don't care
    # about the ordering of the training examples (first len(D)-N),
    # just the random selection of the test examples (last N).
    i = len(D)-1
    while i >= len(D)-k:
        j = randint(0, i)
        D[i], D[j] = D[j], D[i]
        i = i-1
    return(D)

# Evaluate.
def evaluate(filename='steak-risk-survey.csv', fields=fields, values=values, trials=100):
    # Read in the data.
    D = readData(filename, fields, values)
    # Establish size of test set (10% of total examples available).
    N = len(D)//10
    result = 0
    random = 0
    for i in range(trials):
        # Shuffle to randomly select N test examples.
        D = shuffle(D, N)
        # Train the system on first 90% of the examples.
        P = train(D[:-N], fields=fields, values=values)
        # Test on last 10% of examples, chosen at random by shuffle().
        result += test(D[-N:], P, fields=fields, values=values)
        # How well would you do guessing at random?
        random += sum([ len([ True for x in D[-N:] if guess(x)])/N ])
    # Return average accuracy.
    print('NaiveBayes={}, random guessing={}'.format(result/trials, random/trials))


