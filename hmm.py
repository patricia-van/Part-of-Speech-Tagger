## Members:
# 1.	Duan Yaoyi – A0237010Y
# 2.	Michelle Angela Celeste – A0244123U
# 3.	Patricia Vanessa Santoso – A0240278E 
# 4.	Yeo Xue Ling Andrea – A0239291Y

COUNT_TAGS = {}
COUNT_TOKEN = {}

## COUNT_TAGS stores tags and the number of time each tag appears in the training data
## COUNT_TOKEN stores tokens and the number of time each one appears in the training data
train = open('twitter_train.txt', encoding="utf-8")
for l in train.readlines():
    stripped = l.strip()
    if len(stripped) != 0:
        token, tag = (stripped.split('\t'))
        COUNT_TAGS[tag] = COUNT_TAGS.get(tag, 0) + 1
        COUNT_TOKEN[token] = COUNT_TOKEN.get(token, 0) + 1

## NUM_WORDS is the number of unique words in the training data
NUM_WORDS = len(COUNT_TOKEN)
DELTA = 1
UNSEEN_PROB = {}

## for unseen tokens, output_prob = constant / count(y=j) + constant * (num_words + 1)
## UNSEEN_PROB stores the tags and the probability of each tags with unseen token
for tag, count in COUNT_TAGS.items():
    prob = DELTA / (count + (DELTA * (NUM_WORDS + 1)))
    UNSEEN_PROB[tag] = prob
## get the tag with the max probability from the existing ones in UNSEEN_PROB
UNSEEN_TAG = max(UNSEEN_PROB, key=UNSEEN_PROB.get)


### Q2.(a) ###
## helper function to count P(x=w|y=j) ##
def output_probabilities():
    fin = open('twitter_train.txt', encoding="utf-8")
    count = {}

    ## store each token, tag, and number of time the token appears with the tag into a dictionary     
    for l in fin.readlines():
        stripped = l.strip()
        if len(stripped) != 0:
            token, tag = (stripped.split('\t'))
            token = token.lower()

            # get current dict of token for the specific tag 
            curr_tag_tokens = count.get(tag, {})
            # update the number of time the token appears with the tag 
            curr_tag_tokens[token] = curr_tag_tokens.get(token, 0) + 1
            count[tag] = curr_tag_tokens

    naive_output_probs = {}   
    ans = ""     
    
    ## calculate P(x=w|y=j)
    for tag, token_dict in count.items():
        probs = {}
        denom = sum(token_dict.values()) + 1 * (NUM_WORDS + 1)
        for token, count in token_dict.items():
            num = token_dict.get(token) + 1
            probs[token] = num/denom
            ans += tag + "\t" + token + "\t" + str(num/denom) + "\n"
            naive_output_probs[tag] = probs

    ## write the tag, token, and associated prob to the output txt file
    # with open("naive_output_probs.txt", 'w') as f:
    with open("naive_output_probs.txt", 'w', encoding="utf-8") as f:
        f.write(ans)


### Q2.(b) ###
## calculate count(y=j) using COUNT_TAGS dictionary ##
## count num_words using COUNT_TOKEN dictionary ##

# Implement the six functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    ## read all the files to extract the data
    test = open(in_test_filename, encoding="utf-8")
    train_output = open(in_output_probs_filename, encoding="utf-8")
    train_prob = {}
    
    for l in train_output.readlines():
        tag, token, prob = (l.strip().split('\t'))
        token = token.lower()
        prob = float(prob)
        train_prob_token = train_prob.get(token, {})
        train_prob_token[tag] = prob
        train_prob[token] = train_prob_token
    ## for each word, select the tag j that maximises P(x=w|y=j) 
    final_output = ""
    for lines in test.readlines():
        t = lines
        if t == "\n":
            final_output += "\n"
        else:
            t = t.strip().lower()
            prob_tags = train_prob.get(t, 0)
            ## handle cases for unseen tokens
           
            if t[0:5] == "@user":
                max_tag = "@"
            elif prob_tags == 0:
                max_tag = UNSEEN_TAG
            else:
                max_tag = max(prob_tags, key = prob_tags.get)
            final_output += max_tag + "\n"
        
    # write the predictions into the txt
    with open(out_prediction_filename, 'w', encoding="utf-8") as f:
        f.write(final_output)

## Q2c ##
## Naive prediction accuracy: 1005/1378 = 0.7293178519593614

## Q3a ##
## To calculate arg max P(y=j|x=w), we can derive the prob from P(x=w|y=j) by applying Bayes Rules
## By Bayes Rules, P(y=j|x=w) = P(x=w, y=j) / P(x=w) = P(x=w|y=j) * P(y=j) / P(x=w)
## As the denominator is a normalizing constant that doesn't depend on the numerator which we want to make inference from, 
## we can assume P(x=w) is constant and exclude it from the calculation 
## Hence, we can derive that P(y=j|x=w) is proportional to P(x=w, y=j) * P(y=j)
## From which, we obtain the equation P(x=w,y=j) = P(x=w|y=j) * P(y=j) to calculate the probability.

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    test = open(in_test_filename, encoding="utf-8")
    train_output = open(in_output_probs_filename, encoding="utf-8")
    train_prob = {}

    for l in train_output.readlines():
            tag, token, prob = (l.strip().split('\t'))
            token = token.lower()
            prob = float(prob)
            train_prob_token = train_prob.get(token, {})
            train_prob_token[tag] = prob
            train_prob[token] = train_prob_token

    final_output = ""
    for lines in test.readlines():
        t = lines
        if t == "\n":
            final_output += "\n"
        else:
            t = t.strip().lower()
            prob_tags = train_prob.get(t, 0)
            if t[0:5] == "@user":
                max_tag = "@"
            elif prob_tags == 0 :
                max_tag = UNSEEN_TAG
            else:
                updated_prob_tags = {}
                
                #calculate P(x=w,y=j) = P(x=w|y=j) * P(y=j) for each tag
                for tag, prob in prob_tags.items():
                    updated_prob_tags[tag] = prob * COUNT_TAGS[tag]

                ## select j that maximises P(x=w,y=j) then write to txt 
                max_tag = max(updated_prob_tags, key = updated_prob_tags.get)
            final_output += max_tag + "\n"
        
    with open(out_prediction_filename, 'w', encoding="utf-8") as f:
        f.write(final_output)

## Q3c ##
## Naive prediction accuracy: 1014/1378 = 0.7358490566037735

## Q4a ##
## helper function to count P(x=w|y=j) ## (the same function as in Q2a with 1 delta value)
def output_probabilities():
    fin = open('twitter_train.txt', encoding="utf-8")
    count = {}

    ## store each token, tag, and number of time the token appears with the tag into a dictionary     
    for l in fin.readlines():
        stripped = l.strip()
        if len(stripped) != 0:
            token, tag = (stripped.split('\t'))
            token = token.lower()

            # get current dict of token for the specific tag 
            curr_tag_tokens = count.get(tag, {})
            # update the number of time the token appears with the tag 
            curr_tag_tokens[token] = curr_tag_tokens.get(token, 0) + 1
            count[tag] = curr_tag_tokens

    naive_output_probs = {}   
    ans = ""     
    
    ## calculate P(x=w|y=j)
    for tag, token_dict in count.items():
        probs = {}
        denom = sum(token_dict.values()) + 1 * (NUM_WORDS + 1)
        for token, count in token_dict.items():
            num = token_dict.get(token) + 1
            probs[token] = num/denom
            ans += tag + "\t" + token + "\t" + str(num/denom) + "\n"
            naive_output_probs[tag] = probs

    ## write the tag, token, and associated prob to the output txt file
    with open("output_probs.txt", 'w', encoding="utf-8") as f:
        f.write(ans)

## helper function to count P(yt=j|yt-1=i) ## the transition probability
def trans_probabilities():

    fin = open('twitter_train.txt', encoding="utf-8")
    count = {}
    previous_tag = "START"
        ## store each token, tag, and number of time the token appears with the tag into a dictionary     
    for l in fin.readlines():
        stripped = l.strip()
        if (stripped != ""):

            token, tag = (stripped.split('\t'))

        if (stripped == ""):
                # tag = "END"
                curr_tag_transition = count.get(previous_tag, {})
                # update the number of time the token appears with the tag 
                curr_tag_transition["END"] = curr_tag_transition.get("END", 0) + 1
                count[previous_tag] = curr_tag_transition
                previous_tag = "START"

        else: 
            curr_tag_transition = count.get(previous_tag, {})
            # update the number of time the token appears with the tag 
            curr_tag_transition[tag] = curr_tag_transition.get(tag, 0) + 1
            count[previous_tag] = curr_tag_transition
            previous_tag = tag

    TAGS = []
    fin = open('twitter_tags.txt', encoding="utf-8")
    for l in fin.readlines():
        tag = l.strip()
        TAGS.append(tag)

    ## check for unseen transition ##
    for taga in TAGS:
        for tagb in TAGS:
            firstd = count.get(taga)
            # print(firstd)
            firstd[tagb] = firstd.get(tagb,0)
            count[taga] = firstd
            # print(secd)
    ## account for all prob from START -> ##
    for tag in TAGS:
            firstd = count.get("START")
            # print(firstd)
            firstd[tag] = firstd.get(tag,0)
            count["START"] = firstd
    ## account for all prob from _ -> END ##
    for tag in TAGS:
            firstd = count.get(tag)
            # print(firstd)
            firstd['END'] = firstd.get('END',0)
            count[tag] = firstd

    trans_probs = {}   
    ans = ""     

    tag_dict = list(count.keys())
    tag_dict.append("END")
    ## calculate transition probability
    for tag in count.keys():
        
            probs = {}
            denom = sum(count[tag].values()) + 1 * (27 + 1)
            for token in tag_dict:
                if token == "START" or (tag == "START" and token == "END") or (tag == "END"):
                    ans += tag + "\t" + token + "\t" + str(0) + "\n"
                else:
                    num = count[tag].get(token) + 1
                    probs[token] = num/denom
                    ans += tag + "\t" + token + "\t" + str(num/denom) + "\n"
                    trans_probs[tag] = probs
    for token in tag_dict:
        ans += "END" + "\t" + token + "\t" + str(0) + "\n"
            # write the tag, token, and associated prob to the output txt file
    with open("trans_probs.txt", 'w', encoding="utf-8") as f:
        f.write(ans)


## Q4b ##
## read all the different tags and store in a list
    ## include additional 'START' and 'END' tags to indicate start and stop states
    ## here we store it as the global variable as it will be used in the subsequent two questions
tags = ['START'] 
for l in open("twitter_tags.txt", encoding="utf-8").readlines():
    stripped = l.strip()
    if len(stripped) != 0:
        tags.append(stripped)
tags.append('END')

from collections import defaultdict
## the viterbi_predict function
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    ## read test file 
    test = open(in_test_filename, encoding="utf-8")
    tags = ['START'] 
    for l in open(in_tags_filename, encoding="utf-8").readlines():
        stripped = l.strip()
        if len(stripped) != 0:
            tags.append(stripped)
    tags.append('END')

    ## read transition probabilities for tags and store in a dictionary 
    trans_prob = {}
    for l in open(in_trans_probs_filename, encoding="utf-8").readlines():
        prev_tag, next_tag, prob = (l.strip().split('\t'))
        prob = float(prob)
        trans_prob_prev_tag = trans_prob.get(prev_tag, {})
        trans_prob_prev_tag[next_tag] = prob
        trans_prob[prev_tag] = trans_prob_prev_tag


    ## read output probabilities for tokens and store in a dictionary 
    output_prob = {}
    for l in open(in_output_probs_filename, encoding="utf-8").readlines():
        tag, token, prob = (l.strip().split('\t'))
        token = token.lower()
        prob = float(prob)
        train_prob_token = output_prob.get(token, {})
        train_prob_token[tag] = prob
        output_prob[token] = train_prob_token

    final_output = ""

    words = []
    for l in test.readlines():
        stripped = l.strip().lower()

        if len(stripped) != 0:
            words.append(stripped)
        
        #initialize Pi and Back Pointers using defaultdict
        elif stripped == "":
            Pi = defaultdict(lambda: defaultdict(float))
            BP = defaultdict(dict)
            token = words[0]

            #Base case
            for tag in tags:
                P_trans = trans_prob.get('START').get(tag,0)
                P_output = output_prob.get(token, UNSEEN_PROB).get(tag, 0)
                Pi[0][tag] =  float(P_trans * P_output)
                BP[0][tag] = "START"

            #store Pi(i, tag) and BP(i, tag) from each iterative step
            for i in range(1, len(words)):
                for tag in tags:
                    max_prob = float('-inf')
                    max_tag = ''
                    for prev_tag in tags:

                        prob = Pi[i - 1][prev_tag] * trans_prob.get(prev_tag).get(tag) * output_prob.get(words[i], UNSEEN_PROB).get(tag, UNSEEN_PROB.get(tag, 0))
                        if prob > max_prob:
                            max_prob = prob
                            max_tag = prev_tag
                    Pi[i][tag] = max_prob
                    BP[i][tag] = max_tag

            #Last step of the iteration, get the max probability         
            BP_tags = []
            max_prob = float('-inf')
            max_tag = ''
            for tag in tags:
                prob = trans_prob.get(tag).get("END", 0) * Pi[len(words)-1][tag]
                if prob > max_prob:
                            max_prob = prob
                            max_tag = tag
            BP_tags.append(max_tag)

            #get the tag sequence of finalBP (BP_tags)
            for i in range(len(words)-1, 0, -1):
                BP_tags.insert(0, BP[i][BP_tags[0]])
            
            for tag in BP_tags :
                final_output += tag + "\n"
            final_output += "\n"
            words = []
    
    with open(out_predictions_filename, 'w', encoding="utf-8") as f:
        f.write(final_output)

## Q4c ##
## Viterbi prediction accuracy:   971/1378 = 0.704644412191582

## Q5a ##
## To improve on the Viterbi algorithm in Question 4, we propose to do preprocessing of unseen words to better handle them
## From observation, we identified the pattern that user name starts with @user, and it contributes to a lot of unseen words as user names are different
## another pattern is url starts with "http://" or "https://"
## we also identified the pattern of some word starting with "#" representing hashtags in the tweets
## therefore, we want to handle unseen words with these pattern using the preprocess_word(word) function below

## In addition to the 3 patterns identified,
## we also run a grammatical analysis of the word to help better the unseen word prediction according to the grammatical pattern
## this is done by the function statecheck(word)
## In this function, we check the expression to identify whether the word belongs to any of these categories:
## punctuations, number, nouns,verbs and adjectives
## this is important as the linguistic patterns will affect POS
## they are stored as a global variable

## The output probability and transition probability functions are not affected

def preprocess_word (word):
    # from observation, user name starts with @user
    if word.startswith("@user"):
        return "@user"
    
    # from observation, url starts with http:// or https://
    if word.startswith("http://") or word.startswith("https://"):
        return "http://"
    
    if word.startswith("#"):
        return "#"

    else:
        # run a grammatical analysis of the word to help better the unseen word prediction according to the grammatical pattern
        return statecheck(word)

import re
RARE_SYMBOL = '_RARE_'
def statecheck(word):
    # check if they end with these suffix--then return the state accordingly
    if not re.search(r'\w', word):
        return '_PUNCS_'
    elif re.search(r'[A-Z]', word):
        return '_CAPITAL_'
    elif re.search(r'\d', word):
        return '_NUM_'
    elif re.search(r'(ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)',word):
        return '_NOUNLIKE_'
    elif re.search(r'(ate\b|fy\b|ize\b|\ben|\bem)', word):
        return '_VERBLIKE_'
    elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)',word):
        return '_ADJLIKE_'
    else:
        return RARE_SYMBOL

fin = open('twitter_train.txt', encoding="utf-8")
suffix_count = {}

## store each token, tag, and number of time the token appears with the tag into a dictionary     
for l in fin.readlines():
    stripped = l.strip()
    if len(stripped) != 0:
        token, tag = (stripped.split('\t'))
        token = token.lower()
        
        suffix = preprocess_word(token)
        # get current dict of tag for the specific suffix 
        curr_suffix_tags = suffix_count.get(suffix, {})
        # update the number of time the token appears with the tag 
        curr_suffix_tags[tag] = curr_suffix_tags.get(tag, 0) + 1
        suffix_count[suffix] = curr_suffix_tags

tags = ['START'] 
for l in open("twitter_tags.txt", encoding="utf-8").readlines():
    stripped = l.strip()
    if len(stripped) != 0:
        tags.append(stripped)
tags.append('END')

suffix_output_probs = {}     

for suffix in suffix_count.keys():
    probs = {}
    for tag in tags:
        # smoothing
        denom = sum(suffix_count[suffix].values()) + 1 * (NUM_WORDS + 1)
        num = suffix_count[suffix].get(tag, 0) + 1
        # denom = sum(suffix_count[suffix].values())
        # num = suffix_count[suffix].get(tag, 0)
        probs[tag] = num/denom
        suffix_output_probs[suffix]=  probs

## getting the probability of the suffix to the tag with the highest probability
suffix_final_probs = {} 
for suffix, tags in suffix_output_probs.items():
    max_prob = float('-inf')
    max_tag =""
    for tag, prob in tags.items():
        if prob > max_prob:
            max_prob = prob
            max_tag = tag

    suffix_final_probs[suffix]= {max_tag : max_prob} 


## Q5b ##

## The output probability and transition probability functions are not affected
## Thus the respective files are duplicated from Question 4
## The global variable suffix_final_probs is obtained from part a above

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    ## read test file 
    test = open(in_test_filename, encoding="utf-8")

    ## read transition probabilities for tags and store in a dictionary 
    trans_prob = {}
    for l in open(in_trans_probs_filename, encoding="utf-8").readlines():
        prev_tag, next_tag, prob = (l.strip().split('\t'))
        prob = float(prob)
        trans_prob_prev_tag = trans_prob.get(prev_tag, {})
        trans_prob_prev_tag[next_tag] = prob
        trans_prob[prev_tag] = trans_prob_prev_tag

    ## read output probabilities for tokens and store in a dictionary 
    output_prob = {}
    for l in open(in_output_probs_filename, encoding="utf-8").readlines():
        tag, token, prob = (l.strip().split('\t'))
        token = token.lower()
        prob = float(prob)
        train_prob_token = output_prob.get(token, {})
        train_prob_token[tag] = prob
        output_prob[token] = train_prob_token
    
    final_output = ""

    words = []
    for l in test.readlines():
        stripped = l.strip().lower()

        if len(stripped) != 0:
            words.append(stripped)

        elif stripped == "":
            Pi = defaultdict(lambda: defaultdict(float))
            BP = defaultdict(dict)
            token = words[0]


            for tag in tags:
                P_trans = trans_prob.get('START').get(tag,0)
                # check if the token is not unseen words and has no pattern to identify
                if token in output_prob.keys() or preprocess_word(token) != "OTHERS":
                    P_output = output_prob.get(token, UNSEEN_PROB).get(tag, 0)
                else :
                    suffix = preprocess_word(token)
                    P_output = suffix_final_probs[suffix].get(tag, 0)
                Pi[0][tag] =  float(P_trans * P_output)
                BP[0][tag] = "START"

            for i in range(1, len(words)):
                for tag in tags:
                    max_prob = float('-inf')
                    max_tag = ''
                    for prev_tag in tags:
                        if words[i] in output_prob.keys() or preprocess_word(words[i]) != "OTHERS":
                            P_output = output_prob.get(words[i], UNSEEN_PROB).get(tag, 0)
                        else :
                            suffix = preprocess_word(words[i])
                            P_output = suffix_final_probs[suffix].get(tag, 0)

                        prob = Pi[i - 1][prev_tag] * trans_prob.get(prev_tag).get(tag) * P_output
                        if prob > max_prob:
                            max_prob = prob
                            max_tag = prev_tag
                    Pi[i][tag] = max_prob
                    BP[i][tag] = max_tag

            BP_tags = []
            max_prob = float('-inf')
            max_tag = ''
            for tag in tags:
                prob = trans_prob.get(tag).get("END", 0) * Pi[len(words)-1][tag]
                if prob > max_prob:
                            max_prob = prob
                            max_tag = tag
            BP_tags.append(max_tag)

            for i in range(len(words)-1, 0, -1):
                BP_tags.insert(0, BP[i][BP_tags[0]])
            
            for tag in BP_tags :
                final_output += tag + "\n"
            final_output += "\n"
            words = []
    
    with open(out_predictions_filename, 'w', encoding="utf-8") as f:
        f.write(final_output)


## Q5c ##
## Viterbi2 prediction accuracy:  1081/1378 = 0.7844702467343977



def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename, encoding="utf-8") as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename, encoding="utf-8") as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)


def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '/Users/patricia/Desktop/Y2S2/BT3102/BT3102ProjectFinal'

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    
    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                     viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    

if __name__ == '__main__':
   output_probabilities()
   trans_probabilities()
   run()

