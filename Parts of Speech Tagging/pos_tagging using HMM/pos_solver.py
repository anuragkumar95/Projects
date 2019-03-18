import numpy as np


class Solver:
    # Calculate the log of the posterior probability of a given sentence
    def __init__(self):
        self.transition_dict = {}
        self.tag_to_word_dict = {}
        self.tag_dict={}
        self.word_dict={}
        self.initial_prob_tag_dict = {}
        self.transition_2_dict = {}
        
    def posterior(self, model, sentences, labels):
        if model == "Simple":   
            log_prob = 0
            for word , tag in zip(sentences, labels):
                #P(W=word|S=tag)
                if word not in self.tag_to_word_dict[tag].keys():
                    prob_w_t = 0.0000001
                else:
                    prob_w_t = self.tag_to_word_dict[tag][word]\
                           /self.tag_to_word_dict[tag]['word_count_with_tag']
                #P(S=tag|W=word) = P(W=word|S=tag)P(S=tag)/P(all words)
                #the denominator is constssant and thus only numeration is significant.
                prob_t_w = prob_w_t * (self.tag_dict[tag]/sum(self.tag_dict.values()))
                log_prob += np.log(prob_t_w)
            return log_prob
#
        elif model == "Complex":
            log_prob = 0
            for i in range(len(sentences)):
                #P(W=word|S=tag)
                prob_w_t = self.tag_to_word_dict.get(labels[i]).get(sentences[i],0.0000001)
                if prob_w_t != 0.0000001:
                    prob_w_t = prob_w_t/self.tag_to_word_dict.get(labels[i]).get('word_count_with_tag')
                #P(S=tag|W=word) = P(W=word|S=tag)P(S=tag)/P(all words)
                #the denominator is constant and thus only numeration is significant.
                prob_t_w = prob_w_t * (self.tag_dict[labels[i]]/sum(self.tag_dict.values()))
                if i == 0:
                    prob_s = self.tag_dict[labels[i]]/sum(self.tag_dict.values())
                    log_prob += np.log(prob_s)
                elif i == 1:
                    prob_s_s1 = self.transition_dict.get(labels[i-1]).get(labels[i],0.0000001)
                    if prob_s_s1 != 0.0000001:
                        prob_s_s1 = prob_s_s1/self.transition_dict.get(labels[i-1]).get('transition_count')
                    log_prob += np.log(prob_s_s1)
                else:
                    prob_s_s2 = self.transition_2_dict.get(labels[i-2]).get(labels[i-1],0.0000001)
                    if prob_s_s2 != 0.0000001:
                        prob_s_s2 =  self.transition_2_dict.get(labels[i-2]).get(labels[i-1]).get(labels[i],0.0000001)
                        if prob_s_s2 != 0.0000001:
                            prob_s_s2 = prob_s_s2/self.transition_2_dict.get(labels[i-2]).get(labels[i-1]).get('transition_count')
                    log_prob += np.log(prob_s_s2)
                log_prob += np.log(prob_w_t)
            return log_prob
           
        elif model == "HMM":
            log_prob = 0
            for i in range(len(sentences)):
                if i == 0:
                    prob_s0 = self.initial_prob_tag_dict.get(labels[i])\
                              /sum(self.initial_prob_tag_dict.values())
                    log_prob = np.log(prob_s0)
                #P(W=word|S=tag)
                prob_w_t = self.tag_to_word_dict.get(labels[i]).get(sentences[i],0.0000001)
                if prob_w_t != 0.0000001:
                    prob_w_t = prob_w_t/self.tag_to_word_dict.get(labels[i]).get('word_count_with_tag')
                #P(S=tag|W=word) = P(W=word|S=tag)P(S=tag)/P(all words)
                #the denominator is constant and thus only numeration is significant.
                prob_t_w = prob_w_t * (self.tag_dict[labels[i]]/sum(self.tag_dict.values()))
                prob_s_s1 = 1
                if i>=1:
                    if self.transition_dict[labels[i-1]].get(labels[i]) != None:
                        prob_s_s1 = self.transition_dict[labels[i-1]][labels[i]]\
                                    /self.transition_dict[labels[i-1]]['transition_count']
                    else:
                        prob_s_s1 = 0.0000001
                log_prob += np.log(prob_s_s1)+np.log(prob_w_t)
            return log_prob
            
        else:
            print("Unknown algo!")

    # Do the training!
    #%%
    def train(self, data):
        pos_tags = [x[1] for x in data]
        words = [x[0] for x in data]
        #creating transition dictionary to find P(Si+1|Si)
        for sentence in pos_tags:
            for i in range(len(sentence)-1):
                if sentence[i] not in self.transition_dict.keys():
                    self.transition_dict[sentence[i]] = {'transition_count':0}
                if sentence[i+1] not in self.transition_dict[sentence[i]].keys():
                   self.transition_dict[sentence[i]][sentence[i+1]] = 1
                else:
                   self.transition_dict[sentence[i]][sentence[i+1]] += 1
                self.transition_dict[sentence[i]]['transition_count'] += 1
        
        #creating a two level transition dictionary for P(Si+1|Si,Si-1)
        for sentence in pos_tags:
            for i in range (len(sentence)-2):
                if sentence[i] not in self.transition_2_dict.keys():
                    self.transition_2_dict[sentence[i]] = {}
                if sentence[i+1] not in self.transition_2_dict[sentence[i]].keys():
                    self.transition_2_dict[sentence[i]][sentence[i+1]] = {'transition_count':0}
                if sentence[i+2] not in\
                   self.transition_2_dict[sentence[i]][sentence[i+1]].keys():
                    self.transition_2_dict[sentence[i]][sentence[i+1]][sentence[i+2]] = 1
                else:
                    self.transition_2_dict[sentence[i]][sentence[i+1]][sentence[i+2]] += 1
                self.transition_2_dict[sentence[i]][sentence[i+1]]['transition_count'] += 1    
        
        #finding tags that start a sentence
        initial_tags = [i[0] for i in pos_tags]
        
        #creating dictionary for initial tags for P(S0=Si)
        for tag in initial_tags:
            if tag not in self.initial_prob_tag_dict.keys():
                self.initial_prob_tag_dict[tag] = 1
            else:
                self.initial_prob_tag_dict[tag] += 1
        
        #creating tags dictionary for P(S=Si)
        for sentence in pos_tags:
            for tag in sentence:
                if tag not in self.tag_dict.keys():
                    self.tag_dict[tag] = 1
                else:
                    self.tag_dict[tag] += 1
        
        #creating words dictionary for P(W=Wi)
        for sentence in words:
            for word in sentence:
                if word not in self.word_dict.keys():
                    self.word_dict[word] = 1
                else:
                    self.word_dict[word] += 1      
        
        #creating flattened list of tags and words           
        tags = [tag for sentence in pos_tags for tag in sentence]
        words = [word for sentence in words for word in sentence]
        
        word_data = []
        for i, j in zip(words,tags):
            word_data.append((i,j))
        
        tags = set(tags)
        
        #creating dictionary that stores tag wise list of words for P(W|S)
        for i in tags:
            temp_word_list = [word[0] for word in word_data if word[1] == i]
            if i not in self.tag_to_word_dict.keys():
                self.tag_to_word_dict[i] = {'word_count_with_tag':0}
            for word in temp_word_list:
                if word not in self.tag_to_word_dict[i].keys():
                    self.tag_to_word_dict[i][word] = 1
                else:
                    self.tag_to_word_dict[i][word] += 1
                self.tag_to_word_dict[i]['word_count_with_tag'] += 1
        
#%%
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        
        tag_prob = {}
        total_tags = sum(self.tag_dict.values())
        for key in self.tag_dict:
            tag_prob[key] = self.tag_dict[key]/total_tags
        
        word_prob = {}
        total_words = sum(self.word_dict.values())
        for key in self.word_dict:
            word_prob[key] = self.word_dict[key]/total_words
        
        max_prob_list = []
        for w in sentence:
            max_prob = -99999
            max_s = ''
            if w not in self.word_dict.keys():
                max_s = 'noun'
                max_prob = 1
            else:
                for s in self.tag_dict:
                    if w not in self.tag_to_word_dict[s].keys():
                        prob_s_w = 0
                    else:    
                        prob_s_w = (self.tag_to_word_dict[s][w]/self.tag_to_word_dict[s]['word_count_with_tag'])\
                                    *(tag_prob[s]/word_prob[w])
                        if(max_prob <= prob_s_w):
                            max_prob = prob_s_w
                            max_s = s        
            max_prob_list.append(max_s)
        return max_prob_list

    def hmm_viterbi(self, sentence):
        viterbi_list = []
        
        for it in range(len(sentence)):
            tag_prob_dict = {'.':{},'adj':{},'adp':{},'adv':{},'conj':{},'det':{},'noun':{},'num':{},\
                    'pron':{},'prt':{},'verb':{},'x':{}}
            for curr_tag in tag_prob_dict:
                # calculating P(W=sentence[it]|S=tag)
                if sentence[it] not in self.tag_to_word_dict[curr_tag].keys():
                    prob_w_s = 0.000000001
                else:
                    prob_w_s = self.tag_to_word_dict[curr_tag][sentence[it]]\
                                /self.tag_to_word_dict[curr_tag]['word_count_with_tag']
                
                #if a word in testing dataset is not present in training dataset, it is
                #more likely to be a noun. So I assign a probability of 1 for noun 
                #and 0 for all other tags.
                if sentence[it] not in self.word_dict.keys():
                    #if the new noun is in between the sentence, then use the previous entry
                    #in viterbi list to keep track of the prev_tag. Else its the first word
                    #in the sentence so no need to keep track of prev_tag.
                    if len(viterbi_list)>0:
                        prev_tag = max(viterbi_list[-1],key=lambda x:viterbi_list[-1][x]['prob'])
                        tag_prob_dict = {'.':{'prob':0},\
                                         'adj':{'prob':0},\
                                         'adp':{'prob':0},\
                                         'adv':{'prob':0},\
                                         'conj':{'prob':0},\
                                         'det':{'prob':0},\
                                         'noun':{'prob':1,'prev_tag':prev_tag},\
                                         'num':{'prob':0},\
                                         'pron':{'prob':0},\
                                         'prt':{'prob':0},\
                                         'verb':{'prob':0},\
                                         'x':{'prob':0}}
                    else:
                        tag_prob_dict = {'.':{'prob':0},\
                                         'adj':{'prob':0},\
                                         'adp':{'prob':0},\
                                         'adv':{'prob':0},\
                                         'conj':{'prob':0},\
                                         'det':{'prob':0},\
                                         'noun':{'prob':1},\
                                         'num':{'prob':0},\
                                         'pron':{'prob':0},\
                                         'prt':{'prob':0},\
                                         'verb':{'prob':0},\
                                         'x':{'prob':0}}
                    break
                
                #emmission prob = P(W=sentence[it]|S=tag)
                emm_prob = prob_w_s
                #check for start, if start then trans_prob = initial_probability,P(S0=tag)
                #else trans_prob = P(S=curr_tag|S=prev_tag)
                if it == 0:    
                   if curr_tag not in self.initial_prob_tag_dict.keys():
                       init_prob = 0
                   else:
                       init_prob = self.initial_prob_tag_dict[curr_tag]\
                                   /sum(self.initial_prob_tag_dict.values())
                   tag_prob_dict[curr_tag]['prob'] = emm_prob * init_prob
                   tag_prob_dict[curr_tag]['prev_tag'] = ''
                else:
                    max_prob_list = []
                    for prev_tag in viterbi_list[-1].keys():
                        if curr_tag not in self.transition_dict[prev_tag].keys():
                            trans_prob = 0
                        else:
                            trans_prob = self.transition_dict[prev_tag][curr_tag]\
                                    /self.transition_dict[prev_tag]['transition_count']
                        temp_prob = (viterbi_list[-1][prev_tag]['prob']*trans_prob,prev_tag)
                        max_prob_list.append(temp_prob)
                    likely_tag = max(max_prob_list, key=lambda x:x[0])
                    tag_prob_dict[curr_tag]['prob'] = emm_prob*likely_tag[0]
                    tag_prob_dict[curr_tag]['prev_tag'] = likely_tag[1]
            viterbi_list.append(tag_prob_dict)
    
        #Find the tag with the max probability at start for backtracking
        most_prob_seq = []
        probable_tag = max(viterbi_list[-1],key=lambda x:viterbi_list[-1][x]['prob'])
        most_prob_seq.append(probable_tag)
        viterbi_list = viterbi_list[::-1]
        #Start backtracking..
        for i in viterbi_list[:-1]:
            tag_in_seq = i[probable_tag]['prev_tag']
            most_prob_seq.append(tag_in_seq)
            probable_tag = i[probable_tag]['prev_tag']
        return most_prob_seq[::-1]

    
    def calc_prob_distribution(self ,sample_tags ,sentence ,tag_position):
        tags = ['.','adj','adp','adv','conj','det','noun','num','pron','prt','verb','x']
        prob_dist = []
        #if word not in traing data, then its most likely a noun
        if sentence[tag_position] not in self.word_dict.keys():
            return [0,0,0,0,0,0,1,0,0,0,0,0]
        if len(sentence)==1:
            if sentence[0] not in self.word_dict.keys():
                return [0,0,0,0,0,0,1,0,0,0,0,0]
            else:
                for tag in tags:
                    if sentence[0] not in self.tag_to_word_dict[tag].keys():
                       prob_w_s = 0.0000001
                    else:
                       prob_w_s = self.tag_to_word_dict[tag][sentence[0]]\
                             /self.tag_to_word_dict[tag]['word_count_with_tag']
                    prob_s = self.tag_dict[tag]/sum(self.tag_dict.values())
                    prob_dist.append(prob_w_s*prob_s)
            sum_prob = sum(prob_dist)
            prob_dist = [x/sum_prob for x in prob_dist]
            return prob_dist
                
        for i in range (len(sentence)):
            prod = 1
            if i == len(sentence)-1 and tag_position == i and i > 0:
                 for tag in tags:
                    if sentence[i] not in self.tag_to_word_dict[tag].keys():
                       prob_w_s = 0.0000001
                    else:
                       prob_w_s = self.tag_to_word_dict[tag][sentence[i]]\
                             /self.tag_to_word_dict[tag]['word_count_with_tag']
                    prob_s_s2_prev = self.transition_2_dict[sample_tags[i-2]].get(sample_tags[i-1]).get(tag)
                    if prob_s_s2_prev == None:
                        prob_s_s2_prev = 0.0000001
                    else:
                        prob_s_s2_prev = prob_s_s2_prev/self.transition_2_dict[sample_tags[i-2]].get(sample_tags[i-1]).get('transition_count')
                    prob_dist.append(prob_w_s * prob_s_s2_prev)
            
            elif i == 0 and tag_position == 0:
                for tag in tags:
                    if sentence[i] not in self.tag_to_word_dict[tag].keys():
                        prob_w_s = 0.0000001
                    else:
                        prob_w_s = self.tag_to_word_dict[tag][sentence[tag_position]]\
                              /self.tag_to_word_dict[tag]['word_count_with_tag']
                    prob_s = self.tag_dict[tag]/sum(self.tag_dict.values()) 
                    if sample_tags[i+1] not in self.transition_dict[tag].keys():
                       prob_s_s1 = 0.0000001
                    else:
                       prob_s_s1 = self.transition_dict[tag][sample_tags[1]]\
                              /self.transition_dict[tag]['transition_count']
                    prob_dist.append(prob_w_s*prob_s*prob_s_s1)
            
            elif i == 0 and tag_position != 0:
                if sentence[i] not in self.tag_to_word_dict[sample_tags[i]].keys():
                    prob_w_s = 0.0000001
                else:
                    prob_w_s = self.tag_to_word_dict[sample_tags[i]][sentence[i]]\
                              /self.tag_to_word_dict[sample_tags[i]]['word_count_with_tag']
                prob_s = self.tag_dict[sample_tags[i]]/sum(self.tag_dict.values())
                prod = prod * prob_w_s * prob_s
            elif i == 1 and tag_position == 1:
                for tag in tags:
                    if sentence[i] not in self.tag_to_word_dict[tag].keys():
                       prob_w_s = 0.0000001
                    else:
                       prob_w_s = self.tag_to_word_dict[tag][sentence[i]]\
                             /self.tag_to_word_dict[tag]['word_count_with_tag']
                    if tag not in self.transition_dict[sample_tags[i-1]].keys():
                       prob_s_s1 = 0.0000001
                    else:
                       prob_s_s1 = self.transition_dict[sample_tags[i-1]][tag]\
                              /self.transition_dict[sample_tags[i-1]]['transition_count']
                    prob_s_s2 = self.transition_2_dict[sample_tags[0]].get(tag)
                    if prob_s_s2 == None:
                        prob_s_s2 = 0.0000001
                    else:
                        prob_s_s2 = self.transition_2_dict[sample_tags[0]].get(tag).get(sample_tags[2])
                        if prob_s_s2 == None:
                            prob_s_s2 = 0.0000001
                        else:
                            prob_s_s2 = prob_s_s2/self.transition_2_dict[sample_tags[0]].get(tag).get('transition_count')
                    prob_dist.append(prob_w_s*prob_s_s1*prob_s_s2)
            elif i == 1 and tag_position != i:
                if sentence[i] not in self.tag_to_word_dict[sample_tags[i]].keys():
                    prob_w_s = 0.0000001
                else:
                    prob_w_s = self.tag_to_word_dict[sample_tags[i]][sentence[i]]\
                         /self.tag_to_word_dict[sample_tags[i]]['word_count_with_tag']
                if tag_position > i:
                    if sample_tags[i] not in self.transition_dict[sample_tags[i-1]].keys():
                       prob_s_s1 = 0.0000001
                    else:
                       prob_s_s1 = self.transition_dict[sample_tags[i-1]][sample_tags[i]]\
                              /self.transition_dict[sample_tags[i-1]]['transition_count']
                    prod = prod * prob_w_s * prob_s_s1
                elif tag_position < i:
                    prod = prod * prob_w_s    
           
            elif i > 1 and tag_position == i:
                for tag in tags:
                    if sentence[i] not in self.tag_to_word_dict[tag].keys():
                        prob_w_s = 0.0000001
                    else:
                        prob_w_s = self.tag_to_word_dict[tag][sentence[i]]\
                         /self.tag_to_word_dict[tag]['word_count_with_tag']
                    prob_s_s2_prev = self.transition_2_dict[sample_tags[i-2]].get(sample_tags[i-1]).get(tag)
                    if prob_s_s2_prev == None:
                        prob_s_s2_prev = 0.0000001
                    else:
                        prob_s_s2_prev = prob_s_s2_prev/self.transition_2_dict[sample_tags[i-2]].get(sample_tags[i-1]).get('transition_count')
                    prob_s_s2_next = self.transition_2_dict[sample_tags[i-1]].get(tag)
                    if prob_s_s2_next == None:
                        prob_s_s2_next = 0.0000001
                    else:
                        prob_s_s2_next = self.transition_2_dict[sample_tags[i-1]].get(tag).get(sample_tags[i+1])
                        if prob_s_s2_next == None:
                            prob_s_s2_next = 0.0000001
                        else:
                            prob_s_s2_next = prob_s_s2_next/self.transition_2_dict[sample_tags[i-1]].get(tag).get('transition_count')
                    prob_dist.append(prob_w_s * prob_s_s2_prev * prob_s_s2_next)
            elif i > 1 and tag_position != i:
                if sentence[i] not in self.tag_to_word_dict[sample_tags[i]].keys():
                    prob_w_s = 0.0000001
                else:
                    prob_w_s = self.tag_to_word_dict[sample_tags[i]][sentence[i]]\
                               /self.tag_to_word_dict[sample_tags[i]]['word_count_with_tag']
                if tag_position > i or tag_position < i-1:
                    prob_s_s2_prev = self.transition_2_dict[sample_tags[i-2]].get(sample_tags[i-1])
                    if prob_s_s2_prev == None:
                        prob_s_s2_prev = 0.0000001
                    else:
                        prob_s_s2_prev = self.transition_2_dict[sample_tags[i-2]].get(sample_tags[i-1]).get(sample_tags[i])
                        if prob_s_s2_prev == None:
                            prob_s_s2_prev = 0.0000001
                        else:
                            prob_s_s2_prev = prob_s_s2_prev/self.transition_2_dict[sample_tags[i-2]].get(sample_tags[i-1]).get('transition_count')
                    prod = prod * prob_w_s * prob_s_s2_prev   
                elif tag_position == i-1:
                    prod = prod * prob_w_s
        prob_dist = [prod*x for x in prob_dist]
        sum_prob = sum(prob_dist)
        prob_dist = [x/sum_prob for x in prob_dist]
        return prob_dist
                    
    def complex_mcmc(self, sentence):
        tags = ['.','adj','adp','adv','conj','det','noun','num','pron','prt','verb','x']
        iteration = 2500
        warm_period =500
        particles = []
        self.sentence = sentence
        sample = np.random.choice(tags,len(sentence))
        while(iteration>0):
            for i in range (0,len(sentence)):
                self.sample = sample
                prob_dist = self.calc_prob_distribution(sample,sentence,i)
                self.prob_dist = prob_dist
                sample[i] = np.random.choice(tags,1,p=prob_dist)[0]
                if iteration>= warm_period:
                    particles.append(sample)
                iteration -= 1           
        sen_length = len(sentence)
        word_tag = {i:{} for i in range (0,sen_length)}
        for sample in particles:
            for i in range(0,sen_length):
                if sample[i] not in word_tag[i].keys():
                    word_tag[i][sample[i]] = 1
                else:
                    word_tag[i][sample[i]] += 1
        
        pred_tags = [max(word_tag[i],key=lambda x:word_tag[i][x]) for i in range(sen_length)]
        return pred_tags

 
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

