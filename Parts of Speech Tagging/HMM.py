#!/usr/bin/env python3
'''
Author: Anurag Kumar
Created on: 10/19/2019 
'''

from nltk.corpus import brown
from collections import Counter


class HMM:
	def __init__(self):
		#tagged set of words and its POS in our vocab
		self.vocab = {}
		#sentences from brown corpus
		self.corpus = {}
		#Storing marginal probs of tags and words in the corpus
		self.tagCount = {}
		self.wordCount = {}
		#dictionaries for storing probabilities for HMM
		self.initial = {}
		self.emission = {}
		self.transition = {}
	
	def create_basicprobs(self):
		#storing marginals on tags		
		tags = [i[1] for i in self.vocab]
		for tag in tags:
			if tag in self.tagCount:
				self.tagCount[tag] += 1
			else:
				self.tagCount[tag] = 1
		sumCounts = sum(self.tagCount.values())
		self.tagCount = {key:self.tagCount[key]/sumCounts for key in self.tagCount}
		#storing marginals on words
		words = [i[0].lower() for i in self.vocab]
		for word in words:
			if word in self.wordCount:
				self.wordCount[word] += 1
			else:
				self.wordCount[word] = 1
		sumCounts = sum(self.wordCount.values())
		self.wordCount = {key:self.wordCount[key]/sumCounts for key in self.wordCount}


	def create_initial(self):
		#Storing words which start a sentence.
		sents_start = [i[0] for i in self.corpus]
		#Storing counts
		for _,tag in sents_start:
			if tag in self.initial:
				self.initial[tag] += 1
			else:
				self.initial[tag] = 1
		sumCounts = sum(self.initial.values())
		#Changing counts to probabilities
		self.initial = {key:self.initial[key]/sumCounts for key in self.initial}

	def create_transition(self):
		#Storing counts for each transition for each sentence.
		for sentence in self.corpus:
			tags = [x[1] for x in sentence]
			for i in range(len(sentence)-1):
				_curr = tags[i]
				_next = tags[i+1]  
				if _curr in self.transition:
					if _next in self.transition[_curr]:
						self.transition[_curr][_next] += 1
					else:
						self.transition[_curr][_next] = 1
				else:
					self.transition[_curr] = {_next:1}
		#All transitions in this corpus
		totalTrans = 0
		for key1 in self.transition:
			for key2 in self.transition[key1]:
				totalTrans += self.transition[key1][key2]

		#Converting transition counts to probabilities
		for currTag in self.transition:
			prob_currTag = self.tagCount[currTag]
			for nextTag in self.transition[currTag]:
				self.transition[currTag][nextTag] = \
						self.transition[currTag][nextTag]/(totalTrans * prob_currTag)

	def create_emission(self):
		#storing count of emmisions
		for word, tag in self.vocab:
			word = word.lower()
			if tag not in self.emission:
				self.emission[tag] = {word:1}
			else:
				if word not in self.emission[tag]:
					self.emission[tag][word] = 1
				else:
					self.emission[tag][word] += 1

		totalEmis = 0
		for key1 in self.emission:
			for key2 in self.emission[key1]:
				totalEmis += self.emission[key1][key2]
		#Changing emission count to probabilities
		for tag in self.emission:
			prob_tag = self.tagCount[tag] 
			for word in self.emission[tag]:
				self.emission[tag][word] = self.emission[tag][word]/(totalEmis * prob_tag)

	def trainHMM(self, vocab, corpus):
		self.vocab = vocab
		self.corpus = corpus
		self.create_basicprobs()
		self.create_initial()
		self.create_transition()
		self.create_emission()

	def predict(self, sentence, universal = False) -> list:
	#The idea is to store probabilities for each tag for each word and keep a track of prev tag. 
	#Now we start the forward phase and populate our viterbi matrix. 
		viterbi = []
		tagset = self.tagCount.keys()
		for i, word in enumerate(sentence):
			word = word.lower()
			tags = {x:{} for x in tagset}
			for currTag in tagset:
				#If a word in the sentence is not present in our training corpus, then I assume
				#it to be a noun, thus I assign a prob of 1 for tag 'NN' and rest 0. If this 
				#If this word starts the sentence then we don't need to keep a prev tag, else
				#we also store the prev predicted tag.
				if word not in self.wordCount.keys():
					if universal:
						if i == 0:
							tags = {x:({'prob':1, 'prev_tag': ''} \
								if x == 'NOUN' else {'prob':0, 'prev_tag': ''}) for x in tags.keys()}
						else:
							prevTag = max(viterbi[-1], key = lambda x: viterbi[-1][x]['prob'])
							tags = {x:({'prob':1, 'prev_tag': prevTag}\
									  if x == 'NOUN' else {'prob':0, 'prev_tag': ''}) for x in tags.keys()}
					else:
						if i == 0:
							tags = {x:({'prob':1, 'prev_tag': ''} \
								if x == 'NN' else {'prob':0, 'prev_tag': ''}) for x in tags.keys()}
						else:
							prevTag = max(viterbi[-1], key = lambda x: viterbi[-1][x]['prob'])
							tags = {x:({'prob':1, 'prev_tag': prevTag}\
									  if x == 'NN' else {'prob':0, 'prev_tag': ''}) for x in tags.keys()}
					break
				
				#Calculating emission probability. If such emission is not present in training
				#corpus, then I assign it a very small probability
				emm_prob = self.emission[currTag].get(word, 0.0000001)

				#Calculating transition probability. If this is the start of the sentence then
				#this is initial probability, otherwise a transition probability. If its a start
				#of the sentence and the tag doesnt start it, then assign a initial prob of 0. 
				if i == 0:
					init_prob = self.initial.get(currTag, 0)
					tags[currTag]['prob'] = init_prob * emm_prob
					tags[currTag]['prevTag'] = ''
				else:
					prob_list = []
					for prevTag in viterbi[-1].keys():
					#If prevTag never starts a transition then trans_prob is 0
						check = self.transition.get(prevTag, 0)
						if check != 0:
						#If a transition from prev to curr tag is not present in the training
						#corpus then trans_prob is a very small value
							trans_prob = self.transition[prevTag].get(currTag,0.0000001)
						else:
							trans_prob = 0
						#Store the current path's prob and a record of prevTag in a tuple and store
						#this path
						prob = (viterbi[-1][prevTag]['prob'] * trans_prob, prevTag)
						prob_list.append(prob)

					#the path having the maximum prob is the likely path
					likely_tag = max(prob_list, key = lambda x: x[0])
					#Store this path in the list for next viterbi iteration
					tags[currTag]['prob'] = emm_prob * likely_tag[0]
					tags[currTag]['prev_tag'] = likely_tag[1]
			
			viterbi.append(tags)
    
		#Now that we have populated the viterbi matrix, we start the backtracking to predict the 
		#correct sequence of tags for this sentence. The start of the sequence will be the tag 
		#having maximum probability in the last viterbi iteration.
		probable_tag = max(viterbi[-1], key = lambda x: viterbi[-1][x]['prob'])
		most_prob_seq = [probable_tag]
		#Reverse the matrix and start backtrack
		viterbi = viterbi[::-1]
		for i in viterbi[:-1]:
			tag_in_Seq = i[probable_tag]['prev_tag']
			most_prob_seq.append(tag_in_Seq)
			probable_tag = tag_in_Seq
		#The list of predicted tags is in reverse order so reverse it again to get the right tags.
		return most_prob_seq[::-1] 


if __name__ == '__main__':
	sentence1 = "I went to the market and bought cloths ."
	sentence2 = "time flies like an arrow ."
	sent1 = sentence1.split()
	sent2 = sentence2.split()
	#training HMM required a "vocab":list of pos tagged words and "corpus": list of pos tagged 
	#sentences. 
	vocab = brown.tagged_words()
	corpus = brown.tagged_sents()
	HMM = HMM()
	HMM.trainHMM(vocab, corpus)
	print("Sentence:", sentence1)
	predictedTags = HMM.predict(sent1)
	print("Predicted Tags:", predictedTags)
	print("--------------------------------------------------------")
	print("Sentence:",sentence2)
	predictedTags = HMM.predict(sent2)
	print("Predicted Tags:", predictedTags)
	print("--------------------------------------------------------")
	










					






