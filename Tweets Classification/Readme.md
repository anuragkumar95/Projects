The file geolocate.py is the final version for this submission. The file has been uploaded from a windows system, thus please do the following before running on silo.

dos2unix geolocate.py
chmod u+x geolocate.py
Following preprocessing has been done on the tweets before passing through the classifier:-

Removed capitalization.
Removed all punctuations except {'#','@'}.
Removed some trivial frequently occuring words like {and,is,the,on,at,for},etc.
Testing accuracy : 68.2% Training accuracy : 96.6% Training time : 3.2 sec

To make the classification easy, upon reading the training data, I made a dictionary which has each city as a key. Each key has a seperate nested dictionary as its value. The inner dictionary contains all the different words I encountered while reading a tweet from this city. For ex :- Tweet 1 : Los_angeles,_CA,A B C D Tweet 2 : Los_angeles,_CA,A B D E Tweet 3 : San_Francisco,_CA,A F B D Tweet 4 : Manhattan,_NY,D R T F

Then my dictionary would have the below structure : 
data = { 'Los_Angeles,_CA':{'A':2,'B':2,'C':1,'D':2,'E':1,'word_freq':8,'freq':2},
         'San_Francisco,_CA':{'A':1,'B':1,'D':1,'F':1,'word_freq':4,'freq':1},
         'Manhattan,_NY':{'D':1,'F':1,'R':1,'T':1,'word_freq':4,'freq':1} 
       }

The key 'word_freq' contains #of all words in tweets from a given city. The key 'freq' contains #of all tweets from a given city.

The function cond_prob(X,a,b) takes in data(dictionary) and gives out P(b|a)/P(a) The function Naive_Bayes(tweets,data) accepts the testing data(tweets) and training data(dictionary "data") and returns predicted label

The output is stored in the file "Output.txt" in the format {predicted city, actual city, tweet}.
