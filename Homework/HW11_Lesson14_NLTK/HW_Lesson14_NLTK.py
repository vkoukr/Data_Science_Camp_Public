import nltk
from nltk.corpus import gutenberg 
moby_raw = gutenberg.raw('melville-moby_dick.txt') 

#------- Tokenizing---
def example_one():
    from nltk.tokenize import word_tokenize
    return len(word_tokenize(moby_raw)) 

print ('{:,}'.format(example_one()))

#------- Unique---
def example_two():    
    return len(set(nltk.word_tokenize(moby_raw)))
print ('{:,}'.format(example_two()))

#---Lemitizing
from nltk.stem import WordNetLemmatizer

def example_three():
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in nltk.word_tokenize(moby_raw)]
    return len(set(lemmatized))

print ('{:,}'.format(example_three()))

#---- Question 1---
def answer_one():
    return example_two()/example_one()

print(answer_one())

#---- Question 2-----
from nltk.tokenize import word_tokenize
def answer_two():    
    #from nltk.tokenize import word_tokenize
    all_tokens=word_tokenize(moby_raw)
    indices = [i for i, x in enumerate(all_tokens) if ((x == "whale") | (x=='Whale'))]   # -- Number of tokens 'whale' or "Whale"
    return len(indices)/len(all_tokens)*100

print(answer_two())

#---- Question 3-----
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# This function should return a list of 10 tuples where each tuple is of the form (token, frequency). The list should be sorted in descending order of frequency.

def answer_three():
    from nltk import FreqDist
    all_tokens=nltk.word_tokenize(moby_raw)
    dist = FreqDist(all_tokens) # the same as text1.vocab() 
    #print(dist)
    #dist
    return [(x,dist[x]) for x in list(dist)[:20]]

print(answer_three())

#---- Question 4
# What tokens have a length of greater than 5 and frequency of more than 150?
# This function should return a sorted list of the tokens that match the above constraints. To sort your list, use sorted()

def answer_four():
    from nltk import FreqDist
    all_tokens=nltk.word_tokenize(moby_raw)
    dist = FreqDist(all_tokens) # the same as text1.vocab() 
    vocab=dist.keys()
    freq_tokens = [w for w in vocab if len(w) > 5 and dist[w] > 150] 
    return sorted(freq_tokens)

print (answer_four())

# ------Question 5
# Find the longest word in text1 and that word's length.
# This function should return a tuple (longest_word, length).
def answer_five():
    from nltk import FreqDist
    all_tokens=nltk.word_tokenize(moby_raw)
    dist = FreqDist(all_tokens) # the same as text1.vocab() 
    vocab=dist.keys()
    t=[[x,len(x)] for x in vocab]
    index = max(t, key=lambda item: item[1])
    return (index[0], index[1])

print(answer_five())

# ------Question 6
# What unique words have a frequency of more than 2000? What is their frequency?
# This function should return a list of tuples of the form (frequency, word) sorted in descending order of frequency.
def answer_six():
    from nltk import FreqDist
    import pandas as pd
    
    all_tokens=nltk.word_tokenize(moby_raw)
    dist = FreqDist(all_tokens) 
    vocab=dist.keys()
    uniq_tokens=set(nltk.word_tokenize(moby_raw))
    freq_tokens = [(w, dist[w]) for w in uniq_tokens if ((dist[w] > 2000) & (w.isalpha()))] 
    df=pd.DataFrame(freq_tokens,columns=["token", "frequency"])
    df=df.sort_values(by='frequency', ascending=False)
    return list(zip(df.frequency, df.token))

print(answer_six())

# ------Question 7
# What is the average number of tokens per sentence?
# This function should return a float.
from nltk.tokenize import sent_tokenize,word_tokenize
import numpy as np 

def answer_seven():
    sentences = sent_tokenize(moby_raw)
    counts = (len(nltk.word_tokenize(sentence)) for sentence in sentences)
    return sum(counts)/float(len(sentences))

print(answer_seven())

# -----Question 8
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# This function should return a list of tuples of the form (part_of_speech, frequency) sorted in descending order of frequency.
def answer_eight():
    from nltk import FreqDist
    import pandas as pd
    
    all_tokens=nltk.word_tokenize(moby_raw)
    dist = FreqDist(all_tokens)
    df=pd.DataFrame(dist.most_common(), columns=["token", "frequency"])
    tagged = nltk.pos_tag(df.token)
    frequencies = FreqDist([tag for (word, tag) in tagged])
    return frequencies.most_common(5)

print(answer_eight())

# ---- Question 9
# Create spelling recommender, that take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest `edit distance` (you may need  to use `nltk.edit_distance(word_1, word_2, transpositions=True)`), and starts with the same letter as the misspelled word, and return that word as a recommendation.

# Recommender should provide recommendations for the three words: `['cormulent', 'incendenece', 'validrate']`.
# <br>*This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*
def answer_nine(default_words= ['cormulent', 'incendenece', 'validrate']):    
    import pandas as pd
    from nltk.corpus import words
    from nltk.metrics.distance import (
        edit_distance,
        jaccard_distance,
        )
    from nltk.util import ngrams
    correct_spellings = words.words()
    spellings_series = pd.Series(correct_spellings)
    outcomes = []
    gram_number=3
    for entry in default_words:
        spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        distances = ((jaccard_distance(set(ngrams(entry, gram_number)),
                                       set(ngrams(word, gram_number))), word)
                     for word in spellings)
        closest = min(distances)
        outcomes.append(closest[1])
    return outcomes

print(answer_nine())
