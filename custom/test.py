from nltk.tokenize import sent_tokenize, word_tokenize

# Provided text
text = "I waited for the train. The train was late. Dr. Jiao and Seung Jun took the bus. I looked for Seung Jun and Dr. Jiao at the bus station."

# Tokenizing sentences and then words
result = [word_tokenize(t) for t in sent_tokenize(text)]


# Number of sublists
len(result)
