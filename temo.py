from nltk.stem import PorterStemmer

porter=PorterStemmer()
words=['running','jumps','liking','proudness']
str=[porter.stem(word) for word in words]
print(str)