import pandas as pd
import re
from bs4 import BeautifulSoup
import cleantext                    
import contractions

def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = set(line.strip().lower() for line in file)
    return stopwords

def remove_stopwords(text, stopwords):
    filtered = [word for word in text if word not in stopwords]
    return filtered

def remove_html(text):
    return BeautifulSoup(text, "lxml").get_text()

def preprocess_text(text):

    # remove HTML tags
    text = remove_html(text)

    # remove URLs
    # The \S metacharacter matches non-whitespace characters.
    # https://www.w3schools.com/jsref/jsref_obj_regexp.asp 
    text = re.sub(r'http\S+|https\S+|www\S+', '', text)

    # remove emails using cleantext 
    # https://www.tutorialspoint.com/clean-web-scraping-data-using-clean-text-in-python
    text = cleantext.clean(text, no_emails=True)

    # remove hashtags
    # The \w metacharacter matches word characters.
    # https://www.w3schools.com/jsref/jsref_regexp_wordchar.asp
    text = re.sub(r'#\w+', '', text)

    # remove mentions
    text = re.sub(r'@\w+', '', text)

     # remove emoji
    text = cleantext.clean(text, no_emoji=True)

    # lowercase 
    text = text.lower()

    # handle contractions
    # https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/?ref=header_outind
    text = contractions.fix(text)

    # remove special characters and numbers 
    # https://www.w3schools.com/jsref/jsref_regexp_whitespace.asp 
    text = re.sub(r'[^a-z\s]', '', text)

    # remove whitespace
    text = cleantext.clean(text, normalize_whitespace=True)

    # LEMMATIZATION - transform words into their basic form

    # tokenize
    tokens = text.split() # Tokenize using split()

     # remove stopwords
    text = remove_stopwords(tokens, stopwords)

    return text

# global variable for loading stopwords once
stopwords = load_stopwords('stopwords.txt')

# read data-set
dataFrame = pd.read_csv('evaluation.csv')

#print(dataFrame)

# print 10 first enties
#print(dataFrame.head(10))

# using preprocessing
dataFrame['text'] = dataFrame['text'].apply(preprocess_text)

dataFrame.to_csv('processed_evaluation_data.csv', index=False)

# print 10 first enties
#print(dataFrame.head(10))

print(dataFrame)


