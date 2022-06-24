import enum
import re
from nltk import word_tokenize, sent_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from urllib import request
from bs4 import BeautifulSoup
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

'''
# Get Data, Read/Write Files
'''
def scrape_web(url:str):
    with request.urlopen(url) as f:
        html = f.read().decode('utf8')
        raw = BeautifulSoup(html, 'html.parser').get_text()
        tokens = word_tokenize(raw)
        sents = sent_tokenize(raw)
    return tokens, sents

def read_file(file:str):
    with open(file, 'r') as f:
        data = f.read()

    return data

def write_file(file:str, doc:str):
    with open(file, 'w') as f:
        f.write(doc)

def read_pandas_csv(file:str):
    df = pd.read_csv(filepath=file, sep=',')

    return df
'''
# Data Pre-processing & Feature Engineering: text normalization, tokenization, bag of words, TFIDF
'''
def handle_negation(text:str):
    pattern = re.compile(r"([a-zA-Z]*)(n\'t|n't|\s[Nn]ever|\s[Nn]ot|\s[Nn]or|\s[Cc]annot)((\s[a-zA-Z0-9']+)+)(\sand|,|.|!|;|:|\?|-)")
    matches = pattern.finditer(text)
    for match in matches:
        subtext = match.group(3)
        pattern2 = re.compile(r"\s")
        subtext2 = pattern2.sub(r" NOT_", subtext)
        pattern3 = re.compile(subtext)
        text = pattern3.sub(subtext2,text)

    # print(text)

    return text

def tokenize(text:str):
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
    tokens = tokenizer.tokenize(text)

    # from nltk import word_tokenize
    # tokens = word_tokenize(text)
    return tokens

def text_normalize(tokens:list):
    wnl = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    stop_words += string.punctuation
    tokens = [wnl.lemmatize(token.lower()) for token in tokens if token not in stop_words]
    return tokens

def sent_normalize(sents:list):
    wnl = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    stop_words += string.punctuation
    sents_normalized = list()
    for sent in sents:
        tokens = tokenize(sent)
        sent_cleaned = ' '.join([wnl.lemmatize(token.lower()) for token in tokens if token not in stop_words])
        sents_normalized.append(sent_cleaned)
    # List of sentences str
    print(sents_normalized)
    return sents_normalized

def regex(text:str):
    pattern = re.compile(r'[a-zA-Z0-9]')
    matches = pattern.findall(text)
    print(matches)
    return matches

def regex_index(text:str):
    pattern = re.compile(r'\b\w+\b')
    matches = pattern.finditer(text)
    for match in matches:
        print(match)
        print(match.span())
    return 

def regex_sub(text:str):
    pattern = re.compile(r'[^a-zA-Z0-9]')
    matches = pattern.sub(' ', text)
    print(matches)
    # matches is a str
    return matches

def term_frequency(tokens:list):
    word_counts = Counter(tokens)
    for key,value in word_counts.items():
        word_counts[key] /= len(tokens)
    print(word_counts.most_common(10))
    return word_counts

def count_vectorizer(sents:list):
    '''
    sents: list of strings
    '''
    vectorizer = CountVectorizer()
    try:
        X = vectorizer.fit_transform(sents)
    except AttributeError:
        sents = [' '.join(sent) for sent in sents]
        X = vectorizer.fit_transform(sents)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    print(df.head(), df.shape)
    return df

def tfidf_vectorizer(sents:list, vectorizer=None, ngram_range=(1,1)):
    '''
    sents: list of strings
    '''
    if vectorizer:
        try:
            X = vectorizer.transform(sents)
        except AttributeError:
            sents = [' '.join(sent) for sent in sents]
            X = vectorizer.transform(sents)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        try:
            X = vectorizer.fit_transform(sents)
        except AttributeError:
            sents = [' '.join(sent) for sent in sents]
            X = vectorizer.fit_transform(sents)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    print('TFIDF:', df.head(), df.shape)
    return df, vectorizer

def pos_tags_count_vectorizer(docs_cleaned:list):
    '''
    docs_cleaned: list of list of tokens
    '''
    docs_pos_tags = [pos_tag(sent) for sent in docs_cleaned]

    docs_pos_tags_concat = list()
    for sent in docs_pos_tags:
        inner_list = list()
        for token in sent:
            # inner_list.append("{}_{}".format(token[0],token[1]))
            inner_list.append(token[1])
        docs_pos_tags_concat.append(inner_list)

    # Check
    print([len(sent) for sent in docs_pos_tags_concat] == [len(sent) for sent in docs_cleaned])
    # print(docs_cleaned[:5])
    # print(docs_pos_tags_concat[:5])

    df_pos = count_vectorizer(docs_pos_tags_concat)
    return df_pos

def entities_countvectorizer(docs:list):
    '''
    docs: list of strings
    '''
    ents = list()
    for sent in docs:
        doc_ent = nlp(sent)
        ent_inner = list()
        for ent in doc_ent.ents:
            ent_inner.append(ent.label_)
        ents.append(ent_inner)
    print('Ents Len:', len(ents))

    df_ents = count_vectorizer(ents)
    return df_ents

def other_features(docs:list, docs_pre_normalized_tokenized:list):
    '''
    docs: list of strings
    docs_pre_normalized_tokenized: list of list of tokens
    '''
    # Number numbers
    pattern = re.compile(r'\b\d+\b')
    num_numbers = [len(pattern.findall(text)) for text in docs]
    print(len(num_numbers))

    # Number of Words
    len_words = [len(sent) for sent in docs_pre_normalized_tokenized]
    print(len(len_words))

    # Avg Char Len of Each Word
    avg_char_len_word = [mean([len(word) for word in sent]) for sent in docs_pre_normalized_tokenized]
    print(len(avg_char_len_word))

    # Number Stopwords
    stop_words = stopwords.words('english')
    stop_words = [stopword.lower() for stopword in stop_words]
    num_stopwords = [sum([1 for word in sent if word.lower() in stop_words]) for sent in docs_pre_normalized_tokenized]
    print(len(num_stopwords))

    # Number Punctuation
    punc = string.punctuation
    num_punc = [sum([1 for word in sent if word in punc]) for sent in docs_pre_normalized_tokenized]
    print(len(num_punc))

    # Number Names
    names_ = names.words()
    names_ = [name.lower() for name in names_]
    num_names = [sum([1 for word in sent if word.lower() in names_]) for sent in docs_pre_normalized_tokenized]
    print(len(num_names))

    return num_numbers, len_words, avg_char_len_word, num_stopwords, num_punc, num_names

'''
# Modeling Pipeline
'''

'''
# LSA/SVD
'''

def lsa(doc):
    '''
    doc: list of strings
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(doc)

    from sklearn.decomposition import TruncatedSVD
    lsa = TruncatedSVD(n_components=2, n_iter=100)
    lsa.fit(X)
    terms = vectorizer.get_feature_names()

    for i,comp in enumerate(lsa.components_):
        termsInComp = zip(terms, comp)
        sortedterms = sorted(termsInComp, key=lambda x : x[1], reverse=True)[:10]
        print("Concept {}".format(i))
        for term in sortedterms:
            print(term[0])
        print(" ")
