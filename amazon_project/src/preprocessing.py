# Inspired by the demos of the course

import re
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import enchant
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def remove_specials_characters(documents):
    """
    Remove special characters
    :param documents: the documents from where we remove the characters
    :return: the documents without the special characters
    """
    documents_no_specials = []
    for item in documents:
        try:
            documents_no_specials.append(
                item.replace('\r', ' ').replace('/n', ' ').replace('.', ' ').replace(',', ' ').replace('(', ' ') \
                    .replace(')', ' ').replace("'s", ' ').replace('"', ' ') \
                    .replace('!', ' ').replace('?', ' ').replace("'", '') \
                    .replace('>', ' ').replace('$', ' ') \
                    .replace('-', ' ').replace(';', ' ') \
                    .replace(':', ' ').replace('/', ' ').replace('#', ' '))
        except:
            documents_no_specials.append("")
    return documents_no_specials


def remove_numerical(documents):
    """
    remove the words containing numbers
    :param documents: 
    :return: 
    """

    def hasNumbers(inputString):
        return bool(re.search(r'\d', inputString))

    documents_no_stop_no_numeric = [[token for token in text if not (hasNumbers(token))]
                                    for text in documents]
    return documents_no_stop_no_numeric


def clean_text(documents, tester=0):
    """
    remove special characters, stop words and numbers, lemmatize and remove non-english words 
    :param documents: Docuements to clean
    :param tester: which element you would like to display as an example
    :return: the cleaned documents
    """
    # Some elements of the vectorization were inspired by the demos of the course

    eng_dic = enchant.Dict("en_US")
    lemmatizer = WordNetLemmatizer()

    # Remove special characters
    documents_no_specials = remove_specials_characters(documents)
    # remove stop words and tokenize
    documents_no_stop = []
    for document in documents_no_specials:
        new_text = []
        for word in document.lower().split():
            if word not in STOPWORDS:
                new_text.append(word)
        documents_no_stop.append(new_text)

    # Remove numbers
    documents_no_stop_no_numeric = remove_numerical(documents_no_stop)

    # lemmattizing tokens (better than stemming by taking word context into account)
    documents_no_stop_no_numeric_lemmatize = [[lemmatizer.lemmatize(token) for token in text]
                                              for text in documents_no_stop_no_numeric]

    # remove non-english words
    documents_no_stop_no_numeric_lemmatize_english = [[token for token in text if (eng_dic.check(token))]
                                                      for text in documents_no_stop_no_numeric_lemmatize]

    return [" ".join(doc) for doc in documents_no_stop_no_numeric_lemmatize_english]


def keepAdjectives(documents, NEGATIONWORDS, POSITIVEWORDS, NEGATIVEWORDS, negation_distance_threshold):
    """
    Keep only the adjectives and count the positive and negative ones
    :param documents: List of documents
    :param NEGATIONWORDS: Negation words such as not, didn't, don't...
    :param POSITIVEWORDS: Positive adjectives such as beautiful, great...
    :param NEGATIVEWORDS: Negative adjectives such as bad, disgusting...
    :param negation_distance_threshold: distance between a negation word and a adjective under which it is considered as
     inverted ('not good' means 'bad')
    :return: 
    """
    document_neg_pos = []
    document_neg_count = np.zeros(documents.shape[0])
    document_pos_count = np.zeros(documents.shape[0])
    for idx, row in enumerate(documents):
        review = []
        new_word = ""
        negation_distance = 0
        for word in row.split(" "):
            if len(new_word) != 1:
                negation_distance += 1
                if negation_distance > negation_distance_threshold:
                    new_word = ""
            if word in NEGATIONWORDS:
                new_word = word
                negation_distance = 0
            elif word in POSITIVEWORDS:
                if len(new_word) == 0:
                    review.append(word)
                    document_pos_count[idx] += 1
                else:
                    # print(idx)
                    document_neg_count[idx] += 1
                    new_word = new_word + word
                    review.append(new_word)
                    new_word = ""
            elif word in NEGATIVEWORDS:
                if len(new_word) == 0:
                    review.append(word)
                    document_neg_count[idx] += 1
                else:
                    # print(idx)
                    document_pos_count[idx] += 1
                    new_word = new_word + word
                    review.append(new_word)
                    new_word = ""
        document_neg_pos.append(review)
    return document_neg_pos, document_pos_count, document_neg_count


def clean_filter_adj(df, summary_weight, negation_distance_threshold_review, negation_distance_threshold_summary,
                     tester=0):
    """
    remove special characters, stop words and numbers, lemmatize, remove non-english words and keep only the adjectives
    :param df: dataframe containing the documents
    :param summary_weight: how much more important is the summary compared to the review
    :param negation_distance_threshold_review: For a review : distance between a negation word and a adjective under
     which it is considered as inverted ('not good' means 'bad')
    :param negation_distance_threshold_summary: For a summary : distance between a negation word and a adjective under 
    which it is considered as inverted ('not good' means 'bad')
    :param tester: which element you would like to display as an example
    :return: 
    """
    negativeword_file = open('negative_words.txt')
    NEGATIVEWORDS = negativeword_file.read().split()
    positiveword_file = open('positive_words.txt')
    POSITIVEWORDS = positiveword_file.read().split()
    negationword_file = open('negation_words.txt')
    NEGATIONWORDS = negationword_file.read().split()
    # Not Merging both reviewText and summary because summary is considered more important
    documents = np.array(df['reviewText'])
    summaries = np.array(df['summary'])

    print('original test: ', documents[tester], '\n')
    print('summary test: ', summaries[tester], '\n')

    # create corpus (remove special characters, stop words and numbers, lemantize and remove non-english words)
    df['reviewCleaned'] = clean_text(documents, tester)
    df['summaryCleaned'] = clean_text(summaries, tester)

    # Keep only adjectives and see how many are positive
    documents_neg_pos, documents_pos_count, documents_neg_count = keepAdjectives(
        df['reviewCleaned'], NEGATIONWORDS, POSITIVEWORDS, NEGATIVEWORDS, negation_distance_threshold_review)
    print("Kept only adjectives pairs:", documents_neg_pos[tester],
          '\nPositive count=', documents_pos_count[tester], 'Negative counts=', documents_neg_count[tester])
    summaries_neg_pos, summaries_pos_count, summaries_neg_count = keepAdjectives(
        df['summaryCleaned'], NEGATIONWORDS, POSITIVEWORDS, NEGATIVEWORDS, negation_distance_threshold_summary)
    print("Kept only adjectives pairs:", summaries_neg_pos[tester],
          '\nPositive count=', summaries_pos_count[tester], 'Negative counts=', summaries_neg_count[tester])

    # Storing everything
    df["reviewWorded"] = [" ".join(doc) for doc in documents_neg_pos]
    df["summaryWorded"] = [" ".join(doc) for doc in summaries_neg_pos]
    df['review_count_pos'] = documents_pos_count
    df['review_count_neg'] = documents_neg_count
    df['summary_count_pos'] = summaries_pos_count
    df['summary_count_neg'] = summaries_neg_count
    df['review_count_difference'] = df['review_count_pos'] - df['review_count_neg']
    df['summary_count_difference'] = df['summary_count_pos'] - df['summary_count_neg']
    df['total_count_difference'] = df['review_count_difference'] + summary_weight * df['summary_count_difference']

    return df
