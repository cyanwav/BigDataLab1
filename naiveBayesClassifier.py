import pandas as pd
import ast
import numpy as np

train_data = pd.read_csv('processed_train_data.csv')
train_text = train_data['text']
train_score = train_data['score']

def calculate_class_probability(train_score):
    class_probability = train_score.value_counts() / len(train_score)
    positive_class_probability = class_probability.get(1, 0)
    negative_class_probability = class_probability.get(0, 0)
    return positive_class_probability, negative_class_probability

def create_maps(train_text, train_score):
    map_positive = {}
    map_negative = {}

    total_negative_words = 0
    total_positive_words = 0

    for i in range(len(train_text)):
        for word in ast.literal_eval(train_text[i]):
            if train_score[i] == 1:
                total_positive_words += 1
                map_positive[word] = map_positive.get(word, 0) + 1
            else:
                total_negative_words += 1
                map_negative[word] = map_negative.get(word, 0) + 1
    
    all_words = set(map_positive.keys()).union(set(map_negative.keys()))
    total_count = len(all_words)

    return map_positive, map_negative, total_positive_words, total_negative_words, total_count

def calculate_word_given_class_probability(words, map_positive, map_negative, total_positive_words, total_negative_words, total_count):
    alpha = 1 # Laplace smoothing
    log_positive_given_probability = 0
    log_negative_given_probability = 0
    
    for word in ast.literal_eval(words):
        log_positive_given_probability += np.log((map_positive.get(word, 0) + alpha) / (total_positive_words + alpha * total_count))
        log_negative_given_probability += np.log((map_negative.get(word, 0) + alpha) / (total_negative_words + alpha * total_count))

    return log_positive_given_probability, log_negative_given_probability

def naive_bayes_classifier(file, map_positive, map_negative, total_positive_words, total_negative_words, total_count, positive_class_probability, negative_class_probability):
    file_text = file['text']
    file_score = file['score']

    correct_count = 0

    log_positive_class_probability = np.log(positive_class_probability)
    log_negative_class_probability = np.log(negative_class_probability)
    
    for i in range(len(file_text)):
        predicted_score = -1
        log_positive_given_probability, log_negative_given_probability = calculate_word_given_class_probability(
            file_text[i], map_positive, map_negative, total_positive_words, total_negative_words, total_count
        )
        log_positive = log_positive_given_probability + log_positive_class_probability
        log_negative = log_negative_given_probability + log_negative_class_probability

        if log_positive > log_negative:
            predicted_score = 1
        else:
            predicted_score = 0

        if predicted_score == file_score[i]:
            correct_count += 1

    return correct_count / len(file_text)

positive_class_probability, negative_class_probability = calculate_class_probability(train_score)
map_positive, map_negative, total_positive_words, total_negative_words, total_count = create_maps(train_text, train_score)

test_data = pd.read_csv('processed_test_data.csv')
result = naive_bayes_classifier(test_data, map_positive, map_negative, total_positive_words, total_negative_words, total_count, positive_class_probability, negative_class_probability)
print(result)