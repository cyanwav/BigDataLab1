import pandas as pd

train_data = pd.read_csv('train.csv')
train_text = train_data['text']
train_score = train_data['score']

# Calculate P(class)
def calculate_class_probability(train_score):
    print(train_score.value_counts() / len(train_score))
    class_probability = train_score.value_counts() / len(train_score)
    positive_class_probability = class_probability[1]
    negative_class_probability = class_probability[0]
    return positive_class_probability, negative_class_probability

# Create maps of words by class
def create_maps(train_text, train_score):
    map_positive = {}
    map_negative = {}

    total_negative_words = 0
    total_positive_words = 0

    for i in range(len(train_text)):
        words = train_text[i].split()
        for word in words:
            if train_score[i] == 1:
                total_positive_words += 1
                if word in map_positive:
                    map_positive[word] += 1
                else:
                    map_positive[word] = 1
            else:
                total_negative_words += 1
                if word in map_negative:
                    map_negative[word] += 1
                else:
                    map_negative[word] = 1
    
    all_words = set(map_positive.keys()).union(set(map_negative.keys()))
    total_count = len(all_words)

    return map_positive, map_negative, total_positive_words, total_negative_words, total_count

# Calculate P(word | class)
def calculate_word_given_class_probability(text):
    map_positive, map_negative, total_positive_words, total_negative_words, total_count = create_maps(train_text, train_score)

    words = text.split()

    alpha = 1 # Laplace smoothing

    positive_given_probability = 1
    negative_given_probability = 1
    for word in words:
        if word in map_positive:
            positive_given_probability *= (map_positive[word] + alpha) / (total_positive_words + alpha * total_count)
        else:
            positive_given_probability *= (0 + alpha) / (total_positive_words + alpha * total_count)

        if word in map_negative:
            negative_given_probability *= (map_negative[word] + alpha) / (total_negative_words + alpha * total_count)
        else:
            negative_given_probability *= (0 + alpha) / (total_negative_words + alpha * total_count)


    return positive_given_probability, negative_given_probability

# Formula for Naive Bayes Classifier: P(class | word) = P(word | class) * P(class)
def naive_bayes_classifier(text):
    positive_given_probability, negative_given_probability = calculate_word_given_class_probability(text)
    positive_class_probability, negative_class_probability =  calculate_class_probability(train_score)

    print(positive_given_probability * positive_class_probability)
    print(negative_given_probability * negative_class_probability)

    if positive_given_probability * positive_class_probability > negative_given_probability * negative_class_probability:
        return 1
    else:
        return 0


result = naive_bayes_classifier('I would highly recommend this')
print(result)



