import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv('processed_train_data.csv')
train_text = train_data['text']
train_score = train_data['score']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
vectorized_text = vectorizer.fit_transform(train_text).toarray()

learning_rate = 0.001
gamma = 0.01
lambda_ = 0.01
num_iterations = 25
weight = np.zeros(vectorized_text.shape[1])
bias = 0

def train_svm():
    global weight, bias
    for _ in range(num_iterations):
        for j in range(len(vectorized_text)):
            y = train_score[j] * 2 - 1  # Convert 0/1 labels to -1/1
            x = vectorized_text[j]
            if y * (np.dot(weight, x) + bias) < 1:
                weight -= learning_rate * (2 * lambda_ * weight - y * x)
                bias += learning_rate * y
            else:
                weight -= learning_rate * 2 * lambda_ * weight

def predict_svm(test_file):
    test_text = test_file['text']
    test_vectorized_text = vectorizer.transform(test_text).toarray()
    predictions = np.dot(test_vectorized_text, weight) + bias
    return np.sign(predictions)

def calculate_accuracy(predictions, true_labels):
    correct_predictions = np.sum(predictions == true_labels)
    accuracy = correct_predictions / len(true_labels)
    return accuracy

train_svm()

test_data = pd.read_csv('processed_test_data.csv')
file_score = test_data['score']

predictions = predict_svm(test_data)
true_labels = file_score * 2 - 1
accuracy = calculate_accuracy(predictions, true_labels)
print(f"Accuracy: {accuracy:.2f}")