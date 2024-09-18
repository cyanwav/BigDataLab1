import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=10):
        self.lr = learning_rate
        self.lp = lambda_param
        self.ni = n_iters
        self.w = None
        self.b = None

    def train(self, X, y):
        # Extract number of samples and features from feature matrix X
        # row - sample, column - feature
        n_samples, n_features = X.shape

        # Binary classification
        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.ni):
            for idx, x_i in enumerate(X):
                 # The distance of each sample point from the classification boundary is greater than or equal to 1
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lp * self.w)
                else:
                    self.w -= self.lr * (2 * self.lp * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            print(_)


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

if __name__ == "__main__":
    train_data = pd.read_csv('processed_train_data.csv')
    test_data = pd.read_csv('processed_test_data.csv')

    vectorizer = CountVectorizer()
    train_text = vectorizer.fit_transform(train_data['text']).toarray()
    test_text = vectorizer.transform(test_data['text']).toarray()

    train_score = np.where(train_data['score'] <= 0, -1, 1)
    test_score = np.where(test_data['score'] <= 0, -1, 1)

    
    svm_model = SVM()

    start_time = time.time()
    svm_model.train(train_text, train_score)

    predict_score = svm_model.predict(test_text)
    end_time = time.time()

    duration = end_time - start_time
    
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
    
    print("SVM classification accuracy:", accuracy(test_score, predict_score))
    print("Time: ", duration)

