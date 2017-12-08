from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings


class Cosine_sim:
    """
    Estimator based on the cosine similarity of reviews. Does not work with big samples
    Based on the Distributed Information Systems course
    """
    X_train = []
    y_train = []
    categories = []

    def __init__(self, categories):
        self.categories = categories
        warnings.filterwarnings('ignore')

    def fit(self, X_train, y_train):
        """
        Fit the estimator to the data
        :param X_train: 
        :param y_train: 
        :return: 
        """
        self.X_train = X_train
        self.y_train = y_train

    def get_params(self, deep=False):
        """
        Return the parameters, here juste categories
        :param deep: 
        :return: 
        """
        return {'categories': self.categories}

    def predict(self, X_test, verbose=2):
        """
        Predict y
        :param X_test: data to predict from 
        :param verbose: 2 to get some info
        :return: 
        """
        y_pred = []
        # For each element to predict
        for idx, x in enumerate(X_test):
            if verbose == 2 and idx % 100 == 0:
                print(idx)

            best_category = ""
            best_sim = 0
            # for each category to check
            for cat in self.categories:
                # documents representing the category
                ref_docs = self.X_train[[self.y_train == cat][0]]
                # similarity between each reference document and the element
                sim = cosine_similarity(ref_docs, x)
                # If overall similarity better than best, change best
                if np.sum(sim) / len(ref_docs) > best_sim:
                    best_sim = np.sum(sim) / len(ref_docs)
                    best_category = cat
            # Assign best category
            y_pred.append(best_category)
        return y_pred
