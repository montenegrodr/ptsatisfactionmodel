import csv
from enum import Enum
from os.path import join, exists
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier


class Data(object):
    def __init__(self, file_source):
        data = [d for d in self._load_data(file_source)]
        target = [t for t in self._load_target(file_source)]
        self.X_train, \
        self.X_test,  \
        self.y_train, \
        self.y_test = train_test_split(data, target, test_size=0.1)

    def _load_data(self, file_source):
        with open(file_source) as h:
            reader = csv.DictReader(h)
            for r in reader:
                yield r['final_answer']

    def _load_target(self, file_source):
        with open(file_source) as h:
            reader = csv.DictReader(h)
            for r in reader:
                yield r['rate']

    def trainset(self):
        return self.X_train

    def trainset_target(self):
        return self.y_train

    def testset(self):
        return self.X_test

    def testset_target(self):
        return self.y_test


class Classifier(object):
    def __init__(self, storage):
        print('Starting {} model'.format(type(self).__name__))
        self.storage = storage
        self.name = type(self).__name__
        self.clf = None
        self.result = None
        self.data = None

    def build(self, data):
        self.data = data
        self.train()
        self.save()
        self.test()

    def train(self):
        print('{} training...'.format(self.name))

    def save(self):
        print('{} saving model...'.format(self.name))

    def test(self):
        print('{} testing model...'.format(self.name))

    def predict(self):
        pass


class MultinomialNaiveBayes(Classifier):
    def __init__(self, storage):
        super(MultinomialNaiveBayes, self).__init__(storage)
        self.transformer = None
        self.count_vect = None
        self.model_path = join(
            self.storage, '{}_model.pkl'.format(self.name))
        self.transformer_path = join(
            self.storage, '{}_transformer.pkl'.format(self.name))
        self.count_vect_path = join(
            self.storage, '{}_count_vect.pkl'.format(self.name))
        self._load()

    def train(self):
        super(MultinomialNaiveBayes, self).train()
        self.count_vect = CountVectorizer()
        self.transformer = TfidfTransformer()
        train_counts = self.count_vect.fit_transform(self.data.trainset())
        train_tfidf = self.transformer.fit_transform(train_counts)
        self.clf = MultinomialNB().fit(
            train_tfidf, self.data.trainset_target())

    def save(self):
        super(MultinomialNaiveBayes, self).save()
        joblib.dump(self.clf, join(
            self.storage, '{}_model.pkl'.format(self.name)))
        joblib.dump(self.transformer, join(
            self.storage, '{}_transformer.pkl'.format(self.name)))
        joblib.dump(self.count_vect, join(
            self.storage, '{}_count_vect.pkl'.format(self.name)))

    def test(self):
        super(MultinomialNaiveBayes, self).test()
        predicted = self.predict(self.data.testset())

        precision, recall, f_score, true_sum = \
            precision_recall_fscore_support(self.data.testset_target(), predicted, average='micro')
        accuracy = accuracy_score(self.data.testset_target(), predicted)
        self.result = {
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'true_sum': true_sum,
            'accuracy': accuracy
        }

    def _load(self):
        if exists(self.model_path) and \
                exists(self.transformer_path) and \
                exists(self.count_vect_path):
            self.clf = joblib.load(self.model_path)
            self.transformer = joblib.load(self.transformer_path)
            self.count_vect = joblib.load(self.count_vect_path)

    def predict(self, texts):
        test_counts = self.count_vect.transform(texts)
        test_tfidf = self.transformer.transform(test_counts)
        return self.clf.predict(test_tfidf)


class SVM(Classifier):
    def __init__(self, storage):
        super(SVM, self).__init__(storage)
        self.transformer = None
        self.count_vect = None
        self.model_path = join(
            self.storage, '{}_model.pkl'.format(self.name))
        self.transformer_path = join(
            self.storage, '{}_transformer.pkl'.format(self.name))
        self.count_vect_path = join(
            self.storage, '{}_count_vect.pkl'.format(self.name))
        self._load()

    def train(self):
        super(SVM, self).train()
        self.count_vect = CountVectorizer()
        self.transformer = TfidfTransformer()
        train_counts = self.count_vect.fit_transform(self.data.trainset())
        train_tfidf = self.transformer.fit_transform(train_counts)
        self.clf = SGDClassifier().fit(
            train_tfidf, self.data.trainset_target())

    def save(self):
        super(SVM, self).save()
        joblib.dump(self.clf, self.model_path)
        joblib.dump(self.transformer, self.transformer_path)
        joblib.dump(self.count_vect, self.count_vect_path)

    def test(self):
        super(SVM, self).test()
        predicted = self.predict(self.data.testset())

        precision, recall, f_score, true_sum = \
            precision_recall_fscore_support(self.data.testset_target(), predicted, average='micro')
        accuracy = accuracy_score(self.data.testset_target(), predicted)
        self.result = {
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'true_sum': true_sum,
            'accuracy': accuracy
        }

    def _load(self):
        if exists(self.model_path) and \
           exists(self.transformer_path) and \
           exists(self.count_vect_path):
            self.clf = joblib.load(self.model_path)
            self.transformer = joblib.load(self.transformer_path)
            self.count_vect = joblib.load(self.count_vect_path)

    def predict(self, texts):
        test_counts = self.count_vect.transform(texts)
        test_tfidf = self.transformer.transform(test_counts)
        return self.clf.predict(test_tfidf)


class Classifiers(Enum):
    MNB = MultinomialNaiveBayes
    SVM = SVM

    @staticmethod
    def factory(model):
        if model == 'svm':
            return SVM
        elif model == 'mnb':
            return MultinomialNaiveBayes