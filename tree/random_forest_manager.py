import re
from os.path import join as pathjoin

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import scipy.sparse
from scipy.sparse import vstack

import openpyxl as px
import numpy as np
import pickle
from tqdm import tqdm

from pymorph_manager import AllTagSet, PyMorphManager


def extract_words(text):
    return re.findall(r'[\-а-яёА-ЯЁa-zA-Z]+', text.lower())


class Banker:
    def __init__(self, args):
        self.tag_set = AllTagSet()
        self.morpher = PyMorphManager(self.tag_set)
        args['Morpher'] = self.morpher

        self.types = {}

        self.rfc = RandomForestClassifier(
            n_estimators=130,  # best=100
            random_state=46568781)

    def get_tfidf_features(self, sentence):
        """sentence is a str"""

        return self.tfidf_vect.transform([sentence]).toarray().flatten()

    def get_tag_features(self, words):
        """sentence is a str.
        Returns tags over sentence words"""

        if not len(words):
            return np.zeros(self.morpher.tag_set.get_size())

        return np.amax(
            np.stack(
                self.morpher.get_tags(words)
            ),
            0
        )

    def get_char_features(self, words):
        """sentence is a str.
        Returns tags over sentence words"""

        if not len(words):
            return np.zeros(self.char_feat.get_features_len())

        return np.amax(
            np.stack(
                [self.char_feat.get_features(word) for word in words]
            ),
            0
        )

    def get_all_features(self, sentence):
        """sentence is a str"""

        words = extract_words(sentence)

        f_tfidf = self.get_tfidf_features(
            ' '.join(self.morpher.get_normal_form(words)))
        f_tag = self.get_tag_features(words)
        f_char = self.get_char_features(words)

        return np.hstack(
            [
                [0 if len(words) == 0 else np.mean(
                    [len(word) for word in words])],
                [len(words)],
                f_tfidf,
                f_tag,
                f_char
            ]
        ).tolist()

    def get_features_count(self):
        try:
            return self.features_count
        except:
            features = self.get_all_features('dummy string')
            self.features_count = len(features)
            return self.features_count

    def predict(self, sentences):
        return list(map(
            int,
            self.rfc.predict(
                [self.get_all_features(sentence) for sentence in sentences]
            )
        ))

    def predict_prob(self, sentences):
        """Returns float faq probability for each sentence in sentences.

        sentences: list of strings"""

        predict = self.rfc.predict_proba(
            [self.get_all_features(sentence) for sentence in sentences]
        )

        return [max(elem[1:]) for elem in predict]

    def train(self, datasets, to_test=False):
        self.dataset = self.load_dicts(datasets)
        self.tfidf_vect = TfidfVectorizer()
        self.types = {x: str(x) for x in set([z['type id'] for z in self.dataset])}

        texts = []
        for type_id in self.types:
            for q in tqdm(list(filter(lambda q: q['type id'] == type_id, self.dataset)), desc='Creating tfidf'):
                words = extract_words(q['text'])
                texts.append(' '.join(self.morpher.get_normal_form(words)))

        self.tfidf_vect.fit(texts)
        self.char_feat = CharFeaturer([q['text'].lower() for q in self.dataset])

        order = np.arange(len(self.dataset))
        np.random.shuffle(order)

        x = scipy.sparse.csr_matrix(
            (
                0,
                self.get_features_count()
            )
        )

        y = []
        block_size = 3000
        x_local = []
        for ind in tqdm(order, desc='Generating features'):
            x_local.append(self.get_all_features(self.dataset[ind]['text']))
            if len(x_local) == block_size:
                sparse = scipy.sparse.csr_matrix(x_local)
                x = vstack([x, sparse])

                x_local.clear()
            y.append(self.dataset[ind]['type id'])
        if len(x_local):
            x = vstack([x, scipy.sparse.csr_matrix(x_local)])

            x_local.clear()

        if not to_test:
            x_test = None
            y_test = None
            x_train = x
            y_train = y
        else:
            x_test, x_train = x[:len(x) // 10], x[len(x) // 10:]
            y_test, y_train = y[:len(y) // 10], y[len(y) // 10:]

        print('Fitting')
        self.rfc.fit(x_train, y_train)
        print('Done')

        if to_test:
            return self.rfc.score(x_test, y_test)

    @staticmethod
    def load_dicts(trainfile):
        result = []
        print('Loading file {}'.format(trainfile))
        w = px.load_workbook(trainfile)
        for sheet_name in w.get_sheet_names():
            sheet = w[sheet_name]
            for i_row, row in enumerate(sheet.rows):
                type_id = int(row[0].value)
                line = str(row[1].value)
                if line == '':
                    continue
                result.append({'type id': type_id, 'text': re.sub(r'\s', ' ', line)})
        return result

    def pickle_save(self, path):
        pickle.dump(self.rfc, open(pathjoin(path, 'rfc.pickle'), 'wb'))
        pickle.dump(self.tfidf_vect, open(pathjoin(path, 'tfidf.pickle'), 'wb'))
        pickle.dump(self.char_feat, open(pathjoin(path, 'char.pickle'), 'wb'))
        with open(pathjoin(path, 'types.csv'), 'w', encoding='utf-8') as f:
            for type_id in self.types:
                f.write('%d\t%s\n' % (type_id, self.types[type_id]))

    def pickle_load(self, path):
        self.rfc = pickle.load(open(pathjoin(path, 'rfc.pickle'), 'rb'))
        self.tfidf_vect = pickle.load(open(pathjoin(path, 'tfidf.pickle'), 'rb'))
        self.char_feat = pickle.load(open(pathjoin(path, 'char.pickle'), 'rb'))
        with open(pathjoin(path, 'types.csv'), 'r', encoding='utf-8') as f:
            for line in f:
                type_id, text = line.strip().split('\t')
                self.types[int(type_id)] = text


class CharFeaturer:
    def __init__(self, sentences, modifier=None):

        self.ALPHABET = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
        self.ALPHABET_SIZE = 33
        self.FEAT_CHR_MIN = 10
        self.FEAT_CHR_MAX = 2500
        self.FEAT_CHR_RANGE = [3]

        self.ch_codes = dict((ch, self.ALPHABET.index(ch)) for ch in self.ALPHABET)
        self.ch_codes['ё'] = self.ch_codes['е']

        self.hash_ch = dict((elem, index)
                            for index, elem in enumerate(self.get_hash_ch(sentences)))

    def get_ord(self, ch):
        try:
            return self.ch_codes[ch]
        except:
            return 32

    def get_ords(self, word):
        """
        Выделяем полиграммы из слова
        """
        word = '<{}>'.format(word)
        res = []
        for length in self.FEAT_CHR_RANGE:  # для всех полиграмм указанных длин
            for i in range(length - 1, len(word)):  # по диапазонам длины полиграммы
                counter = length - 1  # модификатор индекса
                output = 0
                while counter >= 0:  # с первой буквы до последней буквы диапазона
                    output += self.get_ord(word[i - counter]) * (self.ALPHABET_SIZE ** counter)
                    counter -= 1

                res.append(output)

        return res

    def get_hash_ch(self, sentences):
        f = np.zeros(self.ALPHABET_SIZE ** max(self.FEAT_CHR_RANGE))
        for sentence in tqdm(sentences, desc='Generating char embs'):
            words = extract_words(sentence)
            for word in words:
                for index in self.get_ords(word):
                    f[index] += 1

        temp = np.nonzero(f > self.FEAT_CHR_MIN)[0]
        return np.nonzero(temp < self.FEAT_CHR_MAX)[0]

    def get_features_len(self):
        return len(self.hash_ch)

    def get_features(self, word):
        f = np.zeros(self.get_features_len())
        for index in np.array(self.get_ords(word)):
            if index in self.hash_ch:
                f[self.hash_ch[index]] = 1
        return f
