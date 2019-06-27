import pymorphy2
from tqdm import tqdm
import numpy as np

from morph_manager import MorphManager
from morph_manager import TagSet


class AllTagSet(TagSet):
    def get_tags(self):
        try:
            return self.tags
        except:
            self.tags = [
                tag for tag in pymorphy2.MorphAnalyzer().TagClass.KNOWN_GRAMMEMES]
            self.tags.sort()
            return self.tags


class PyMorphManager(MorphManager):
    def __init__(self, tag_set: TagSet):
        super().__init__(tag_set)

        self.morph = pymorphy2.MorphAnalyzer()

        self._norm_cache = {}
        self._tag_cache = {}

    def get_normal_form(self, tokens):
        '''
        `words` is a `list` of `str` which may contain single spaces
        '''

        result = []
        for token in tokens:
            try:
                result.append(self._norm_cache[token])
            except:
                normal_token = self.normalise(token)
                self._norm_cache[token] = normal_token
                result.append(normal_token)
        return result

    def normalise(self, word):
        if ' ' not in word:
            w = self.morph.parse(word)[0]
            if w.tag.POS in ['COMP', 'GRND']:
                word_norm = w.word
            elif w.tag.POS == 'PRTF':
                word_norm = w.inflect({'masc', 'sing', 'nomn'}).word
            elif w.tag.POS == 'PRTS':
                word_norm = w.inflect({'masc', 'sing'}).word
            else:
                word_norm = w.normal_form
            return word_norm
        else:
            return ' '.join([self.normalise(w) for w in word.split(' ')])

    def get_tags(self, words):
        result = []
        for word in words:
            try:
                result.append(self._tag_cache[word])
            except:
                res = np.zeros([self.tag_set.get_size()])
                tags = self.morph.parse(word)[0].tag
                for index, tag in enumerate(self.tag_set.get_tags()):
                    if tag in tags:
                        res[index] = 1
                self._tag_cache[word] = res
                result.append(res)
        return result
