import pymorphy2
import numpy as np


class PyMorphManager:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.tag_set = [tag for tag in self.morph.TagClass.KNOWN_GRAMMEMES]
        self.tag_set.sort()
        self.tags_size = len(self.tag_set)

    def get_normal_form(self, tokens):
        result = []
        for token in tokens:
            result.append(self.normalise(token))
        return result

    def normalise(self, word):
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

    def get_tags(self, words):
        result = []
        for word in words:
            res = np.zeros([self.tags_size])
            tags = self.morph.parse(word)[0].tag
            for index, tag in enumerate(self.tag_set):
                if tag in tags:
                    res[index] = 1
            result.append(res)
        return result
