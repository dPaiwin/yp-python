from tqdm import tqdm
import numpy as np
import pickle


class TagSet(object):

    def __init__(self):
        self.TAG_SIZE = len(self.get_tags())

    def get_tags(self):
        raise NotImplementedError("Please implement this method")

    def get_size(self):
        return self.TAG_SIZE

    def pickle_save(self, path):
        pickle.dump(self, open('%s/tag_set.pickle' % path, 'wb'))

    @staticmethod
    def pickle_load(path):
        return pickle.load(open('%s/tag_set.pickle' % path, 'rb'))


class MorphManager(object):
    def __init__(self, tag_set: TagSet):
        self._tag_set = tag_set

    def get_normal_form(self, words):
        raise NotImplementedError('Implement this method')

    def get_tags(self, words):
        '''
        Takes `list` of `str` `words` which have no inner spaces.
        Should return a `list` of `lists` of `int` (0 or 1).
        '''
        raise NotImplementedError('Implement this method')

    @property
    def tag_set(self):
        return self._tag_set
