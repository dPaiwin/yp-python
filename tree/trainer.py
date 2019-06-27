from banker import Banker


class Trainer:
    def __init__(self, args):
        self.args = args
        super().__init__()

        self.banker = Banker(args)
        print('Fett has been created')

    def start_train(self):
        print('Fett started the train')
        self.banker.train(self.args['trainfile'])
        print('Fett finished the train')
        self.banker.pickle_save(self.args['savepath'])

        print('Fett saved the model')

    def return_epoch(self):
        return 0
