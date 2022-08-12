from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.Model.bidirectLSTM import BiDirectLSTM

class Trainer:
    def __init__(self, config, classes, logger, vocab_size, embedding_matrix):
        self.config = config
        self.logger = logger
        self.classes = classes
        self.vocab_size = vocab_size
        self.model = None
        self.embedding = embedding_matrix
        self._create_model(self.classes)

    def _create_model(self, classes):
        if self.config['model_name'] == 'BiLSTM':
            self.logger.info("Creating BiLSTM model...")
            self.model = BiDirectLSTM(self.config, classes, self.vocab_size, self.logger, self.embedding)
        else:
            self.logger.warning("Currently model {} is not be supported".format(self.config['model_name']))

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self.model

    def metrics(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        cls_report = classification_report(labels, predictions, zero_division=1)
        return accuracy, cls_report

    def validate(self, validate_x, validate_y):
        predictions = self.model.predict(validate_x, validate_y)
        return self.metrics(predictions, validate_y)

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        predictions, fitted = self.model.fit_and_validate(train_x, train_y, validate_x, validate_y)
        accuracy, report = self.metrics(predictions, validate_y)
        return self.model, accuracy, report, fitted
        

