from keras.layers import Dense, Input, Bidirectional, GRU, concatenate, LSTM
from keras.layers import Embedding, SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from module.Model.model_utils.callbacks import generate_callbacks
from utils.utils import show_parameters
import numpy as np
from keras.utils.np_utils import to_categorical


class BiDirectLSTM:
    def __init__(self, config, classes, vocab_size, logger, embedding_matrix):
        self.models = {}
        self.logger = logger
        self.vocab_size = vocab_size
        self.config = config
        self.classes = classes
        self.nums_class = len(classes)
        self.pretrained_embedding = embedding_matrix
        self.model = self._build()
        self.checkpoint_best_model = 'model/block_seg_BiRNN.hdf5'
        self.callback_list = generate_callbacks(self.checkpoint_best_model)

    def _show_training_config_para(self):
        show_parameters(self.logger, self.config, 'Training')

    def _build(self):
        self._show_training_config_para()
        inputs = Input(shape=(self.config['max_len'], ))
        if self.pretrained_embedding is not None:
            self.logger.info("Found embedding matrix, setting trainable=false")
            embedding = Embedding(self.vocab_size,
                                  self.config['embedding_col'],
                                  weights=[self.pretrained_embedding],
                                  input_length=self.config['max_len'],
                                  trainable=False)
        else:
            self.logger.info("Not found embedding matrix, skip using pretrained model, setting trainable=true")
            embedding = Embedding(self.vocab_size,
                                  self.config['embedding_col'],
                                  embeddings_initializer='uniform',
                                  input_length=self.config['max_len'],
                                  trainable=True)

        x = embedding(inputs)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        x = Bidirectional(GRU(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        #x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
        avg_pooling = GlobalAveragePooling1D()(x)
        max_pooling = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pooling, max_pooling])
        outputs = Dense(self.nums_class, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        history = self.model.fit(train_x,
                                 train_y,
                                 epochs=self.config['epochs'],
                                 verbose=True,
                                 validation_data=(validate_x, validate_y),
                                 batch_size=self.config['batch_size'],
                                 callbacks=self.callback_list)

        predictions = self.predict(validate_x)
        return predictions, history

    def predict(self, validate_x):
        predictions = self.model.predict(validate_x)
        prediction_ = to_categorical(np.argmax(predictions, axis=1))
        return prediction_




