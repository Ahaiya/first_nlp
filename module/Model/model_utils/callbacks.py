from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


def generate_callbacks(model_path):
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    check_point = ModelCheckpoint(model_path,
                                  monitor='val_accuracy',
                                  mode='max',
                                  verbose=1,
                                  save_best_only=True)

    return [early_stopping, check_point]


def load_trained_model(logger, model_path):
    logger.info("Loading saved model {}...".format(model_path))
    saved_model = load_model(model_path)
    return saved_model

