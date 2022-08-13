import argparse

import yaml
from datetime import datetime
import logging
import os
import sys
from module.preprocessor import Preprocessor
from module.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processing command line")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--loglevel', type=str, default='INFO')
    args = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    LOG_DIR = './logs'

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    logfile = datetime.now().strftime("%Y%m%d_%H%M")
    logging.basicConfig(filename=LOG_DIR + '/' + logfile, format=FORMAT, level=args.loglevel)
    logger = logging.getLogger('global_logger')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)

    classes = ["Personal_Info", "Job_Intent", "Education", "Working_Experience", "Projects", "Skill", "Introduction",
               "Key_Word", "Others"]

    with open(args.config) as cfg:
        try:
            config = yaml.safe_load(cfg)
            # print(config['preprocessing']['data_path'])
            preprocessor = Preprocessor(config['preprocessing'], logger)
            train_x, validate_x, train_y, validate_y= preprocessor.process()
            vocab_size = preprocessor.vocab_size
            #print(vocab_size)
            pretrained_embedding = preprocessor.embedding_matrix

            trainer = Trainer(config['training'], classes, logger, vocab_size, pretrained_embedding)
            model, accuracy, cls_report, history = trainer.fit_and_validate(train_x, train_y, validate_x, validate_y)
            logger.info("Accuracy : {}".format(accuracy))
            logger.info("\n{}\n".format(cls_report))


        except yaml.YAMLError as err:
            print("config file error : {}".format(err))