import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from utils.utils import show_parameters, clean_data
from utils.jieba_emb import pretrained_embdding, tokenizer
from keras.utils.np_utils import to_categorical


class Preprocessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.df = pd.DataFrame()        #加载时 暂存数据
        self.train_data, self.trian_label = None, None
        self.train_x, self.train_y, self.val_x, self.val_y = None, None, None, None
        self.embedding_matrix = None            # vocab_size x embedding col
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
        
        self._load_xlsx_files()
        self.embedding = self.generate_embedding()

    def _show_config_parameters(self):
        show_parameters(self.logger, self.config, 'Preprocessing')
        #title = "Preprocessing config parameters"
        #self.logger.info(title.center(40, '-'))
        #self.logger.info("---split_ratio = {}".format(self.config['split_ratio']))
        #self.logger.info("---random_state = {}".format(self.config['random_state']))
        #self.logger.info("---max_len = {}".format(self.config['max_len']))
        

    def _load_xlsx_files(self):
        self._show_config_parameters()
        self.logger.info("Loading training data...")
        # 加载数据, 并划分 数据集
        nums_resume = 0
        for sub_dir in os.listdir(self.config['data_path']):
            cur_dir = os.path.join(self.config['data_path'], sub_dir)
            if not os.path.isdir(cur_dir):
                continue
            for filename in os.listdir(cur_dir):
                file_path = os.path.join(cur_dir, filename)
                if os.path.isfile(file_path):
                    nums_resume += 1
                    self.read_xlsx(file_path, nums_resume)
        self.df = self.df.reset_index(drop=True)

        orig_data = self.df
        self.train_data, self.trian_label = self._parse_orig_data(orig_data)


        self.logger.info('Splitting datasets, the splits ratio is {}, random state is {}'.format(self.config['split_ratio'],
                                                                                                self.config['random_state']))
        # 划分 数据集                                                                                        
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(
                                                                self.train_data, 
                                                                self.trian_label, 
                                                                test_size=self.config['split_ratio'], 
                                                                random_state=self.config['random_state'])
        self.logger.info('Loading done.')

    def read_xlsx(self, file, id):
        data = pd.read_excel(file, header=None, names=['resume', 'label_detail'])
        data['resume_id'] = np.full(data.shape[0], id)
        data['resume'] = data['resume'].astype(str)
        data['label_detail'] = data['label_detail'].astype(str)
        data['label'] = data['label_detail'].str.split("-").str.get(0)
        ls = data.loc[data['label'].str.len() == 0, 'label'].tolist()
        if len(ls) != 0:
            print(file)
        self.df = pd.concat([self.df, data])

    def _parse_orig_data(self, orig_data):
        self.logger.info('Parsing original resume dataset...')
        orig_data.drop(orig_data[~orig_data.label.str.isdigit()].index, inplace=True)
        orig_data['label'] = orig_data['label'].astype(int) - 1

        orig_data.drop(orig_data[(orig_data.label < 0) | (orig_data.label > 9)].index, inplace=True)
        train_data = orig_data['resume']
        label = orig_data['label'].values
        label = to_categorical(label, num_classes=9)
        train_data = clean_data(train_data).values         # 数据清洗
        train_data = tokenizer(train_data).values          # 分词
        return train_data, label


# 词嵌入,  暂时不考虑 测试集

    #生成 pre_embedding
    def generate_embedding(self):
        train_data = self.train_data
        return pretrained_embdding(train_data)

    def process(self):
        convertor = self.config['convertor']
        train_x, val_x, train_y, val_y = self.train_x, self.val_x, self.train_y, self.val_y

        if convertor == 'ok':
            train_x, val_x = self.nn_text2vec(train_x, val_x)

        return train_x, val_x, train_y, val_y

    def nn_text2vec(self, train_x, val_x):
        self.logger.info("Vecterizing data for neural network training...")
        specialchars = ['<pad>', '<unk>']           #特殊标记
       
        embedding = self.embedding[0]

        self.logger.info("Creating vocabulary...")
        vocab = specialchars + list(embedding.keys())  #所有词
        self.vocab_size = len(vocab)
        self.embedding_matrix = np.zeros((self.vocab_size, self.config['embedding_col']))

        #特殊标记， 以均匀分布 产生 词向量
        for token in specialchars:
            embedding[token] = np.random.uniform(low=-1, high=1, size=(self.config['embedding_col']))
        
        # 建立 wordidx 和 idx2word  对应关系
        for index, word in enumerate(vocab):
            self.word2idx[word] = index
            self.idx2word[index] = word
            self.embedding_matrix[index] = embedding[word]      #导入

        self.logger.info("Done. Got {} words".format(len(self.word2idx.keys())))
        
        # paddding 成 长度一致的 sequence           需要重新分词嘛？
        self.logger.info("Preparing data for training...")
        train_x_idx = []
        for sentence in train_x:
            indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence]
            train_x_idx.append(indices)
        # train_x_idx = np.array(train_x_idx)

        val_x_idx = []
        for sentence in val_x:
            indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence]
            val_x_idx.append(indices)
        # val_x_idx = np.array(val_x_idx)

        train_x_in = pad_sequences(train_x_idx,
                                    maxlen=self.config['max_len'],
                                    padding='post',
                                    value=self.word2idx['<pad>'])

        val_x_in = pad_sequences(val_x_idx,
                                    maxlen=self.config['max_len'],
                                    padding='post',
                                    value=self.word2idx['<pad>'])
        return train_x_in, val_x_in


    # @staticmethod
    # def loa
    #
    # d_word_embedding(filename):
    #     # 这里 假设 第一个是 token，后面的 是 该token 的 vec
    #     file_in = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #     data = {}
    #     for line in file_in:
    #         tokens = line.rstrip().split(' ')
    #         data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    #     return data