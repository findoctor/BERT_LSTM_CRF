import torch
DATA_PATH = "../seq2seq/Bert2Tag/DATA/cached_features"
args = {'data_path': DATA_PATH, 'run_mode':'train', 'batch_size': 16, 'seed':1001, \
        'device':torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
       }


START_TAG = "<START>"
STOP_TAG = "<STOP>"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = '<PAD>'
tag_to_ix = {"O": 0, "B": 1, "I": 2, "E": 3, "U": 4, START_TAG: 5, STOP_TAG: 6}

LSTM_HIDDEN = 768
BERT_HIDDEN = 512
MAX_LEN = 126  # rest two for [CLS] and [SEP]  Unified Length in mini-batch

n_epoch = 8
batch_size = 64
output_folder = "output_model"

num_test = 1280
num_train = 128000

batch_size_test = 64

lr = 1e-4

num_split = 2  # Split train data into 2 splits