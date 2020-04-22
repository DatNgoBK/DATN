from preprocess import *
from training import Training
from transformers import *
from modeling import *
from utils import init_logger
import logging
import torch
from vncorenlp import VnCoreNLP

torch.cuda.manual_seed_all(2018)
torch.manual_seed(2018)
torch.backends.cudnn.deterministic = True

init_logger()
logger = logging.getLogger(__name__)

config = {}
config['batch_size'] = 16
config['epochs'] = 5
config['lr'] = 4e-5
config['max_seq_length'] = 256



zalo = ZaloDatasetProcessor()
zalo.load_from_path(dataset_path='dataset', train_filename='train.json', test_filename='test.json', dev_filename=None)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
toolkit = VnCoreNLP('VnCoreNLP-master/VnCoreNLP-1.1.1.jar', annotators='wseg,pos,ner,parse', max_heap_size='-Xmx2g')
zalo.convert_examples_to_features(zalo.train_data, zalo.label_list, 256, tokenizer)
zalo.convert_examples_to_vietnamese_features(zalo.train_data, zalo.label_list, 256, toolkit)

if __name__ == "__main__":
    
    NUM_OF_INTENT = 2
    config_model = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)

    model = QAModel(config_model, NUM_OF_INTENT, len(zalo.vocab), 300)
    training = Training(zalo, model, logger, config)
    training.train()
