from preprocess import *
from training import Training
from transformers import *
from modeling import *
from utils import init_logger
import logging
import torch

torch.cuda.manual_seed_all(2018)
torch.manual_seed(2018)
torch.backends.cudnn.deterministic = True

init_logger()
logger = logging.getLogger(__name__)

config = {}
config['batch_size'] = 16
config['epochs'] = 5
config['lr'] = 4e-5



zalo = ZaloDatasetProcessor()
zalo.load_from_path(dataset_path='dataset', train_filename='train.json', test_filename='test.json', dev_filename=None, train_augmented_filename='train_bqa.json')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
features_train = zalo.convert_examples_to_features(zalo.train_data, zalo.label_list, 256, tokenizer)
features_test = zalo.convert_examples_to_features(zalo.test_data, zalo.label_list, 256, tokenizer)
if __name__ == "__main__":
    
    NUM_OF_INTENT = 2
    config_model = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)#


    model = QAModel(config_model, NUM_OF_INTENT)
    training = Training(features_train, features_test, model, logger, zalo.label_list, config)
    training.train()