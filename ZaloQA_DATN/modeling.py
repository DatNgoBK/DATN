import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers import RobertaConfig
from transformers import *
from torch.nn.functional import softmax, tanh
import torch 

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class QAModel(nn.Module):
    def __init__(self, config, num_classes, vob_size, embeddings_size):
        super(QAModel, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased', config=config)
        self.embeddings = nn.Embedding(vob_size, embeddings_size)
        self.config = config
        self.lstm = nn.LSTM(embeddings_size, 128)
        self.out_lstm = nn.Linear(128, 128)
        self.classifier = Classifier(4 * self.config.hidden_size + 128, self.num_classes)
    #    self.fn1 = nn.Linear(4 * self.config.hidden_size + 128, 4 * self.config.hidden_size + 128)
        self.fn2 = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, segment_ids, ner_ids):
        outputs = self.bert(input_ids, attention_mask, segment_ids)

        x_1 = outputs[2][-1][:,0, ...]
        x_2 = outputs[2][-2][:,0, ...]
        x_3 = outputs[2][-3][:,0, ...]
        x_4 = outputs[2][-4][:,0, ...]
        pool_output = torch.cat((x_1, x_2, x_3, x_4), -1)#.view(-1, 1, self.config.hidden_size * 4).repeat(1, 256, 1)


       # seq_output = outputs[0]
        #pool_output = outputs[1].view(-1, 1, self.config.hidden_size).repeat(1, 255, 1)
        seq_output = self.embeddings(ner_ids)
        lstm_input = seq_output.permute(1, 0, 2)
        outputs_, states = self.lstm(lstm_input)
        outputs_ = outputs_.permute(1, 0, 2)
        outputs_ = self.out_lstm(outputs_)
       # cat_v = torch.cat((pool_output, outputs_), 2)
       # out_at = self.fn1(cat_v)
        attn_out = self.fn2(torch.tanh(outputs_))
        scores = softmax(attn_out, 1)
        context_vec = scores * outputs_
        context_vec =  torch.sum(context_vec, axis=1)
        cat_v = torch.cat((pool_output, context_vec), 1)
        logits = self.classifier(cat_v)
    #    slot_logits = self.slot_classifier(seq_output)

       # logits = self.classifier(cls_output)

        return logits

