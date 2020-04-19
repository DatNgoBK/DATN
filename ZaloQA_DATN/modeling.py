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
    def __init__(self, config, num_classes):
        super(QAModel, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased', config=config)
        self.config = config
        self.lstm = nn.LSTM(self.config.hidden_size, 128)

        self.classifier = Classifier(128, self.num_classes)
        self.fn1 = nn.Linear(self.config.hidden_size + 128, self.config.hidden_size + 128)
        self.fn2 = nn.Linear(self.config.hidden_size + 128, 1)

    def forward(self, input_ids, attention_mask, segment_ids):
        outputs = self.bert(input_ids, attention_mask, segment_ids)
        seq_output = outputs[0]
        pool_output = outputs[1].view(-1, 1, self.config.hidden_size).repeat(1, 255, 1)

        lstm_input = seq_output.permute(1, 0, 2)
        outputs, states = self.lstm(lstm_input)
        outputs = outputs[1:].permute(1, 0, 2)
        cat_v = torch.cat((pool_output, outputs), 2)
        out_at = self.fn1(cat_v)
        attn_out = self.fn2(torch.tanh(out_at))
        scores = softmax(attn_out, 1)
        context_vec = scores * outputs
        context_vec =  torch.sum(context_vec, axis=1)
        logits = self.classifier(context_vec)
    #    slot_logits = self.slot_classifier(seq_output)
    #    x_1 = outputs[2][-1][:,0, ...]
    #    x_2 = outputs[2][-2][:,0, ...]
    #    x_3 = outputs[2][-3][:,0, ...]
    #    x_4 = outputs[2][-4][:,0, ...]
    #    cls_output = torch.cat((x_1, x_2, x_3, x_4), -1)
       # logits = self.classifier(cls_output)

        return logits

