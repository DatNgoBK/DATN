import torch.optim as optim
from tqdm import tqdm, trange
from torch.nn.functional import softmax
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import seaborn as sn
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.decomposition import TruncatedSVD
import joblib
from transformers import * 
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
np.random.seed(2018)





def save_model(filename, clf):
    with open(filename, 'wb') as f:
        joblib.dump(clf, f, compress=3)


class Training:
    def __init__(self, features, model, logger, label_list, config):
        self.best_sc = 0
        self.features = features
        #self.test_features = test_features
        self.model = model
        self.logger = logger
        self.epoches = config['epochs']
        self.batch_size = config['batch_size']
        self.label_list = label_list 

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = config['lr']

        self.loss_c = nn.MSELoss() if self.model.num_classes == 1 else nn.CrossEntropyLoss()

    def prepare_dataset_for_train_valid(self, features):
        input_ids = [feature.input_ids for feature in features]
        input_masks = [feature.input_mask for feature in features]
        segment_ids = [feature.segment_ids for feature in features]
        labels_id = [feature.label_id for feature in features]

        X_train = [(input_id, input_mask, segment_id) for (input_id, input_mask, segment_id) in zip (input_ids, input_masks, segment_ids)]
        X_train = np.array(X_train)
        y_train = np.array(labels_id)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=2018, stratify=y_train)

        return X_train, y_train, X_test, y_test


    def train(self):
        self.logger.info('Preprare data for training')
        X_train, y_train, X_test, y_test = self.prepare_dataset_for_train_valid(self.features)
        dataset_test = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
        dataset_test = DataLoader(dataset_test, batch_size=self.batch_size)
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2018).split(X_train, y_train))
        
        self.logger.info('Start training')

        for n_fold, (train_idx, val_idx) in enumerate(splits):
            self.logger.info('Training Fold_{}'.format(n_fold))
            #if n_fold!=5:
            #    continue
            X_train_, y_train_ = X_train[train_idx], y_train[train_idx]
            X_val_, y_val_ = X_train[val_idx], y_train[val_idx]
            dataset_train = TensorDataset(torch.LongTensor(X_train_), torch.LongTensor(y_train_))
            dataset_valid = TensorDataset(torch.LongTensor(X_val_), torch.LongTensor(y_val_))

            dataset_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
            dataset_valid = DataLoader(dataset_valid, batch_size=self.batch_size)
        
            self.model.to(self.device)
            all_sen_vecs = []

            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            num_train_optimization_steps = int(self.epoches * len(dataset_train) / 5)
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
            scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler
            self.model.train()
            optimizer.zero_grad()

            tsfm = self.model.bert

            for child in tsfm.children():
                for param in child.parameters():
                    if not param.requires_grad:
                        print("whoopsies")
                    param.requires_grad = False
            frozen = True

            for epoch in tqdm(range(self.epoches), desc="Epoch"):

                if epoch > 0 and frozen:
                    for child in tsfm.children():
                        for param in child.parameters():
                            param.requires_grad = True
                    frozen = False
                    del scheduler0
                    torch.cuda.empty_cache()

                for i, (xb_train, yb_train) in enumerate(dataset_train):
                    input_ids = xb_train[:, 0, :].to(self.device)
                    input_mask = xb_train[:, 1, :].to(self.device)
                    segment_ids = xb_train[:, 2, :].to(self.device)
                    intent_label = yb_train.to(self.device)

                    logits = self.model(input_ids, input_mask, segment_ids)
                    loss_s = self.loss_c(logits.view(-1, self.model.num_classes), intent_label.view(-1))

                    total_loss = loss_s.item() / (i + 1)
                    loss_s.backward()
                    if i!=0 and i%100==0:
                        self.logger.info('Loss average:{}'.format(total_loss))

                    if i % 5 == 0 or len(dataset_train) - 1 == i:
                        optimizer.step()
                        optimizer.zero_grad()
                        if not frozen:
                            scheduler.step()
                        else:
                            scheduler0.step()
            
                self.valid(dataset_train, des='Train', show_confusion_matrix=False)
                self.valid(dataset_valid, des='Valid', show_confusion_matrix=False)
            self.valid(dataset_test, des='Test')

        self.logger.info('Loading best model...')
        self.model.load_state_dict(torch.load('models/model-211.bin'))
        self.valid(dataset_test, des='Test')

    def valid(self, dataset_valid, des, show_confusion_matrix=True):
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
           # self.model.to(self.device)
            self.model.eval()
            for i, (x_train, y_train) in enumerate(dataset_valid):
                input_ids = x_train[:, 0, :].to(self.device)
                input_mask = x_train[:, 1, :].to(self.device)
                segment_ids = x_train[:, 2, :].to(self.device)
                label_id = y_train.to(self.device)

                logits = self.model(input_ids, input_mask, segment_ids)
                predictions = torch.argmax(softmax(logits, 1), 1)

                total += logits.size(0)
                correct += torch.sum(predictions==label_id).item()
                all_labels.extend(label_id.cpu())
                all_preds.extend(predictions.cpu())
            
            all_labels = [self.label_list[la] for la in all_labels]
            all_preds = [self.label_list[la] for la in all_preds]
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            acc = accuracy_score(all_labels, all_preds)
            f1_sc = f1_score(all_labels, all_preds, average='micro')
            f2_sc = f1_score(all_labels, all_preds, average='macro')
            f3_sc = f1_score(all_labels, all_preds, average='weighted')
            self.logger.info('{} acc:{}'.format(des, acc))
            self.logger.info('F1 score : {}'.format(f1_sc))
            self.logger.info('F1 score_macro : {}'.format(f2_sc))
            self.logger.info('F1 score_weighted : {}'.format(f3_sc))
            self.logger.info('Recall score: {}'.format(recall_score(all_labels, all_preds, average='macro')))
            self.logger.info('Precision score: {}'.format(precision_score(all_labels, all_preds, average='macro')))
            if f3_sc > self.best_sc and des=='Valid':
              torch.save(self.model.state_dict(), 'models/model-211.bin')
              self.best_sc = f2_sc
            
            if des=='Valid' or des=='Test':
                self.logger.info(classification_report(all_labels, all_preds))
  
        if show_confusion_matrix:
            y_true = pd.Series(all_labels, name="Actual")
            y_pred = pd.Series(all_preds, name="Predicted")
            df_confusion = pd.crosstab(y_true, y_pred, )
            df_confusion.to_csv('Maxtrix.csv')

