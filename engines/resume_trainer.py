import numpy as np
from tqdm import tqdm
import os
import paddle
from paddle import optimizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ResumeTrainer(object):
    def __init__(self, config, model):
        self.config = config

        self.model = model

        if config.use_cuda:
            paddle.device.set_device('gpu:0')
        else:
            paddle.device.set_device('cpu')

        self.text_model = self.model.text_encoder
        self.image_model = self.model.image_encoder
        self.multi_model = self.model.multi_encoder

        self.text_optim = optimizer.Adam(learning_rate=self.config.pre_lr,
                                         weight_decay=self.config.weight_decay_text,
                                         parameters=list(self.text_model.parameters()))
        self.image_optim = optimizer.Adam(learning_rate=self.config.pre_lr,
                                          weight_decay=self.config.weight_decay_image,
                                          parameters=list(self.image_model.parameters()))
        self.multi_optim = optimizer.Adam(learning_rate=self.config.multi_lr,
                                          weight_decay=self.config.weight_decay_multi,
                                          parameters=list(self.multi_model.parameters()))

    def train(self, train_loader):
        train_tqdm = tqdm(train_loader)
        all_out = []
        all_label = []
        all_loss = []

        # set text encoder and image encoder's learning_rate in fine_tune process
        if self.config.is_pretrain:
            self.text_optim.set_lr(self.config.text_ft_lr)
            self.image_optim.set_lr(self.config.image_ft_lr)

        for batch in train_tqdm:
            labels = batch['labels']
            output = self.model(**batch, eta=self.config.eta)

            # here zoom the value of penalty score by deviding the hyper-param tau
            text_p = output['text_penalty'] / self.config.tau
            image_p = output['image_penalty'] / self.config.tau

            # penalty score decides the output and training
            if text_p.item() < self.config.gamma and image_p.item() < self.config.gamma:
                loss = output['text_loss'] + output['image_loss'] + output['multi_loss']
                self.text_optim.clear_grad()
                self.image_optim.clear_grad()
                self.multi_optim.clear_grad()
                loss.backward()
                self.text_optim.step()
                self.image_optim.step()
                self.multi_optim.step()
                out = output['multi_logit']
            else:
                loss = output['text_loss'] + output['image_loss']
                self.text_optim.clear_grad()
                output['text_loss'].backward(retain_graph=True)
                self.text_optim.step()
                self.image_optim.clear_grad()
                output['image_loss'].backward(retain_graph=True)
                self.image_optim.step()

                text_conf = paddle.max(output['text_logit'], axis=1)
                image_conf = paddle.max(output['image_logit'], axis=1)
                # select final output using confidence
                if text_conf > image_conf:
                    out = output['text_logit']
                else:
                    out = output['image_logit']

            # if len(out.size()) == 1:
            #     out = paddle.unsqueeze(out, axis=0)
            all_out += out.detach().cpu().numpy()[:, 1].tolist()
            all_label += labels.cpu().numpy().tolist()
            all_loss.append(loss.item())
            train_tqdm.set_description('Loss: {}, text_p: {}, image_p: {}'.format(
                np.mean(all_loss), text_p.item(), image_p.item()))
        auc = roc_auc_score(all_label, all_out)
        return np.mean(all_loss), auc

    def pre_train(self, train_loader):
        train_tqdm = tqdm(train_loader)
        all_text_loss = []
        all_image_loss = []
        for batch in train_tqdm:
            output = self.model(**batch, eta=self.config.eta)
            loss = output['text_celoss'] + output['image_celoss']
            self.text_optim.clear_grad()
            self.image_optim.clear_grad()
            loss.backward()
            self.text_optim.step()
            self.image_optim.step()
            all_text_loss.append(output['text_celoss'].item())
            all_image_loss.append(output['image_celoss'].item())
        return np.mean(all_text_loss), np.mean(all_image_loss)

    def evaluate(self, model, valid_loader):
        all_out = []
        all_label = []
        all_loss = []
        for batch in valid_loader:
            with paddle.no_grad():
                labels = batch['labels']
                output = model(**batch, eta=self.config.eta)
                loss = output['text_loss'] + output['image_loss'] + output['multi_loss']
                text_p = output['text_penalty'] / self.config.tau
                image_p = output['image_penalty'] / self.config.tau
                text_conf = paddle.max(output['text_logit'], axis=1)
                image_conf = paddle.max(output['image_logit'], axis=1)

                if text_p < self.config.gamma and image_p < self.config.gamma:
                    out = output['multi_logit'].detach().cpu().numpy()
                elif text_conf > image_conf:
                    out = output['text_logit'].detach().cpu().numpy()
                else:
                    out = output['image_logit'].detach().cpu().numpy()
            all_loss.append(loss.item())
            all_out += out[:, 1].tolist()
            all_label += out.tolist()
        auc = roc_auc_score(all_label, all_out)
        return np.mean(all_loss), auc, all_out, all_label

    def test(self, model, test_loader, choose_threshold=0.5):
        loss, auc, all_out, all_label = self.evaluate(model, test_loader)
        predict = np.array(np.array(all_out) >= choose_threshold, dtype='int')
        acc = accuracy_score(all_label, predict)
        precision = precision_score(all_label, predict, average='macro')
        recall = recall_score(all_label, predict, averag='macro')
        f1 = f1_score(all_label, predict, average='macro')
        return {
            'auc': auc,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save(self, model_path):
        paddle.save(self.model, model_path)






















