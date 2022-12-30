import paddle
import paddle.nn as nn
from paddle.vision import models


# Fully Convolutional Network
class Fcn(nn.Layer):
    def __init__(self,
                 input_dim,
                 feature_dim,
                 num_labels):
        '''
        :param input_dim:
        :param feature_dim:
        :param num_labels:
        '''
        super(Fcn, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_labels = num_labels

        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, round(self.input_dim/2)),
            nn.ReLU(),
            nn.Linear(round(self.input_dim/2), round(self.input_dim/4)),
            nn.ReLU(),
            nn.Linear(round(self.input_dim/4), self.feature_dim),
            nn.ReLU()
        )

        self.prediction_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.num_labels),
            nn.Sigmoid()
        )
        self.criterion = nn.CrossEntropyLoss(soft_label=False)

    def forward(self, batch, labels):
        feature = self.feature_layer(batch)
        logit = self.prediction_layer(feature)
        loss = self.criterion(logit, labels)
        return feature, logit, loss


class DOMFN(nn.Layer):
    def __init__(self,
                 attr_hidden_dim=None,
                 context_hidden_dim=None,
                 image_hidden_dim=None,
                 feature_dim=None,
                 num_labels=None,
                 fusion='concat'):
        '''
        :param attr_hidden_dim:
        :param context_hidden_dim:
        :param image_hidden_dim:
        :param feature_dim:
        :param num_labels:
        :param fusion: type of fusion
        '''

        super(DOMFN, self).__init__()

        self.attr_hidden_dim = attr_hidden_dim
        self.context_hidden_dim = context_hidden_dim
        self.image_hidden_dim = image_hidden_dim
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.fusion = fusion
        # hidden dim of concatenation of attributes vector and contextual embedding
        text_dim = self.attr_hidden_dim + self.context_hidden_dim if self.attr_hidden_dim is not None \
            else self.context_hidden_dim
        self.text_encoder = Fcn(text_dim, self.feature_dim, self.num_labels)
        self.image_encoder = Fcn(self.image_hidden_dim, self.feature_dim, self.num_labels)

        self.multi_encoder = nn.Sequential(
            nn.Linear((2 if self.fusion == 'concat' else 1)*self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_labels),
            nn.Sigmoid()
        )
        self.criterion = nn.CrossEntropyLoss(soft_label=False)

    def forward(self,
                attr_embeds=None,
                context_embeds=None,
                image_embeds=None,
                labels=None,
                eta=None):
        '''

        :param attr_embeds: tensor of shape (batch_size, attr_hidden_dim)
        :param context_embeds: tensor of shape (batch_size, max_exp_num, hidden_dim)
        exp_num indicates the number of experience in the resume
        :param image_embeds: tensor of shape (batch_size, max_page_num, hidden_dim)
        page_num indicates the number of resume
        :param labels: tensor of shape (batch_size)
        :param eta: float
        :return:
        '''

        context_vec = paddle.max(context_embeds, axis=1)
        text_embeds = paddle.concat([context_vec, attr_embeds], axis=1) \
            if attr_embeds is not None else context_vec
        image_embeds = paddle.max(image_embeds, axis=1)
        text_feat, text_logit, text_celoss = self.text_encoder(text_embeds, labels)
        image_feat, image_logit, image_celoss = self.image_encoder(image_embeds, labels)
        multi_logit = None
        if self.fusion == 'concat':
            multi_feat = paddle.concat([text_feat, image_feat], axis=1)
            multi_logit = self.multi_encoder(multi_feat)
        elif self.fusion == 'mean':
            multi_feat = paddle.mean(paddle.stack([text_feat, image_feat],
                                                  axis=1), axis=1)
            multi_logit = self.multi_encoder(multi_feat)
        elif self.fusion == 'max':
            multi_feat = paddle.max(paddle.stack([text_feat, image_feat],
                                                     axis=1), axis=1)
            multi_logit = self.multi_encoder(multi_feat)

        multi_celoss = self.criterion(multi_logit, labels)
        text_penalty = paddle.max((text_logit-multi_logit)**2, axis=1)
        image_penalty = paddle.max((image_logit-multi_logit)**2, axis=1)
        text_loss = 0.5 * text_celoss * text_celoss - eta * text_penalty
        image_loss = 0.5 * image_celoss * image_celoss - eta * image_penalty
        multi_loss = text_celoss + image_celoss + multi_celoss

        return {
            'text_logit': text_logit,
            'image_logit': image_logit,
            'multi_logit': multi_logit,

            'text_celoss': text_celoss,
            'image_celoss': image_celoss,
            'multi_celoss': multi_celoss,

            'text_loss': text_loss,
            'image_loss': image_loss,
            'multi_loss': multi_loss,

            'text_penalty': text_penalty,
            'image_penalty': image_penalty
        }













