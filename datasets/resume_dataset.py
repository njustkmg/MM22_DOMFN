import numpy as np
import paddle
from paddle.io import Dataset


class ResumeDataset(Dataset):
    def __init__(self,
                 attr_data=None,
                 context_data=None,
                 img_data=None,
                 data_content=None,
                 ):
        '''
        Dataset for multi-modal resume data
        :param attr_data: attributes in the resume, one-hot vector
        :param context_data: context in the resume, preprocess from pre-train bert
        :param img_data: layout of the resume, preprocess from resnet
        :param data_content: random index for train, dec, test set and corresponding label
        '''
        super(ResumeDataset, self).__init__()
        self.index = np.array(data_content['index'])
        self.label = np.array(data_content['label'])

        self.attr_vec = attr_data[self.index]
        self.context_vec = context_data[self.index]
        self.img_vec = img_data[self.index]

    def __getitem__(self, index):
        inst_attr = self.attr_vec[index]
        inst_context = self.context_vec[index]
        inst_img = self.img_vec[index]
        inst_label = self.label[index]

        return {
            'attr_embeds': paddle.to_tensor(inst_attr, dtype='float32'),
            'context_embeds': paddle.to_tensor(inst_context, dtype='float32'),
            'image_embeds': paddle.to_tensor(inst_img, dtype='float32'),
            'labels': paddle.to_tensor(inst_label, dtype='int64')
        }

    def __len__(self):
        return len(self.index)







