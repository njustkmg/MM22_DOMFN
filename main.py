import sys
sys.path.append('..')
import pandas as pd
from datasets.resume_dataset import ResumeDataset
from models.encoder import DOMFN
from engines.resume_trainer import *
from config.resume_config import *
from paddle.io import DataLoader
import paddle

def setup_seed(seed):
    paddle.seed(seed)

def main(config):
    data_path = config.data_dir
    stats_vec = np.load(os.path.join(data_path, 'vec/stats.npy'))
    bert_vec = np.load(os.path.join(data_path, 'vec/bert.npy'))
    resnet_vec = np.load(os.path.join(data_path, 'vec/resnet50.npy'))
    train_data = pd.read_csv(os.path.join(data_path, 'index/train_data.csv'))
    valid_data = pd.read_csv(os.path.join(data_path, 'index/dev_data.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'index/test_data.csv'))
    train_dataset = ResumeDataset(stats_vec, bert_vec, resnet_vec, train_data)
    valid_dataset = ResumeDataset(stats_vec, bert_vec, resnet_vec, valid_data)
    test_dataset = ResumeDataset(stats_vec, bert_vec, resnet_vec, test_data)
    model = DOMFN(config.attr_dim, config.context_dim, config.image_dim,
                  config.feature_dim, config.num_label, config.fusion)
    trainer = ResumeTrainer(config, model)
    model_path = config.model_path + 'checkpoint_' + config.version + '.pdparams'
    # for pre-train
    if config.is_pretrain:
        train_loader = DataLoader(train_dataset, batch_size=config.pre_batch, shuffle=True)
        for epoch in range(config.pre_epoch):
            text_loss, image_loss = trainer.pre_train(train_loader)

    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch, shuffle=False)
    best_auc = 0
    # start training
    for epoch in range(config.epoch):
        train_loss, train_auc = trainer.train(train_loader)
        evaluate_loss, evaluate_auc, _, _ = trainer.evaluate(trainer.model, valid_loader)
        if evaluate_auc < best_auc:
            best_auc = evaluate_auc
            trainer.save(model_path)
    best_model = paddle.load(model_path)
    best_results = trainer.test(best_model, test_loader)
    print('best results:', best_results)


if __name__ == '__main__':
    config = get_args()
    setup_seed(config.seed)
    main(config)



