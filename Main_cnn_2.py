root = '/home/yyang2/data/yyang2/pycharm/char_pytorch/'
import sys
sys.path.append(root)
import torch
from utils import train_step, validate_step, save_checkpoint_step
from tqdm import tqdm
from classifier_cnn_py import CNNClassifier
from cdata_loader_2 import DatasetGenerator, load_data
import os
import pandas as pd
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
writer = SummaryWriter('/home/yyang2/data/yyang2/Data/char_cnn/results/tensorboard/runs_jd_2_over')
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import shutil

# label_idx = ['negetive', 'positive']
n_classes = 2
epochs = 200
batch_size = 1024
lr = 0.0001

file_name_train = '/home/yyang2/data/yyang2/Data/交大NLP资料/data_train_over.csv'
data_raw_train, labels_raw_train,vocabulary_train, sequence_length_train = load_data(file_name_train)

file_name_val = '/home/yyang2/data/yyang2/Data/交大NLP资料/data_val_.csv'
data_raw_val, labels_raw_val,vocabulary_val, sequence_length_val = load_data(file_name_val)

file_name_test = '/home/yyang2/data/yyang2/Data/交大NLP资料/data_test_.csv'
data_raw_test, labels_raw_test,vocabulary_test, sequence_length_test = load_data(file_name_test)





model = CNNClassifier(400, n_classes, 38376, 256, [3,5,8], 256)

# loaded_model = torch.load(os.path.join(model_dir, save_model + '.tar'))
# model.load_state_dict(loaded_model['state_dict'])

model = model.cuda()

# model.to(device)

# criterion = torch.nn.CrossEntropyLoss(reduction='mean')
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = lr_scheduler.StepLR(optimizer, 40, gamma=0.2, last_epoch=-1)


print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))



data_train = DatasetGenerator(data_raw_train, labels_raw_train,vocabulary_train, sequence_length_train, set_name='train')
train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=8,
                                           shuffle=True, num_workers=0, pin_memory=True)

data_val = DatasetGenerator(data_raw_val, labels_raw_val,vocabulary_val, sequence_length_val, set_name='val')
val_loader = torch.utils.data.DataLoader(dataset=data_val, batch_size=8,
                                           shuffle=True, num_workers=0, pin_memory=True)


global best_acc1
best_acc1 = 0

# Train and val
for epoch in tqdm(range(0, epochs)):
    print('开始epoch:', epoch)
    acc_t1, kappa_t = train_step(train_loader, model, criterion, optimizer)

    # evaluate on validation set
    acc_v1, kappa_v = validate_step(val_loader, model, criterion)
    writer.add_scalars('data/acc', {'train': acc_t1, 'val': acc_v1}, epoch)
    writer.add_scalars('data/kappa', {'train': kappa_t, 'val': kappa_v}, epoch)
    is_best = acc_v1 > best_acc1
    if is_best:
        epo = epoch
    best_acc1 = max(acc_v1, best_acc1)

    save_checkpoint_step(
        {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        },
        is_best,
        filename='/home/yyang2/data/yyang2/Data/char_cnn/checkpoint_jd_2_over.pth.tar',
        filebest = '/home/yyang2/data/yyang2/Data/char_cnn/model_best_jd_2_over.pth.tar'
    )
    print('epoch:',epo,'模型最好的acc：', best_acc1)










