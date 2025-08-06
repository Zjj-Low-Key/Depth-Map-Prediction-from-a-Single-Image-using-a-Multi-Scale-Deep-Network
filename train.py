import os
import time
import argparse
from tracemalloc import start
import mindspore
# mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
# print(mindspore.__version__)
# print(mindspore.get_context("device_target"))
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Generator
from data import NYUDataset
from model import coarseNet, fineNet

from loss import custom_loss_function, scale_invariant
from metric import threeshold_percentage, rmse_linear, rmse_log, abs_relative_difference, squared_relative_difference
parser = argparse.ArgumentParser(description='PyTorch depth map prediction example')
parser.add_argument('--batch-size', type = int, default = 32,
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default = 50,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed (default: 1)')
args = parser.parse_args()

generator = Generator()
generator.manual_seed(args.seed)

train_dataset = ds.GeneratorDataset(NYUDataset('train'), column_names=['image', 'depth', 'valid_mask'], shuffle=True, num_parallel_workers=1)
train_dataset = train_dataset.batch(args.batch_size, drop_remainder=False)
train_dataloader = train_dataset.create_dict_iterator()

test_dataset = ds.GeneratorDataset(NYUDataset('test'), column_names=['t_image', 't_depth', 't_valid_mask'], shuffle=False, num_parallel_workers=1)
test_dataset = test_dataset.batch(1, drop_remainder=False)
test_dataloader = test_dataset.create_dict_iterator()

coarse = coarseNet()
fine = fineNet()


coarse_optimizer = nn.optim.SGD([{'params': coarse.conv1.trainable_params(), 'lr': 0.001},
                                 {'params': coarse.conv2.trainable_params(), 'lr': 0.001},
                                 {'params': coarse.conv3.trainable_params(), 'lr': 0.001},
                                 {'params': coarse.conv4.trainable_params(), 'lr': 0.001},
                                 {'params': coarse.conv5.trainable_params(), 'lr': 0.001},
                                 {'params': coarse.fc1.trainable_params(), 'lr': 0.1},
                                 {'params': coarse.fc2.trainable_params(), 'lr': 0.1}], learning_rate = 0.001, momentum = 0.9, loss_scale = 5)
fine_optimizer = nn.optim.SGD([{'params': fine.conv1.trainable_params(), 'lr': 0.001},
                               {'params': fine.conv2.trainable_params(), 'lr': 0.01},
                               {'params': fine.conv3.trainable_params(), 'lr': 0.001}], learning_rate = 0.001, momentum = 0.9, loss_scale = 5)
# coarse_optimizer = nn.optim.Adam(coarse.trainable_params(), learning_rate = 0.0001)
# fine_optimizer = nn.optim.Adam(fine.trainable_params(), learning_rate = 0.00001)

log_path = "/data/local_userdata/zhujiajun/mindspore/logs/train_log_paper.txt"


def train_coarse(epoch):
    
    def foward_coarse(rgb, depth, valid_mask):
        output = coarse(rgb)
        loss = custom_loss_function(output, depth, valid_mask)
        return loss, output
    
    coarse.set_train()
    train_coarse_loss = 0
    grad_fn = mindspore.value_and_grad(foward_coarse, None, coarse_optimizer.parameters,has_aux=True)
    start = time.time()
    for batch_idx, batch in enumerate(train_dataloader):
        # variable
        rgb, depth, valid_mask = batch['image'], batch['depth'], batch['valid_mask']
        if (ops.sum(valid_mask, (1, 2, 3)) == 0).any():
            print("No valid mask in this batch, skip it")
            continue
        
        (loss, _),grads = grad_fn(rgb, depth, valid_mask)
        if ops.any(ops.isnan(loss)):
            print("Loss is nan, skip this batch")
            continue
        coarse_optimizer(grads)
        train_coarse_loss += loss.item()
        print(f"Coarse Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
    end = time.time()
    train_coarse_loss /= (batch_idx + 1)
    with open(log_path, 'a') as f:
        f.write(f"Coarse Epoch {epoch}_training_time: {end - start:.4f} loss: {train_coarse_loss:.4f}\n")
    
    return train_coarse_loss

def train_fine(epoch):
    
    coarse.set_train(False)
    fine.set_train()
    train_fine_loss = 0
    
    def foward_fine(rgb, depth, valid_mask):
        coarse_output = coarse(rgb)
        output = fine(rgb,coarse_output)
        loss = custom_loss_function(output, depth, valid_mask)
        return loss, output
    
    grad_fn = mindspore.value_and_grad(foward_fine, None, fine_optimizer.parameters,has_aux=True)
    start = time.time()
    for batch_idx, batch in enumerate(train_dataloader):
        # variable
        rgb, depth, valid_mask = batch['image'], batch['depth'], batch['valid_mask']
        if (ops.sum(valid_mask, (1, 2, 3)) == 0).any():
            print("No valid mask in this batch, skip it")
            continue
        
        (loss, _),grads = grad_fn(rgb, depth, valid_mask)
        if ops.any(ops.isnan(loss)):
            print("Loss is nan, skip this batch")
            continue
        fine_optimizer(grads)
        train_fine_loss += loss.item()
        print(f"Fine Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
    end = time.time()
    train_fine_loss /= (batch_idx + 1)
    with open(log_path, 'a') as f:
        f.write(f"Fine Epoch {epoch}_training_time: {end - start:.4f} loss: {train_fine_loss:.4f}\n")
    return train_fine_loss

def coarse_validation(epoch):
    coarse.set_train(False)
    coarse_validation_loss = 0
    scale_invariant_loss = 0
    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0

    for batch_idx, batch in enumerate(test_dataloader):
        # variable
        rgb, depth, valid_mask= batch['t_image'], batch['t_depth'], batch['t_valid_mask']
        coarse_output = coarse(rgb)
        coarse_validation_loss += custom_loss_function(coarse_output, depth, valid_mask).item()
        # all error functions
        scale_invariant_loss += scale_invariant(coarse_output, depth, valid_mask)
        delta1_accuracy += threeshold_percentage(coarse_output, depth, valid_mask, 1.25)
        delta2_accuracy += threeshold_percentage(coarse_output, depth, valid_mask, 1.25*1.25)
        delta3_accuracy += threeshold_percentage(coarse_output, depth, valid_mask, 1.25*1.25*1.25)
        rmse_linear_loss += rmse_linear(coarse_output, depth, valid_mask)
        rmse_log_loss += rmse_log(coarse_output, depth, valid_mask)
        abs_relative_difference_loss += abs_relative_difference(coarse_output, depth, valid_mask)
        squared_relative_difference_loss += squared_relative_difference(coarse_output, depth, valid_mask)

    coarse_validation_loss /= (batch_idx + 1)
    delta1_accuracy /= (batch_idx + 1)
    delta2_accuracy /= (batch_idx + 1)
    delta3_accuracy /= (batch_idx + 1)
    rmse_linear_loss /= (batch_idx + 1)
    rmse_log_loss /= (batch_idx + 1)
    abs_relative_difference_loss /= (batch_idx + 1)
    squared_relative_difference_loss /= (batch_idx + 1)
    with open(log_path, 'a') as f:
        f.write(f"coarse validation loss_Epoch {epoch}: loss: {coarse_validation_loss:.4f}\n")
        f.write(f'a1: {delta1_accuracy.item():.4f} a2: {delta2_accuracy.item():.4f} a3: {delta3_accuracy.item():.4f}\n')
        f.write(f'rmse_linear: {rmse_linear_loss.item():.4f} rmse_log: {rmse_log_loss.item():.4f}\n')
        f.write(f'abs_relative_difference: {abs_relative_difference_loss.item():.4f} squared_relative_difference: {squared_relative_difference_loss.item():.4f}\n')
    return delta1_accuracy.item()
        
def fine_validation(epoch):
    fine.set_train(False)
    fine_validation_loss = 0
    scale_invariant_loss = 0
    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0
    

    for batch_idx, batch in enumerate(test_dataloader):
        # variable
        rgb, depth, valid_mask= batch['t_image'], batch['t_depth'], batch['t_valid_mask']
        coarse_output = coarse(rgb)
        fine_output = fine(rgb, coarse_output)
        fine_validation_loss += custom_loss_function(fine_output, depth, valid_mask).item()
        # all error functions
        scale_invariant_loss += scale_invariant(fine_output, depth, valid_mask)
        delta1_accuracy += threeshold_percentage(fine_output, depth, valid_mask, 1.25)
        delta2_accuracy += threeshold_percentage(fine_output, depth, valid_mask, 1.25*1.25)
        delta3_accuracy += threeshold_percentage(fine_output, depth, valid_mask, 1.25*1.25*1.25)
        rmse_linear_loss += rmse_linear(fine_output, depth, valid_mask)
        rmse_log_loss += rmse_log(fine_output, depth, valid_mask)
        abs_relative_difference_loss += abs_relative_difference(fine_output, depth, valid_mask)
        squared_relative_difference_loss += squared_relative_difference(fine_output, depth, valid_mask)

    fine_validation_loss /= (batch_idx + 1)
    delta1_accuracy /= (batch_idx + 1)
    delta2_accuracy /= (batch_idx + 1)
    delta3_accuracy /= (batch_idx + 1)
    rmse_linear_loss /= (batch_idx + 1)
    rmse_log_loss /= (batch_idx + 1)
    abs_relative_difference_loss /= (batch_idx + 1)
    squared_relative_difference_loss /= (batch_idx + 1)
    with open(log_path, 'a') as f:
        f.write(f"fine validation loss_Epoch {epoch}: loss: {fine_validation_loss:.4f}\n")
        f.write(f'a1: {delta1_accuracy.item():.4f} a2: {delta2_accuracy.item():.4f} a3: {delta3_accuracy.item():.4f}\n')
        f.write(f'rmse_linear: {rmse_linear_loss.item():.4f} rmse_log: {rmse_log_loss.item():.4f}\n')
        f.write(f'abs_relative_difference: {abs_relative_difference_loss.item():.4f} squared_relative_difference: {squared_relative_difference_loss.item():.4f}\n')
    return delta1_accuracy.item()   
        

# from mindspore import load_checkpoint, load_param_into_net
# param_dict = load_checkpoint("/data/local_userdata/zhujiajun/mindspore/checkpoint/coarse_model_34.ckpt")

# load_param_into_net(coarse, param_dict)
print("********* Training the Coarse Model **************")
print("Epochs:     Train_loss  Val_loss    Delta_1     Delta_2     Delta_3    rmse_lin    rmse_log    abs_rel.  square_relative")
print("Paper Val:                          (0.618)     (0.891)     (0.969)     (0.871)     (0.283)     (0.228)     (0.223)")    
best_d1 = 0
for epoch in range(1, args.epochs + 1):
    # print("********* Training the Coarse Model **************")
    training_loss = train_coarse(epoch)
    delta1_accuracy = coarse_validation(epoch)
    if delta1_accuracy > best_d1:
        best_d1 = delta1_accuracy
        mindspore.save_checkpoint(coarse, '/data/local_userdata/zhujiajun/mindspore/checkpoint'+ "/" + 'coarse_best_model_' + str(epoch) + '.ckpt')

coarse.set_train(False)# stoping the coarse model to train.

print("********* Training the Fine Model ****************")
print("Epochs:     Train_loss  Val_loss    Delta_1     Delta_2     Delta_3    rmse_lin    rmse_log    abs_rel.  square_relative")
print("Paper Val:                          (0.611)     (0.887)     (0.971)     (0.907)     (0.285)     (0.215)     (0.212)")
best_d1 = 0
for epoch in range(1, args.epochs + 1):
    # print("********* Training the Fine Model ****************")
    training_loss = train_fine(epoch)
    delta1_accuracy = fine_validation(epoch)
    if delta1_accuracy > best_d1:
        best_d1 = delta1_accuracy
        mindspore.save_checkpoint(fine, '/data/local_userdata/zhujiajun/mindspore/checkpoint'+ "/" + 'fine_best_model_' + str(epoch) + '.ckpt')
    
