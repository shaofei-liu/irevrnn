# -*-coding:gbk-*-

import os
import argparse
import torch
import torchvision
import copy
from torchvision.transforms import ToTensor, Compose, Lambda
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import shutil
import numpy as np
from action.action_datareader import DataReader, BatchThread
from irevrnn_mnist_action_model import MnistActionIRevRNNPlain, MnistActionIRevRNNResNet


def action_train():
    action_model.train()
    total_accuracy = 0
    total_loss = 0
    iteration_accuracy = 0
    counter = 0
    for i in range(training_batch):
        training_data, training_targets = training_dataloader.get_batch()
        training_data = torch.from_numpy(training_data).cuda()
        training_data = torch.transpose(training_data, 1, 0, 2, 3)
        training_targets = torch.from_numpy(training_targets).cuda()
        training_targets -= 1
        seq_len, batch_size, num_joints, _ = training_data.size()
        training_data = training_data.view(seq_len, batch_size, 3 * num_joints)
        action_model.zero_grad()
        # next line from indRNN
        for name, param in action_model.named_parameters():
            if 'weights' in name or 'factor' in name:
                param.data.clamp_(-1.002, 1.002)
        training_output = action_model(training_data)
        loss = criterion(training_output, training_targets)
        total_loss += loss
        prediction = training_output.data.max(1)[1]
        accuracy = prediction.eq(training_targets.data).sum()
        accuracy = accuracy / (training_targets.size(0) + 0.0)
        loss.backward()
        # next line from indRNN
        # for param in mnist_model.parameters():
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-10, 10)
        nn.utils.clip_grad_norm_(action_model.parameters(), 20)
        optimizer.step()
        total_accuracy += accuracy
        iteration_accuracy += accuracy
        counter += 1
        if (counter % 100 == 0):
            print("iteration", counter - 99, "to", counter, ":")
            print("training accuracy: ", iteration_accuracy.item() / 100.0)
            iteration_accuracy = 0
    print("training accuracy for the epoch: ", total_accuracy.item() / (counter + 0.0))
    return total_loss.item() / (counter + 0.0)


def action_eval(test: bool = False):
    with torch.no_grad():
        action_model.eval()
        total_accuracy = 0
        counter = 0
        if test:
            for iteration in range(20 * test_batch - 1):
                test_data, test_targets = test_dataloader.get_batch()
                test_data = torch.from_numpy(test_data).cuda()
                test_data = torch.transpose(test_data, 1, 0, 2, 3)
                test_targets = torch.from_numpy(test_targets).cuda()
                test_targets -= 1
                seq_len, batch_size, num_joints, _ = test_data.size()
                test_data = test_data.view(seq_len, batch_size, 3 * num_joints)
                test_output = action_model(test_data)
                prediction = test_output.data.max(1)[1]
                accuracy = prediction.eq(test_targets.data).sum()
                accuracy = accuracy / (test_targets.size(0) + 0.0)
                total_accuracy += accuracy
                counter += 1
            test_accuracy = total_accuracy.item() / (counter + 0.0)
            return test_accuracy
        else:
            for iteration in range(5 * eval_batch - 1):
                eval_data, eval_targets = eval_dataloader.get_batch()
                eval_data = torch.from_numpy(eval_data).cuda()
                eval_data = torch.transpose(eval_data, 1, 0, 2, 3)
                eval_targets = torch.from_numpy(eval_targets).cuda()
                eval_targets -= 1
                seq_len, batch_size, num_joints, _ = eval_data.size()
                eval_data = eval_data.view(seq_len, batch_size, 3 * num_joints)
                eval_output = action_model(eval_data)
                prediction = eval_output.data.max(1)[1]
                accuracy = prediction.eq(eval_targets.data).sum()
                accuracy = accuracy / (eval_targets.size(0) + 0.0)
                total_accuracy += accuracy
                counter += 1
            eval_accuracy = total_accuracy.item() / (counter + 0.0)
            return eval_accuracy


def get_args():
    args = argparse.ArgumentParser(
        description='Independent Reversible Recurrent Neural Network')
    args.add_argument(
        '--results',
        default='./results',
        help='the folder to store results')
    args.add_argument(
        '--checkpoints',
        default='./checkpoints',
        help='the folder to store checkpoints')
    args.add_argument(
        '--model_path',
        default='./torch_irevrnn',
        help='the folder for the model')
    args.add_argument(
        '--model', type=str, default='plain', help='choose from plain or resnet')
    args.add_argument(
        '--num_training_epochs', type=int, default=2000, help='maximal number of training epochs')
    args.add_argument(
        '--num_layers', type=int, default=6, help='number of network layers')
    args.add_argument(
        '--seq_len', type=int, default=4, help='pre-specified sequence length')
    args.add_argument(
        '--rev_len', type=int, default=3, help='pre-specified reversible residual block length')
    args.add_argument(
        '--batch_size', type=int, default=32, help='size of each (training) batch')
    args.add_argument(
        '--hidden_size', type=int, default=128, help='number of hidden neurons at each sequence step')
    args.add_argument(
        '--learning_rate', type=float, default=2e-4, help='initial learning rate')
    args.add_argument(
        '--ind_act_typ_str', type=str, default='relu', help='type of independent activation functions')
    args.add_argument(
        '--res_act_typ_str', type=str, default='tanh', help='type of reversible residual activation functions (if any)')
    args.add_argument(
        '--seed', type=int, default=100, help='random seed for pytorch')
    args.add_argument(
        '--reset', default=False, action='store_true',
        help='whether to delete the stored model and restart training')
    return args


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    print('args: ', args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    model_path = args.model_path
    input_size = 150
    output_size = 60
    if args.model == 'plain':
        action_model = MnistActionIRevRNNPlain(args.num_layers, input_size, output_size, args.hidden_size,
                                               args.rev_len, args.ind_act_typ_str, args.res_act_typ_str).cuda()
    elif args.model == 'resnet':
        action_model = MnistActionIRevRNNResNet(args.num_layers, input_size, output_size, args.hidden_size,
                                                args.rev_len, args.ind_act_typ_str, args.res_act_typ_str).cuda()

    training_dataloader = DataReader(args.batch_size, args.seq_len, training='train')
    eval_dataloader = DataReader(args.batch_size, args.seq_len, training='eval')
    test_dataloader = DataReader(args.batch_size, args.seq_len, training='eval')

    training_batch = int(np.ceil(training_dataloader.get_data_size() / (args.batch_size + 0.0)))
    eval_batch = int(np.ceil(eval_dataloader.get_data_size() / (args.batch_size + 0.0)))
    test_batch = int(np.ceil(test_dataloader.get_data_size() / (args.batch_size + 0.0)))

    criterion = nn.CrossEntropyLoss()

    params = list(action_model.parameters()) + list(criterion.parameters())
    total_params = 0

    param_decay = []
    param_nodecay = []

    for name, param in action_model.named_parameters():
        # print('name', name)
        # print('param size', param.size())
        if len(param.size()) > 2:
            total_params += param.size()[0] * param.size()[1] * param.size()[2]
        elif len(param.size()) > 1:
            total_params += param.size()[0] * param.size()[1]
        else:
            total_params += param.size()[0]
        if 'weights' in name or 'bias' in name:
            param_nodecay.append(param)
            # print('parameters no weight decay: ',name)
        else:
            param_decay.append(param)
            # print('parameters with weight decay: ',name)
        # print('total_params here', total_params)
    print('Model total parameters:', total_params)

    # optimizer = torch.optim.AdamW(action_model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

    optimizer = torch.optim.Adam([{'params': param_nodecay},
                                  {'params': param_decay, 'weight_decay': 0.0001}],
                                 lr=args.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=100, min_lr=0, eps=1e-9, verbose=True)
    num_training_epochs = args.num_training_epochs
    best_accuracy = 0
    current_time = time.time()
    path = '/action/model_' + args.model + '/num_layers_' + str(args.num_layers) + '/rev_len_' \
           + str(args.rev_len) + '/hidden_size_' + str(args.hidden_size) + '/seed_' + str(args.seed) \
           + '/ind_act_typ_str_' + str(args.ind_act_typ_str) + '/res_act_typ_str_' + str(args.res_act_typ_str)
    writer = SummaryWriter(args.results + path)
    epoch = 0
    if args.reset and os.path.exists(args.checkpoints + path):
        shutil.rmtree(args.checkpoints + path)
    if os.path.exists(args.checkpoints + path + '/action_model.pth'):
        checkpoints = torch.load(args.checkpoints + path + '/action_model.pth')
        action_model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        epoch = checkpoints['epoch']
        best_accuracy = checkpoints['best_accuracy']
    for i in range(epoch, num_training_epochs):
        print("epoch: ", epoch + 1)
        epoch += 1
        total_loss = action_train()
        print("training loss for the epoch: ", total_loss)
        writer.add_scalar('total training loss', total_loss, epoch)
        eval_accuracy = action_eval(test=False)
        print("evaluation accuracy: ", eval_accuracy)
        writer.add_scalar('evaluation accuracy', eval_accuracy, epoch)
        if eval_accuracy > best_accuracy:
            model_clone = copy.deepcopy(action_model.state_dict())
            optimizer_clone = copy.deepcopy(optimizer.state_dict())
            best_accuracy = eval_accuracy
        scheduler.step(eval_accuracy)
        print('Time difference', time.time() - current_time)
        current_time = time.time()
        if not os.path.exists(args.checkpoints + path):
            os.makedirs(args.checkpoints + path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': action_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
        }, args.checkpoints + path + '/action_model.pth')
        if (optimizer.param_groups[0]['lr'] < 1e-6):
            break

    action_model.load_state_dict(model_clone)
    # for name, param in action_model.named_parameters():
    #     print('name', name)
    #     print('param size', param.size())
    #     if param.requires_grad and '_weights' in name:
    #         print(name, param.data)
    #     if param.requires_grad and 'cell_factor' in name:
    #         print(name, param.data)
    test_accuracy = action_eval(test=True)
    print("test accuracy: ", test_accuracy)
    print('max memory:', torch.cuda.max_memory_allocated())
    writer.flush()
    shutil.rmtree(args.checkpoints + path)
