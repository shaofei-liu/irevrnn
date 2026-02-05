# -*-coding:gbk-*-

import os
import argparse
import torch
import torchvision
import copy
from torchvision.transforms import ToTensor, Compose, Lambda
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import shutil
from irevrnn_mnist_model import MnistIRevRNNPlain, MnistIRevRNNResNet


def data_transform(image):
    image = torch.reshape(image, (image.size(0), -1)).cuda()
    image -= 0.5
    image *= 2
    return image


def data_preprocessing(data_transform, batch_size):
    data_transform = Compose(
        [
            ToTensor(),
            Lambda(data_transform)
        ]
    )
    training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                              transform=data_transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=data_transform)
    num_all_data = training_set.data.size(0)
    training_eval_split = 0.95
    num_training_data = int(num_all_data * training_eval_split)
    mnist_training_set = Subset(training_set, list(range(0, num_training_data)))
    mnist_eval_set = Subset(training_set, list(range(num_training_data, num_all_data)))
    mnist_test_set = test_set
    training_dataloader = DataLoader(mnist_training_set, batch_size, shuffle=True)
    eval_dataloader = DataLoader(mnist_eval_set, batch_size)
    test_dataloader = DataLoader(mnist_test_set, batch_size)
    if args.permutation:
        mnist_permutation = torch.randperm(training_set.data.size(1) * training_set.data.size(2))
    else:
        mnist_permutation = None
    return training_dataloader, eval_dataloader, test_dataloader, mnist_permutation


def mnist_train():
    mnist_model.train()
    total_accuracy = 0
    total_loss = 0
    iteration_accuracy = 0
    counter = 0
    for iteration, data in enumerate(training_dataloader, 0):
        training_data, training_targets = data
        training_data = training_data.permute(2, 0, 1)
        if args.permutation:
            training_data = training_data[args.mnist_permutation].view(training_data.size())
        training_targets = training_targets.cuda()
        mnist_model.zero_grad()
        # next line from indRNN
        for name, param in mnist_model.named_parameters():
            if 'weights' in name or 'factor' in name:
                param.data.clamp_(-1.002, 1.002)
        training_output = mnist_model(training_data)
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
        nn.utils.clip_grad_norm_(mnist_model.parameters(), 20)
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


def mnist_eval(test: bool = False):
    with torch.no_grad():
        mnist_model.eval()
        total_accuracy = 0
        counter = 0
        if test:
            for iteration, data in enumerate(test_dataloader, 0):
                test_data, test_targets = data
                test_data = test_data.permute(2, 0, 1)
                if args.permutation:
                    test_data = test_data[args.mnist_permutation].view(test_data.size())
                test_targets = test_targets.cuda()
                test_output = mnist_model(test_data)
                prediction = test_output.data.max(1)[1]
                accuracy = prediction.eq(test_targets.data).sum()
                accuracy = accuracy / (test_targets.size(0) + 0.0)
                total_accuracy += accuracy
                counter += 1
            test_accuracy = total_accuracy.item() / (counter + 0.0)
            return test_accuracy
        else:
            for iteration, data in enumerate(eval_dataloader, 0):
                eval_data, eval_targets = data
                eval_data = eval_data.permute(2, 0, 1)
                if args.permutation:
                    eval_data = eval_data[args.mnist_permutation].view(eval_data.size())
                eval_targets = eval_targets.cuda()
                eval_output = mnist_model(eval_data)
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
        '--permutation', default=False, action='store_true',
        help='learn on permuted or original sequential MNIST dataset')
    args.add_argument(
        '--model', type=str, default='plain', help='choose from plain or resnet')
    args.add_argument(
        '--num_training_epochs', type=int, default=2000, help='maximal number of training epochs')
    args.add_argument(
        '--num_layers', type=int, default=6, help='number of network layers')
    args.add_argument(
        '--rev_len', type=int, default=3, help='pre-specified reversible residual block length')
    args.add_argument(
        '--batch_size', type=int, default=32, help='size of each (training) batch')
    args.add_argument(
        '--output_size', type=int, default=10, help='size of an output neuron at each sequence step')
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
    input_size = 1
    if args.model == 'plain':
        mnist_model = MnistIRevRNNPlain(args.num_layers, input_size, args.output_size, args.hidden_size,
                                            args.rev_len, args.ind_act_typ_str, args.res_act_typ_str).cuda()
    elif args.model == 'resnet':
        mnist_model = MnistIRevRNNResNet(args.num_layers, input_size, args.output_size, 
                                             args.hidden_size, args.rev_len, args.ind_act_typ_str, 
                                             args.res_act_typ_str).cuda()
    training_dataloader, eval_dataloader, test_dataloader, args.mnist_permutation \
        = data_preprocessing(data_transform, args.batch_size)
    criterion = nn.CrossEntropyLoss()

    params = list(mnist_model.parameters()) + list(criterion.parameters())
    total_params = 0

    param_decay = []
    param_nodecay = []

    for name, param in mnist_model.named_parameters():
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

    # optimizer = torch.optim.AdamW(mnist_model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

    optimizer = torch.optim.Adam([{'params': param_nodecay},
                                   {'params': param_decay, 'weight_decay': 0.0001}],
                                  lr=args.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=100, min_lr=0, eps=1e-9, verbose=True)
    num_training_epochs = args.num_training_epochs
    best_accuracy = 0
    current_time = time.time()
    path = '/mnist/model_' + args.model + '/num_layers_' + str(args.num_layers) + '/rev_len_' \
           + str(args.rev_len) + '/hidden_size_' + str(args.hidden_size) + '/seed_' + str(args.seed) \
           + '/ind_act_typ_str_' + str(args.ind_act_typ_str) + '/res_act_typ_str_' + str(args.res_act_typ_str) \
           + '/permutation_' + str(args.permutation)
    writer = SummaryWriter(args.results + path)
    epoch = 0
    if args.reset and os.path.exists(args.checkpoints + path):
        shutil.rmtree(args.checkpoints + path)
    if os.path.exists(args.checkpoints + path + '/mnist_model.pth'):
        checkpoints = torch.load(args.checkpoints + path + '/mnist_model.pth')
        mnist_model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        epoch = checkpoints['epoch']
        best_accuracy = checkpoints['best_accuracy']
    for i in range(epoch, num_training_epochs):
        print("epoch: ", epoch + 1)
        epoch += 1
        total_loss = mnist_train()
        print("training loss for the epoch: ", total_loss)
        writer.add_scalar('total training loss', total_loss, epoch)
        eval_accuracy = mnist_eval(test=False)
        print("evaluation accuracy: ", eval_accuracy)
        writer.add_scalar('evaluation accuracy', eval_accuracy, epoch)
        if eval_accuracy > best_accuracy:
            model_clone = copy.deepcopy(mnist_model.state_dict())
            optimizer_clone = copy.deepcopy(optimizer.state_dict())
            best_accuracy = eval_accuracy
        scheduler.step(eval_accuracy)
        print('Time difference', time.time() - current_time)
        current_time = time.time()
        if not os.path.exists(args.checkpoints + path):
            os.makedirs(args.checkpoints + path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': mnist_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
        }, args.checkpoints + path + '/mnist_model.pth')
        if (optimizer.param_groups[0]['lr'] < 1e-6):
            break

    mnist_model.load_state_dict(model_clone)
    # for name, param in mnist_model.named_parameters():
    #     print('name', name)
    #     print('param size', param.size())
    #     if param.requires_grad and '_weights' in name:
    #         print(name, param.data)
    #     if param.requires_grad and 'cell_factor' in name:
    #         print(name, param.data)
    test_accuracy = mnist_eval(test=True)
    print("test accuracy: ", test_accuracy)
    print('max memory:', torch.cuda.max_memory_allocated())
    writer.flush()
    shutil.rmtree(args.checkpoints + path)
