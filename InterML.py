import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import os
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from get_dataset import load_val_dataset, load_train_dataset_k_shot, load_test_dataset
from learner import Learner
from learner_freeze import LearnerFreeze
from TripletLoss import TripletLoss
from augment import Rotate_DA


def train_dataset_prepared(n_classes, k_shot):
    x_train, value_y_train = load_train_dataset_k_shot(n_classes, k_shot)
    x_train, value_y_train = Rotate_DA(x_train, value_y_train)

    x_val, value_y_val = load_val_dataset(n_classes)

    x_test, y_test = load_test_dataset(n_classes)

    min_value = x_train.min()
    min_in_val = x_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = x_train.max()
    max_in_val = x_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    x_train = (x_train - min_value) / (max_value - min_value)
    x_val = (x_val - min_value) / (max_value - min_value)
    x_test = (x_test - min_value) / (max_value - min_value)

    x_train = x_train.transpose(0, 2, 1)
    x_val = x_val.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)

    return x_train, x_val, value_y_train, value_y_val, x_test, y_test


def rand_bbox(size, lamb):
    length = size[2]
    cut_rate = 1.-lamb
    cut_length = np.int(length*cut_rate)
    cx = np.random.randint(length)
    bbx1 = np.clip(cx - cut_length//2, 0, length)
    bbx2 = np.clip(cx + cut_length//2, 0, length)
    return bbx1, bbx2


def train(model, loss, train_data_loader, optimizer, epoch, writer, triplet_loss):
    model.train()
    correct = 0
    result_loss = 0
    for data, target in train_data_loader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        # new edition: Triplet + Rotate + CutMix
        lam = np.random.beta(1, 1)  # beta distribution to generate cropped area
        index = torch.randperm(data.size()[0]).to(device)
        target_a, target_b = target, target[index]
        bbx1, bbx2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2] = data[index, :, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) / data.size()[-1])

        optimizer.zero_grad()

        output = model(data)  # overall output
        embedding = output[0]  # features
        logistic = output[1]  # probability

        # CE loss
        output_ce = F.log_softmax(logistic, dim=1)
        result_loss_ce = lam * loss(output_ce, target_a) + (1 - lam) * loss(output_ce, target_b)

        # Triplet loss
        result_loss_triplet = lam * triplet_loss(embedding, target_a) + (1 - lam) * triplet_loss(embedding, target_b)

        # overall loss
        result_loss = result_loss_ce + 0.01 * result_loss_triplet

        result_loss.backward()
        optimizer.step()
        result_loss += result_loss.item()
        pred = output_ce.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    result_loss /= len(train_data_loader.dataset)

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        result_loss,
        correct,
        len(train_data_loader.dataset),
        100.0 * correct / len(train_data_loader.dataset))
    )


def evaluate(model, loss, val_data_loader, epoch, writer, triplet_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_data_loader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)

            output = model(data)
            embedding = output[0]
            logistic = output[1]
            target = target.squeeze()

            # CE loss
            output_ce = F.log_softmax(logistic, dim=1)
            result_loss_ce_batch = loss(output_ce, target)

            # Triplet loss
            result_loss_triplet_batch = triplet_loss(embedding, target)

            # overall loss
            result_loss_batch = result_loss_ce_batch + 0.01 * result_loss_triplet_batch

            test_loss += result_loss_batch.item()

            pred = output_ce.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_data_loader.dataset)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(val_data_loader.dataset),
            100.0 * correct / len(val_data_loader.dataset),
        )
    )

    f1 = open('results/acc_rotate_cutmix_triplet.txt', 'a+')
    f1.write(str(100.0 * correct / len(val_data_loader.dataset)) + " " + str(epoch) + '\n')

    return test_loss


def train_and_evaluate(model, loss_function, train_data_loader, val_data_loader, optimizer, epochs, writer, save_path, triplet_loss):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_function, train_data_loader, optimizer, epoch, writer, triplet_loss)
        test_loss = evaluate(model, loss_function, val_data_loader, epoch, writer, triplet_loss)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model.state_dict(), save_path)  # torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")


def test(i, model, test_data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            output = F.log_softmax(output[1], dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    fmt = '\nTest set: Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_data_loader.dataset),
            100.0 * correct / len(test_data_loader.dataset),
        )
    )

    f3 = open('results/monte_acc_rotate_cutmix_triplet.txt', 'a+')
    f3.write(str(100.0 * correct / len(test_data_loader.dataset)) + " " + str(i) + '\n')


class Config:
    def __init__(
        self,
        batch_size: int = 16,
        test_batch_size: int = 16,
        epochs: int = 150,
        lr: float = 0.01,
        log_interval: int = 10,
        n_classes: int = 16,
        k_shot: int = 1,
        save_path: str = 'weight/monte_rotate_cutmix_triplet.pth',
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.log_interval = log_interval
        self.n_classes = n_classes
        self.k_shot = k_shot
        self.save_path = save_path


def main():
    conf = Config()
    writer = SummaryWriter("logs")

    for i in range(100):
        x_train, x_val, value_y_train, value_y_val, x_test, y_test = train_dataset_prepared(conf.n_classes, conf.k_shot)

        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(value_y_train))
        train_data_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(value_y_val))
        val_data_loader = DataLoader(val_dataset, batch_size=conf.test_batch_size, shuffle=True)

        # fine tuning
        config = [
            ('complex_conv', [64, 1, 3, 1, 0]),  # block1
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block2
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block3
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block4
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block5
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block6
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block7
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block8
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block9
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('flatten', []),
            ('linear', [1024, 1152]),
            ('relu', [False]),
            ('linear', [conf.n_classes, 1024])
        ]
        model = Learner(config)

        if torch.cuda.is_available():
            model = model.to(device)

        loss = nn.NLLLoss()
        if torch.cuda.is_available():
            loss = loss.to(device)

        triplet_loss = TripletLoss(margin=5)
        if torch.cuda.is_available():
            triplet_loss = triplet_loss.to(device)

        optim = torch.optim.Adam(model.parameters(), lr=conf.lr)

        train_and_evaluate(model, loss_function=loss, train_data_loader=train_data_loader, val_data_loader=val_data_loader,
                           optimizer=optim, epochs=conf.epochs, writer=writer, save_path=conf.save_path, triplet_loss=triplet_loss)

        f2 = open('results/acc_rotate_cutmix_triplet.txt', 'a+')
        f2.write('\n \n')

        # test
        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_data_loader = DataLoader(test_dataset, batch_size=conf.test_batch_size, shuffle=True)

        config2 = [
            ('complex_conv', [64, 1, 3, 1, 0]),  # block1
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block2
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block3
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block4
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block5
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block6
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block7
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block8
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('complex_conv', [64, 64, 3, 1, 0]),  # block9
            ('relu', [False]),
            ('bn', [128]),
            ('max_pool1d', [2, 2, 0]),
            ('flatten', []),
            ('linear', [1024, 1152]),
            ('relu', [False]),
            ('linear', [conf.n_classes, 1024])
        ]
        model2 = LearnerFreeze(config2)
        weights = torch.load(conf.save_path, map_location=device)
        model2.load_state_dict(weights)

        model2 = model2.to(device)
        test(i, model2, test_data_loader)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
