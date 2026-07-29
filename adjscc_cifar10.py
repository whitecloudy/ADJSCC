import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import dataset_cifar10
from util_channel import Channel
from util_module import (
    Attention_Encoder, Attention_Decoder,
    Attention_Encoder_H, Attention_Decoder_H
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ADJSCC_Model(nn.Module):
    def __init__(self, channel_type, tcn):
        super(ADJSCC_Model, self).__init__()
        self.channel_type = channel_type
        if channel_type in ['slow_fading', 'slow_fading_eq']:
            self.encoder = Attention_Encoder_H(tcn)
            self.decoder = Attention_Decoder_H(tcn)
        else:
            self.encoder = Attention_Encoder(tcn)
            self.decoder = Attention_Decoder(tcn)
        self.channel = Channel(channel_type)

    def forward(self, inputs):
        imgs = inputs[0]
        snrdb = inputs[1]
        if self.channel_type in ['slow_fading', 'slow_fading_eq']:
            h_real = inputs[2]
            h_imag = inputs[3]
            enc = self.encoder(imgs, snrdb, h_real, h_imag)
            rv = self.channel(enc, snr_db=snrdb, h_real=h_real, h_imag=h_imag)
            dec = self.decoder(rv, snrdb, h_real, h_imag)
        elif self.channel_type == 'burst':
            b_prob = inputs[2]
            b_stddev = inputs[3]
            enc = self.encoder(imgs, snrdb)
            rv = self.channel(enc, snr_db=snrdb, b_prob=b_prob, b_stddev=b_stddev)
            dec = self.decoder(rv, snrdb)
        else:
            enc = self.encoder(imgs, snrdb)
            rv = self.channel(enc, snr_db=snrdb)
            dec = self.decoder(rv, snrdb)
        return dec * 255.0


def train(args, model):
    epoch_list = []
    loss_list = []
    val_loss_list = []
    min_loss = 10 ** 8

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.loss_dir, exist_ok=True)

    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_train) + 'to' + str(
        args.snr_up_train) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = os.path.join(args.model_dir, filename + '.pt')

    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(0, args.epochs):
        if args.channel_type == 'awgn':
            (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_range(args.snr_low_train,
                                                                                                 args.snr_up_train)
        elif args.channel_type in ['slow_fading', 'slow_fading_eq']:
            (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_range_and_h(
                args.snr_low_train, args.snr_up_train)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Train loop
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)

        epoch_loss = running_loss / train_nums

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = [inp.to(device) for inp in inputs]
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * len(targets)

        val_loss = running_val_loss / test_nums

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print('Epoch:', epoch + 1, ',loss=', epoch_loss, 'val_loss:', val_loss, 'save')
        else:
            print('Epoch:', epoch + 1, ',loss=', epoch_loss, 'val_loss:', val_loss)

        epoch_list.append(epoch)
        loss_list.append(epoch_loss)
        val_loss_list.append(val_loss)
        with open(os.path.join(args.loss_dir, filename + '.json'), mode='w') as f:
            json.dump({'epoch': epoch_list, 'loss': loss_list, 'val_loss': val_loss_list}, f)


def eval(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_eval) + 'to' + str(
        args.snr_up_eval) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = os.path.join(args.model_dir, filename + '.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, filename + '.h5')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    os.makedirs(args.eval_dir, exist_ok=True)
    criterion = nn.MSELoss()
    model.eval()

    snr_list = []
    mse_list = []
    psnr_list = []

    for snrdb in range(args.snr_low_eval, args.snr_up_eval + 1):
        imse = []
        for i in range(0, 10):
            if args.channel_type == 'awgn':
                (_, _), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(snrdb)
            elif args.channel_type in ['slow_fading', 'slow_fading_eq']:
                (_, _), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_and_h(snrdb)

            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
            running_mse = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = [inp.to(device) for inp in inputs]
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_mse += loss.item() * len(targets)

            mse = running_mse / test_nums
            imse.append(mse)

        mse = float(np.mean(imse))
        psnr = 10 * np.log10(255 ** 2 / mse)
        snr_list.append(snrdb)
        mse_list.append(mse)
        psnr_list.append(psnr)
        with open(os.path.join(args.eval_dir, filename + '.json'), mode='w') as f:
            json.dump({'snr': snr_list, 'mse': mse_list, 'psnr': psnr_list}, f)


def eval_burst(args):
    model = ADJSCC_Model(channel_type='burst', tcn=args.transmit_channel_num).to(device)
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_eval) + 'to' + str(
        args.snr_up_eval) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = os.path.join(args.model_dir, filename + '.pt')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    os.makedirs(args.eval_dir, exist_ok=True)
    criterion = nn.MSELoss()
    model.eval()

    prob_list = []
    psnr_list = []
    for b_prob in np.arange(0, 0.225, 0.025):
        imse = []
        for i in range(0, 10):
            test_ds, test_nums = dataset_cifar10.get_test_dataset_burst(args.burst_snr_eval, b_prob,
                                                                         args.burst_standard_derivation)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
            running_mse = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = [inp.to(device) for inp in inputs]
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_mse += loss.item() * len(targets)
            mse = running_mse / test_nums
            imse.append(mse)

        mse = float(np.mean(imse))
        psnr = 10 * np.log10(255 ** 2 / mse)
        prob_list.append(float(b_prob))
        psnr_list.append(psnr)
        with open(os.path.join(args.eval_dir, 'snr_evaldb' + str(args.snr_low_eval) + '_burst_sigma' + str(
                args.burst_standard_derivation) + filename + '.json'), mode='w') as f:
            json.dump({'prob': prob_list, 'psnr': psnr_list}, f)


def main(args):
    model = ADJSCC_Model(args.channel_type, args.transmit_channel_num).to(device)
    print(model)

    if args.command == 'train':
        train(args, model)
    elif args.command == 'eval':
        eval(args, model)
    elif args.command == 'eval_burst':
        eval_burst(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help='train/eval/eval_burst')
    parser.add_argument("-ct", '--channel_type', help="awgn/slow_fading/slow_fading_eq/burst")
    parser.add_argument("-md", '--model_dir', help="dir for model", default='model/')
    parser.add_argument("-lmp", '--load_model_path', help="model path for loading")
    parser.add_argument("-bs", "--batch_size", help="Batch size for training", default=128, type=int)
    parser.add_argument("-e", "--epochs", help="epochs for training", default=1280, type=int)
    parser.add_argument("-lr", "--learning_rate", help="learning_rate for training", default=0.0001, type=float)
    parser.add_argument("-tcn", "--transmit_channel_num", help="transmit_channel_num for djscc model", default=16,
                        type=int)
    parser.add_argument("-snr_low_train", "--snr_low_train", help="snr_low for training", default=0, type=int)
    parser.add_argument("-snr_up_train", "--snr_up_train", help="snr_up for training", default=20, type=int)
    parser.add_argument("-snr_low_eval", "--snr_low_eval", help="snr_low for evaluation", default=0, type=int)
    parser.add_argument("-snr_up_eval", "--snr_up_eval", help="snr_up for evaluation", default=20, type=int)
    parser.add_argument("-ldd", "--loss_dir", help="loss_dir for training", default='loss/')
    parser.add_argument("-ed", "--eval_dir", help="eval_dir", default='eval/')
    parser.add_argument("-b_snr_eval", "--burst_snr_eval", help="snr_eval for eval_burst", default=10, type=int)
    parser.add_argument("-b_stddev", "--burst_standard_derivation", help="burst_standard_derivation for eval_burst",
                        default=0., type=float)
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
