# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import argparse
import os
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pygame.examples.fastevents import post_them
from sklearn.metrics.pairwise import euclidean_distances

from followbot.ml.helper.history import History
from followbot.ml.helper.checkpoint import CheckPoint
from followbot.ml.helper.utils_data import BalanceBatchSampler, Reporter

import torch
import torch.nn as nn
import torch.optim as optim
import logging

K_NEIGHBOR = 3

# Manual Random Seed: 1
torch.manual_seed(1)
np.random.seed(1)

class ClassifierModel(nn.Module):
    def __init__(self, n_class):
        super(ClassifierModel, self).__init__()
        # embed velocity vectors
        vel_hidden_size = 32
        self.embed_vel = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.1),
                                       nn.Linear(16, vel_hidden_size), nn.LeakyReLU(0.1))

        # embed distance values
        dist_hidden_size = 32
        self.embed_dist = nn.Sequential(nn.Linear(K_NEIGHBOR, 16), nn.LeakyReLU(0.1),
                                        nn.Linear(16, dist_hidden_size), nn.LeakyReLU(0.1))

        # pooling
        self.pooling = lambda x: torch.mean(x, dim=0)

        # Memory Layer
        self.lstm_dim = 64
        self.hidden_dim = vel_hidden_size + dist_hidden_size
        self.lstm = nn.LSTM(self.hidden_dim, self.lstm_dim, batch_first=True)
        self.memory = torch.empty(1)

        # output layer
        self.classifier = nn.Sequential(nn.Linear(self.lstm_dim, 64), nn.LeakyReLU(0.1),
                                        nn.Linear(64, 32), nn.LeakyReLU(0.1),
                                        nn.Linear(32, n_class))
        
        self.softmax = nn.Softmax()

    def device(self):
        return next(self.parameters()).device

    def reset_memory(self):
        h0 = torch.zeros((1, 1, self.lstm_dim), device=self.device())
        c0 = torch.randn((1, 1, self.lstm_dim), device=self.device())
        self.memory = (h0, c0)

    def forward(self, sample_features):
        # outputs = []

        last_inds_of_features = np.cumsum([len(x["vel"]) for x in sample_features])
        last_inds_of_features = np.insert(last_inds_of_features, 0, 0)

        all_vel_tensors = torch.cat([x["vel"] for x in sample_features])
        all_dist_tensors = torch.cat([x["dist"] for x in sample_features])
        all_h_tensors = []

        # Apply an exponential function on distance values
        all_dist_tensors = torch.exp(-all_dist_tensors/5.)

        V = self.embed_vel(all_vel_tensors)
        D = self.embed_dist(all_dist_tensors)

        for ii in range(len(sample_features)):
            v = V[last_inds_of_features[ii]:last_inds_of_features[ii + 1]]
            d = D[last_inds_of_features[ii]:last_inds_of_features[ii + 1]]

        # for sample in sample_features:
        #     v = self.embed_vel(sample["vel"])
        #     d = self.embed_dist(sample["dist"])

            v_ = self.pooling(v)
            d_ = self.pooling(d)
            h = torch.cat([v_, d_])  # concatenate

            # # x, self.memory = self.lstm(h.reshape((1, 1, self.hidden_dim)), self.memory)
            # # y = self.classifier(x).squeeze()
            # y = self.classifier(h).squeeze()
            # y2 = self.softmax(y)
            # outputs.append(y2)

            all_h_tensors.append(h)

        all_h_tensors = torch.stack(all_h_tensors)
        y = self.classifier(all_h_tensors).squeeze()
        y2 = self.softmax(y)

        return y2

        # return outputs


class CrowdClassifier:
    def __init__(self, device_):
        self.model = {}
        self.device = device_
        self.class_names = []

    def init_model(self, class_names):
        n_class = len(class_names)
        self.class_names = class_names
        self.model = ClassifierModel(n_class)
        self.model.to(self.device)

    def classify(self, sample_features):  # return class name
        y_hat = np.argmax(self.model([sample_features]).detach().cpu().numpy())
        return self.class_names[y_hat]

    def extract_features(self, raw_data_df):
        data_features = []
        data_frames = raw_data_df[["pos_x", "pos_y", "vel_x", "vel_y", "frame_id"]].groupby("frame_id")
        for frame_id, frame_data in data_frames:
            data_features.append({"frame_id": frame_id})
            n = len(frame_data)
            data_features[-1]["vel"] = torch.tensor(frame_data[["vel_x", "vel_y"]].to_numpy(),
                                                    device=self.device).float()

            data_pos = frame_data[["pos_x", "pos_y"]].to_numpy()
            pair_distance = euclidean_distances(data_pos)         # Euclidean distance between any pair of agents
            dists_tril = pair_distance[np.tril_indices(n, k=-1)]  # lower triangular matrix (remove redundancies)

            # data_features[-1]["dist"] = torch.tensor(dists_tril, device=self.device).float()

            # Todo: take the 3 closest neighbor
            dists_wo_diag = pair_distance[~np.eye(n, dtype=bool)].reshape(n, n - 1)
            if n - 1 > K_NEIGHBOR:
                k_min_dists = [np.partition(dists_wo_diag[i, :], K_NEIGHBOR)[:K_NEIGHBOR] for i in range(n)]
                k_min_dists = np.stack(k_min_dists)
            elif n - 1 < K_NEIGHBOR:
                k_min_dists = np.ones((n, K_NEIGHBOR), dtype=float) * 10000
                k_min_dists[:, :n-1] = dists_wo_diag
            else:  # n - 1 == K_NEIGHBOR
                k_min_dists = dists_wo_diag
            data_features[-1]["dist"] = torch.tensor(k_min_dists, device=self.device).float()

        return data_features

    def load_data(self, path):
        class_filenames = glob.glob(os.path.join(path, '*.txt'))

        raw_data = {}
        data_features = {}
        for filename in class_filenames:
            columns_name = ["frame_id", "agent_id", "pos_x", "pos_y", "vel_x", "vel_y"]

            label = os.path.basename(filename)[:-4]
            raw_data[label] = pd.read_csv(filename, sep=" ", names=columns_name, header=None)

            data_features[label] = self.extract_features(raw_data[label])
        return raw_data, data_features


def train():
    # learning checkpointer
    ckpter = CheckPoint(model=crowd_classifier.model, optimizer=optimizer, path=ckpt_path,
                        prefix=run_name, interval=1, save_num=1, loss0=loss0)
    train_hist = History(name='train_hist' + run_name)
    # validation_hist = History(name='validation_hist' + run_name)

    for epoch in range(last_epoch + 1, n_epoch):
        train_loss = 0
        for label, samples in sorted(data_features_.items()):
            target = targets[label]
            crowd_classifier.model.reset_memory()
            output = crowd_classifier.model(samples)
            loss = criterion(output, torch.stack(target))
            train_loss += loss.item()
            crowd_classifier.model.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch = [%04d]: training loss = %.4f" % (epoch, train_loss))
        train_logs = {'loss': train_loss, 'acc': 0}
        ckpter.check_on(epoch=epoch, monitor='loss', loss_acc=train_logs)
        train_hist.add(logs=train_logs, epoch=0)
        logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, train_hist.name, str(train_hist.recent)))
        # logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, val_hist.name, str(val_hist.recent)))


def test():
    for i in range(10):
        # take a random index
        category_index = np.random.randint(0, len(raw_data_))
        categ_name = all_labels[category_index]
        sample_index = np.random.randint(0, len(categ_name))
        x = data_features_[categ_name][sample_index]

        y = np.argmax(targets[categ_name][sample_index].cpu().numpy())
        y_hat = np.argmax(crowd_classifier.model([x]).detach().cpu().numpy())
        print(y, y_hat)

        frame_id = data_features_[categ_name][sample_index]["frame_id"]
        raw_categ = raw_data_[categ_name]
        raw_frame = raw_categ[raw_categ["frame_id"] == frame_id]
        n = len(raw_frame)

        plt.figure()
        pos_and_vel_t = raw_frame[["pos_x", "pos_y", "vel_x", "vel_y"]].to_numpy()
        for ped_i in pos_and_vel_t:
            p1 = ped_i[:2]
            v = ped_i[2:4]
            p2 = p1 + v / 2
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b')
        plt.scatter(pos_and_vel_t[:, 0], pos_and_vel_t[:, 1])
        plt.title("Category[gt]:%s | [predicted]:%s"
                  % (all_labels[y], all_labels[y_hat]))
        plt.show()
        dummy = 1


if __name__ == "__main__":
    # Program Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', '--rn', type=str, default='Run001',
                        help='The name for this run (default: "Run01")')
    parser.add_argument('--num_workers', '--nw', type=int, default=8,
                        help='number of workers for Dataloader (num_workers: 8)')
    parser.add_argument('--start', '--start', type=int, default=False,
                        help='Start from scratch (default: 1)')
    parser.add_argument('--n_epoch', '--n_epoch', type=int, default=100000,
                        help='Number of epochs')
    parser.add_argument('--data_path', type=str,
                        default="/home/cyrus/workspace2/ros-catkin/src/followbot/src/followbot/temp/robot_obsvs/test",
                        help='The path of crowd logs')
    parser.add_argument('--ckpt_path', type=str,
                        default="/home/cyrus/workspace2/ros-catkin/src/followbot/model-ckpnts",
                        help='The path for storing model checkpoints')

    args = parser.parse_args()
    run_name = args.run_name
    num_workers = args.num_workers
    start_from_scratch = args.start
    n_epoch = args.n_epoch
    data_path = args.data_path
    ckpt_path = args.ckpt_path

    logging.basicConfig(filename=os.path.join(ckpt_path, "crowd-classifier.log"),
                        level=getattr(logging, "INFO", None))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    crowd_classifier = CrowdClassifier(device)

    # Load Data
    raw_data_, data_features_ = crowd_classifier.load_data(data_path)
    all_labels = sorted(list(data_features_.keys()))
    n_classes = len(all_labels)

    crowd_classifier.init_model(all_labels)

    optimizer = optim.SGD(crowd_classifier.model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    targets = {}
    for ii, label in enumerate(all_labels):
        target_ii = torch.zeros(n_classes, device=device).float()
        target_ii[ii] = 1
        targets[label] = [target_ii] * len(data_features_[label])

    # load torch model
    if not start_from_scratch:
        try:
            reporter = Reporter(ckpt_root=ckpt_path, exp='', monitor='loss')
            last_model_filename = reporter.select_best(run=run_name).selected_ckpt
            last_epoch = int(reporter.last_epoch)
            loss0 = reporter.last_loss
            crowd_classifier.model.load_state_dict(torch.load(last_model_filename)['model_state_dict'])
            optimizer.load_state_dict(torch.load(last_model_filename)['optimizer_state_dict'])
        except:
            print("Warning: couldn't find/load model checkpoint")
            last_epoch = -1
            loss0 = 0
    else:
        print("start the training from scratch...")
        last_epoch = -1
        loss0 = 0

    # train()
    test()




