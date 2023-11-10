import argparse
import torch
import torch.utils.data
from data.dataset import create_dataloader
from crowd.model import GNN
import os
from torch import nn, optim
import json
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np
import time

parser = argparse.ArgumentParser(description='Graph Mechanics Networks')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='output/logs', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='gnn', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, egnn_vel, rf_vel')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=500, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-10, metavar='N',
                    help='weight decay')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--delta_frame', type=int, default=50,
                    help='Number of frames delta.')
parser.add_argument('--mol', type=str, default='aspirin',
                    help='Name of the molecule.')
parser.add_argument('--data_dir', type=str, default='CrowdESL/spatial_graph/data',
                    help='Data directory.')
parser.add_argument('--learnable', type=eval, default=False, metavar='N',
                    help='Use learnable FK.')

parser.add_argument("--config_by_file", default=False, action="store_true", )


args = parser.parse_args()
if args.config_by_file:
    job_param_path = 'configs/simple_config_md17.json'
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        args.exp_name = hyper_params["exp_name"]
        args.batch_size = hyper_params["batch_size"]
        args.epochs = hyper_params["epochs"]
        args.no_cuda = hyper_params["no_cuda"]
        args.seed = hyper_params["seed"]
        args.lr = hyper_params["lr"]
        args.nf = hyper_params["nf"]
        args.model = hyper_params["model"]
        args.attention = hyper_params["attention"]
        args.n_layers = hyper_params["n_layers"]
        args.max_training_samples = hyper_params["max_training_samples"]
        args.data_dir = hyper_params["data_dir"]
        args.weight_decay = hyper_params["weight_decay"]
        args.norm_diff = hyper_params["norm_diff"]
        args.tanh = hyper_params["tanh"]
        args.learnable = hyper_params["learnable"]

        args.delta_frame = hyper_params["delta_frame"]
        args.mol = hyper_params["mol"]

args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda:0" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

# torch.autograd.set_detect_anomaly(True)


def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # loader_train = create_dataloader(data_dir=args.data_dir, partition='train', batch_size=args.batch_size,
    #                                      shuffle=True,  num_workers=8)

    # loader_val = create_dataloader(data_dir=args.data_dir, partition='valid', batch_size=args.batch_size,
    #                                      shuffle=False,  num_workers=8)

    # loader_test = create_dataloader(data_dir=args.data_dir, partition='test', batch_size=args.batch_size,
                                        #  shuffle=False,  num_workers=8)

    loader_train, loader_val, loader_test = create_dataloader(data_dir=args.data_dir, partition='train', batch_size=args.batch_size,
                                         shuffle=True,  num_workers=8)
    
    rotate_90 = torch.FloatTensor([[0, 1], [-1, 0]])
    rotate_120 = torch.FloatTensor([[-0.5, -0.866], [0.866, -0.5]])
    reflect_x = torch.FloatTensor([[-1, 0], [0, 1]])
    reflect_y = torch.FloatTensor([[1, 0], [0, -1]])
    
    #group = [torch.eye(2), rotate_90, torch.mm(rotate_90, rotate_90), torch.mm(rotate_90, torch.mm(rotate_90, rotate_90))]
    #group = [torch.eye(2), reflect_x] ###BIA Tjunc mouthhole
    #group = [torch.eye(2), torch.mm(rotate_90, reflect_x)] ###BIA Tjunc mouthhole
    group = [torch.eye(2), rotate_120, torch.mm(rotate_120, rotate_120)]
    group = [op.to(device) for op in group]

    model = GNN(input_dim=6, hidden_nf=args.nf, group=group, n_layers=args.n_layers, device=device, recurrent=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8

    for epoch in range(args.epochs):

        train_loss = train(model, optimizer, epoch, loader_train)


        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:

            val_loss = train(model, optimizer, epoch, loader_val, partition='valid', backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, partition='test', backprop=False)

            results['epochs'].append(epoch)
            results['loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), args.outf + '/' + 'saved_model.pth')
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best apoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))
            
            if epoch - best_epoch > 100:
                break


        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)
    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, partition='train', backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'loss_stick': 0, 'loss_vel': 0, 'reg_loss': 0}

    for batch_idx, data in enumerate(loader):
        
        # data = [d.to(device) for d in data]
        # data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs

        # for d in data:
        #     print(d.shape)

        ped = data.ped
        loc, vel, loc_end = data.pos.to(device), data.x.to(device), data.y.to(device)
        node_type = data.node_attr.to(device)
        edges = data.edge_index.to(device)
        # edges = [edges[0].to(device), edges[1].to(device)]
        # print(loc.shape, vel.shape)
        # ddd
        batch_size = loc.shape[0]

        optimizer.zero_grad()

        # helper to compute reg loss
        reg_loss = 0

        rows, cols = edges
        edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)
        # nodes = torch.cat([loc, vel], dim=1)
        nodes = torch.cat([loc, vel, F.one_hot(node_type)], dim=1)
        loc_pred = model(nodes, edges, edge_attr)

        # compute regularization loss
        # print(loc_pred.shape, loc_end.shape)
        # ddd
        loc_pred = loc_pred[torch.where(node_type==0)]

        # if partition=="test":
        #     loss = loss_mse(loc_pred[:, 2:4], loc_end)
        # else:
        #     loss = loss_mse(loc_pred[:, :2], loc_end)

        loss = loss_mse(loc_pred[:, :2], loc_end)


        res['loss'] += loss.item()*batch_size

        if backprop:
            # if epoch % 1 == 0:
            aug_loc_end = []
            for i in range(1, len(model.group)):
                g = model.group[i]
                aug_loc_end.append(torch.mm(loc_end, g))

            
            aug_loc_end = torch.cat(aug_loc_end, dim=1)
            # print("aug_loc_end",aug_loc_end.shape)
            # print("loc_pred_shape",loc_pred.shape)
            reg_loss = loss_mse(loc_pred[:, 2:], aug_loc_end)
            loss += 0.6 * reg_loss

            loss.backward()
            optimizer.step()
        try:
            res['reg_loss'] += reg_loss.item()*batch_size
        except:  # no reg loss (no sticks and hinges)
            pass
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f reg loss: %.5f'
          % (prefix+partition, epoch,
             res['loss'] / res['counter'], res['reg_loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)





