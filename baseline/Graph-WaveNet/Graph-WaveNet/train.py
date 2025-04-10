import os
import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
parser.add_argument('--data', type=str, default='data/processed/NYCBike2_part1_graphwavenet.npz', help='data path')
parser.add_argument('--adjdata', type=str, default='data/processed/NYCBike2_part1_adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='sequence length')
parser.add_argument('--nhid', type=int, default=32, help='number of hidden units')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension (will be overridden by dataset)')
parser.add_argument('--num_nodes', type=int, default=170, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--print_every', type=int, default=50, help='print frequency')
parser.add_argument('--save', type=str, default='', help='save path (will be auto-generated)')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()

def main():
    # Automatically create garage directory for the specific dataset
    dataset_name = os.path.splitext(os.path.basename(args.data))[0]
    save_dir = os.path.join('garage', dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Update save path to use the auto-generated directory
    args.save = os.path.join(save_dir, f'{dataset_name}_')

    # Set up logging
    log_file = os.path.join(save_dir, f'{dataset_name}_exp{args.expid}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    # Override in_dim based on the dataset
    args.in_dim = dataloader['x_train'].shape[-1]  # Use the last dimension of x_train as in_dim
    args.num_nodes = dataloader['x_train'].shape[1]  # Update num_nodes based on dataset

    logging.info(str(args))

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)

    engine.model.to(device)

    logging.info("start training...")
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)  # [batch_size, seq_length, num_nodes, in_dim]
            trainx = trainx.transpose(1, 3)      # [batch_size, in_dim, num_nodes, seq_length]
            trainx = trainx.transpose(2, 3)      # [batch_size, in_dim, seq_length, num_nodes]
            trainy = torch.Tensor(y).to(device)  # [batch_size, seq_length, num_nodes, 1]
            trainy = trainy.transpose(1, 3)      # [batch_size, 1, num_nodes, seq_length]
            trainy = trainy.transpose(2, 3)      # [batch_size, 1, seq_length, num_nodes]
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mae.append(metrics[0])  # MAE is the same as loss in this case
            train_mape.append(metrics[1] * 100)  # Convert MAPE to percentage
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}%, Train RMSE: {:.4f}'
                logging.info(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]))
        t2 = time.time()
        train_time.append(t2 - t1)

        # Validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testx = testx.transpose(2, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            testy = testy.transpose(2, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[0])  # MAE is the same as loss
            valid_mape.append(metrics[1] * 100)  # Convert MAPE to percentage
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        logging.info(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}%, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}%, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        logging.info(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)))
        torch.save(engine.model.state_dict(), args.save + f"epoch{i}_" + str(round(mvalid_loss, 2)) + ".pth")

    logging.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logging.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # Testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + f"epoch{bestid + 1}_" + str(round(his_loss[bestid], 2)) + ".pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)
    realy = realy.transpose(2, 3)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testx = testx.transpose(2, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
            outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    logging.info("Training finished")
    logging.info("The valid loss on best model is " + str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, 0, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}%, Test RMSE: {:.4f}'
        logging.info(log.format(i + 1, metrics[0], metrics[1] * 100, metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1] * 100)
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}%, Test RMSE: {:.4f}'
    logging.info(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save + f"_exp{args.expid}best" + str(round(his_loss[bestid], 2)) + ".pth")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    logging.info("Total time spent: {:.4f}".format(t2 - t1))