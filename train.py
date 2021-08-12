import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--lambda_2', type=float, default=1, help='scale loss2 (residual)')
parser.add_argument('--lambda_3', type=float, default=0.01, help='scale loss3 (mean)')
parser.add_argument('--lambda_4', type=float, default=1, help='scale loss4 (mae)')
parser.add_argument('--freeze', type=int, default=0, help='freeze before classification')
parser.add_argument('--start', type=int, default=0, help='the starting branch')
parser.add_argument('--end', type=int, default=12, help='the ending branch (exclusive)')
args = parser.parse_args()

#python train.py --device cuda:0 --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --epoch 100 --expid 5  --save ./experiment/metr/metr21 > ./experiment/metr/train-21.log



def main():

    # set which branch(es) to train
    # e.g. start = 1, end = 2 means train only the 10 min branch
    # end is **exclusive**
    start = args.start
    end = args.end

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # create save folder
    Path(args.save).mkdir(parents=True, exist_ok=True)

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)

    # load model weights 2.09
    # x = torch.load('experiment/metr/24_epoch_97_2.09.pth')
    # x = torch.load('experiment/metr/g5/metr_exp5_best_2.72.pth')
    # del x['end_conv_2.weight']
    # del x['end_conv_2.bias']
    # del x['end_conv_3.weight']
    # del x['end_conv_3.bias']
    # engine.model.load_state_dict(x, strict=False)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    lambda_2 = args.lambda_2
    lambda_3 = args.lambda_3
    lambda_4 = args.lambda_4
    if args.freeze == 0:
        freeze = False
    else:
        freeze = True

    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = [[]]
        train_mape = [[]]
        train_rmse = [[]]

        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1,3)[:, 0, :, :] # TODO confirm this with collaborator; same for validation
            trainy = torch.unsqueeze(trainy, 1)
            trainy = torch.unsqueeze(trainy, -1)
            for i_0 in range(start, end, 1):
                metrics = engine.train(trainx, trainy[:,:,:,i_0,:], lambda_2, lambda_3, lambda_4, freeze, i_0) # i-1 = 0, 1, ... 11
                if len(train_loss) == i_0:
                    train_loss.append([])
                if len(train_mape) == i_0:
                    train_mape.append([])
                if len(train_rmse) == i_0:
                    train_rmse.append([])
                train_loss[i_0].append(metrics[0])
                train_mape[i_0].append(metrics[1])
                train_rmse[i_0].append(metrics[2])

                if iter % args.print_every == 0:
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, {:02d} min'
                    print(log.format(iter, train_loss[i_0][-1], train_mape[i_0][-1], train_rmse[i_0][-1], (i_0+1)*5), flush=True)
        t2 = time.time()
        train_time.append(t2-t1)

        # update lr -- follow 1912.07390v1
        for param_group in engine.optimizer.param_groups:
            param_group['lr'] *= 0.97 

        #validation
        valid_loss = [[]]
        valid_mape = [[]]
        valid_rmse = [[]]


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1,3)[:, 0, :, :]
            testy = torch.unsqueeze(testy, 1)
            testy = torch.unsqueeze(testy, -1)
            for j in range(start, end, 1):
                metrics = engine.eval(testx, testy[:,:,:,j,:], j)
                if len(valid_loss) == j:
                    valid_loss.append([])
                if len(valid_mape) == j:
                    valid_mape.append([])
                if len(valid_rmse) == j:
                    valid_rmse.append([])
                valid_loss[j].append(metrics[0])
                valid_mape[j].append(metrics[1])
                valid_rmse[j].append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        for k in range(start,end,1):
            mtrain_loss = np.mean(train_loss[k])
            mtrain_mape = np.mean(train_mape[k])
            mtrain_rmse = np.mean(train_rmse[k])

            mvalid_loss = np.mean(valid_loss[k])
            mvalid_mape = np.mean(valid_mape[k])
            mvalid_rmse = np.mean(valid_rmse[k])
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Time: {:.4f}/epoch {:02d} min'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1), (k+1)*5), flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth") # save in each folder
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))



'''
    print("Training finished")  #testing
    for i in range(5):
        bestid = np.argmin(his_loss)
        pre_test = torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth")
        engine.model.load_state_dict(pre_test, strict=False)
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))

        outputs = []
        amae = []
        amape = []
        armse = []
        # realy = torch.Tensor(dataloader['y_test']).to(device)
        # realy = realy.transpose(1, 3)[:, 0, :, 0]
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)[:, 0, :, 0]
            testy = torch.unsqueeze(testy, 1)
            testy = torch.unsqueeze(testy, -1)
            with torch.no_grad():
                classify= engine.model(testx)
                preds= make_predict(classify)
                preds = dataloader['scaler'].inverse_transform(preds)
            metrics = util.metric(preds,testy)
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(1, len(amae), np.mean(amape), np.mean(armse)))


        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(i)+str(round(his_loss[bestid],2))+".pth")
        his_loss.remove(his_loss[bestid])
'''

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
