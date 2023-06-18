# get libraries
import torch
import torch.nn as nn
import os
import argparse
import torch.optim as optim
# get modules
from config import config_ncf
from evaluate import evaluate_metrics
from dataset import NCFTraining_Dataset, NCFTesting_Dataset
from model import NeuCF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_test_dataloaders(ratings_train_pos_path:str, ratings_train_neg_path:str, ratings_test_path:str):
    
    # load the datasets
    train_dataset = NCFTraining_Dataset(ratings_train_pos_path = ratings_train_pos_path, ratings_train_neg_path = ratings_train_neg_path)
    test_dataset = NCFTesting_Dataset(ratings_test_path = ratings_test_path)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size= 100, shuffle= False, num_workers= 0)
    return train_dataloader, test_dataloader

def train(model, loss_criterion, optimizer, train_dataloader, test_dataloader, epoch, device):
    writer = SummaryWriter()  # for visualization
    
    # transfer the model, and loss function to cuda, if available
    model = model.to(device)
    loss_criterion = loss_criterion.to(device)

    # train the model
    model.train()
    total_loss = 0
    for iteration, (userid, movieid, label) in enumerate(train_dataloader):
        # transfer the data to cuda, if available
        userid = userid.to(device)
        movieid = movieid.to(device)
        label = label.float().to(device)

        optimizer.zero_grad()

        # Forward Propagation
        pred_label = model(userid, movieid)
        loss = loss_criterion(pred_label, label)

        # Back Propagation
        loss.backward()
        optimizer.step()

    # add train average loss
        total_loss += userid.shape[0]*loss.item() 

    # calculate the avg_loss for the epoch
    avg_loss = total_loss/len(train_dataloader)
    writer.add_scalar("data/trainloss", avg_loss, epoch)

    # evaluate the model
    model.eval()
    HR, NDCG = evaluate_metrics(model, test_dataloader, config_ncf["top_k"])
    writer.add_scalar("data/eval_HR", HR, epoch)
    writer.add_scalar("data/eval_NDCG", NDCG, epoch)
    print("Test metrics: HR = {:.4f} and NDCG = {:.4f}".format(HR, NDCG))

    # save the weights
    if not os.path.exists("weights/"):
        os.makedirs("weights/")
    model_weight_path = f"weights/NCF_weights_{epoch}_.pth"
    torch.save(model.state_dict(), model_weight_path)
    print("Checkpoint saved to {}".format(model_weight_path))
    

if __name__ == "__main__":


    #initialize argparse
    parser = argparse.ArgumentParser()
    # add arguments to Parser
    parser.add_argument("--ratings_train_pos_path", type = str, default = "data/ratings_train_pos")
    parser.add_argument("--ratings_train_neg_path", type = str, default = "data/ratings_train_neg")
    parser.add_argument("--ratings_test_path", type = str, default= "data/ratings_test_neg")
    parser.add_argument("--epochs", type = int, default = 25)
    args = parser.parse_args()

    # generate traindataloader and testdataloader
    train_dataloader, test_dataloader = train_test_dataloaders(ratings_train_pos_path= args.ratings_train_pos_path, ratings_train_neg_path = args.ratings_train_neg_path, ratings_test_path = args.ratings_test_path)

    # create model and optimizer instance 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuCF(config_nuemf = config_ncf)
    loss_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config_ncf["lr_rate"])

    # train the model
    for epoch in range(args.epochs):
        train(model = model, loss_criterion = loss_criterion, optimizer = optimizer, \
              train_dataloader = train_dataloader, test_dataloader = test_dataloader, \
              epoch = epoch, device = device)
    