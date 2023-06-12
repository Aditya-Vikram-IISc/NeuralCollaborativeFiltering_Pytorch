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



# create an instance of the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuCF(config_nuemf = config_ncf)
loss_criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config_ncf["lr_rate"])

# Load the datasets
train_dataset = NCFTraining_Dataset(ratings_train_pos_path = "data/ratings_train_pos", ratings_train_neg_path = "data/ratings_train_neg")
test_dataset = NCFTesting_Dataset(ratings_test_path = "data/ratings_test_neg")
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size= 100, shuffle= False, num_workers= 0)
print("Dataset created")
# TODO : make batchsize in config or argsparse


def train(model, loss_criterion, optimizer, args, epoch, device = device):
    writer = SummaryWriter()  # for visualization

    if device == "cuda":
        model = model.to(device)
        loss_criterion = loss_criterion.to(device)

    # train the model
    model.train()
    for iteration, (userid, movieid, label) in enumerate(train_dataloader):
        if device == "cuda":
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
        writer.add_scalar("data/loss", loss.item(), iteration)

    # evaluate the model
    model.eval()
    HR, NDCG = evaluate_metrics(model, test_dataloader, config_ncf["top_k"])
    print("Test metrics: HR = {:.4f} and NDCG = {:.4f}".format(HR, NDCG))

    # save the weights
    model_weight_path = f"weights/NCF_weights_{epoch}_.pth"
    # if not os.path.exists(model_weight_path):
    #     os.makedirs(model_weight_path)
    torch.save(model.state_dict(), model_weight_path)
    print("Checkpoint saved to {}".format(model_weight_path))
    

if __name__ == "__main__":


    #initialize argparse
    parser = argparse.ArgumentParser()
    # add arguments to Parser
    parser.add_argument("--ratings_train_pos_path", type = str, default = "data/ratings_train_pos")
    parser.add_argument("--ratings_train_neg_path", type = str, default = "data/ratings_train_neg")
    parser.add_argument("--ratings_test_path", type = str, default= "data/ratings_test_neg")
    args = parser.parse_args()

    for epoch in range(10):
        train(model, loss_criterion, optimizer, args, epoch, device = device)
    