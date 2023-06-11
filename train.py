# get libraries
import torch
import torch.nn as nn
import argparse
import torch.optim as optim



# get modules
from config import config_ncf
from evaluate import evaluate_metrics
from dataset import NCFTraining_Dataset, NCFTesting_Dataset
from model import NeuCF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create an instance of the model
model = NeuCF(config_nuemf = config_ncf)
loss_criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config_ncf["lr_rate"])
print(device)
# Load the datasets

def train(args, device = device):
    # create dataset
    print("Dataset creation started")
    train_dataset = NCFTraining_Dataset(ratings_train_pos_path = args.ratings_train_pos_path, ratings_train_neg_path= args.ratings_train_neg_path)
    test_dataset = NCFTesting_Dataset(ratings_test_path = args.ratings_test_path)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size= 100, shuffle= False, num_workers= 0)
    print("Dataset created")
    
    # TODO : make batchsize in config or argsparse

    writer = SummaryWriter()  # for visualization

    if device == "cuda":
        model = model.to(device)
        loss_criterion = loss_criterion.to(device)

    







if __name__ == "__main__":


    #initialize argparse
    parser = argparse.ArgumentParser()
    # add arguments to Parser
    parser.add_argument("--ratings_train_pos_path", type = str, default = "data/ratings_train_pos")
    parser.add_argument("--ratings_train_neg_path", type = str, default = "data/ratings_train_neg")
    parser.add_argument("--ratings_test_path", type = str, default= "data/ratings_test_neg")
    args = parser.parse_args()

    train(args = args)