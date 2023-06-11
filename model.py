import torch
import random
import torch.nn as nn
from config import config_ncf

class NeuCF(nn.Module):
    def __init__(self, config_nuemf):
        super(NeuCF, self).__init__()

        # get the arguments
        self.num_users = config_nuemf["num_users"]
        self.num_items = config_nuemf["num_items"]
        self.mlp_latent_vector_dim = config_nuemf["mlp_latent_vector_dim"]
        self.gmf_latent_vector_dim = config_nuemf["gmf_latent_vector_dim"]
        self.mlp_layer_neurons = config_nuemf["mlp_layer_neurons"]         #list of layer neurons

        # GMF part
        self.embedding_user_gmf = nn.Embedding(num_embeddings=self.num_users , embedding_dim= self.gmf_latent_vector_dim)
        self.embedding_item_gmf = nn.Embedding(num_embeddings=self.num_items , embedding_dim= self.gmf_latent_vector_dim)

        # MLP Part
        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users , embedding_dim= self.mlp_latent_vector_dim)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items , embedding_dim= self.mlp_latent_vector_dim)

        # MLP Part: Dense Neural Networks
        dnn_layers = []
        for idx, (in_size, out_size) in enumerate(zip(self.mlp_layer_neurons[:-1], self.mlp_layer_neurons[1:])):
            dnn_layers.append(nn.Linear(in_size, out_size))
            dnn_layers.append(nn.ReLU())
        self.dnn_layers_mlp = nn.Sequential(*dnn_layers)

        # NeuMF layer: Con catenation of GMF + MLP Layers
        self.NeuMFlayer = nn.Linear(in_features= self.mlp_layer_neurons[-1]+ self.gmf_latent_vector_dim, out_features= 1)

    def forward(self, user_indices, item_indices):
        # Get Embeddings for User and Item
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_gmf = self.embedding_user_gmf(user_indices)
        item_embedding_gmf = self.embedding_item_gmf(item_indices) 

        # MF part
        gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)

        # MLP part
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim = -1)
        mlp_vector = self.dnn_layers_mlp(mlp_vector)

        # CONCATENATE THE TWO PARTS: MF + MLP
        out = torch.cat([gmf_vector, mlp_vector], dim = -1)
        out = self.NeuMFlayer(out)

        return out.view(-1)
    


if __name__ == "__main__":
    model = NeuCF(config_nuemf = config_ncf)
    # get random user and item ids
    uid = random.randint(0, config_ncf["num_users"]-1)
    movid = random.randint(0, config_ncf["num_items"]-1)
    output = model(torch.tensor(uid), torch.tensor(movid))
    print(output)

