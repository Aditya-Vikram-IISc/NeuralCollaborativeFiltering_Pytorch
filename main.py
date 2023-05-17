import torch
import torch.nn as nn
from config.py import config_ncf

class NeuCF(nn.Module):
    def __init__(self, config_nuemf):
        super(NeuCF, self).__init__()

        # get the arguments
        self.num_users = config_ncf["num_users"]
        self.num_items = config_ncf["num_items"]
        self.latent_vector_dim = config_ncf["latent_vector_dim"]
        self.mlp_layer_neurons = config_ncf["mlp_layer_neurons"]                      #list of layer neurons

        # MF part
        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users , embedding_dim= self.latent_vector_dim)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items , embedding_dim= self.latent_vector_dim)

        # MLP Part
        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users , embedding_dim= self.latent_vector_dim)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items , embedding_dim= self.latent_vector_dim)

        # MLP Part: Dense Neural Networks
        dnn_layers = []
        for idx, (in_size, out_size) in enumerate(zip(self.mlp_layer_neurons[:-1], self.mlp_layer_neurons[1:])):
            dnn_layers.append(nn.Linear(in_size, out_size))
            dnn_layers.append(nn.ReLU())
        self.dnn_layers_mlp = nn.Sequential(*dnn_layers)

        # NeuMF layer: Con catenation of GMF + MLP Layers
        self.NeuMF = nn.Linear(in_features= self.mlp_layer_neurons[-1]+ self.latent_vector_dim, out_features= 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # Get Embeddings for User and Item
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices) 

        # MF part
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # MLP part
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim = -1)
        mlp_vector = self.dnn_layers_mlp(mlp_vector)

        # CONCATENATE THE TWO PARTS: MF + MLP
        out = torch.cat([mf_vector, mlp_vector], dim = -1)
        out = self.NeuMF(out)
        out = self.sigmoid(out)

        return out

