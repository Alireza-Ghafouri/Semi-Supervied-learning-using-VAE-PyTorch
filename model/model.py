import torch
from torch import nn
from .utils import Flatten, UnFlatten
import torch.nn.functional as F

class DCVAE(nn.Module):
    def __init__(self, image_channels, image_size, hidden_size, latent_size, rec_weight, kl_weight):
        super(DCVAE, self).__init__()

        self.rec_weight = rec_weight
        self.kl_weight = kl_weight

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2),
            nn.LeakyReLU(0.2),
            Flatten(),
        )
        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(hidden_size, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 6, 2),
            nn.Sigmoid(),
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.encoder(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        z = self.sample(log_var, mean)
        x = self.fc(z)
        x = self.decoder(x)

        return x, mean, log_var
    
    def loss(self, reconstructed_image, images, mean, log_var):
        CE = F.binary_cross_entropy(reconstructed_image, images, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # print('CE Loss', CE , "KL Loss:", KLD)
        loss = self.rec_weight * CE + self.kl_weight* KLD
        return loss


class LatentMapper(nn.Module):
    def __init__(self, y_dim, z_dim, hidden_dims, contrastive_margin, contrastive_similarity):
        super(LatentMapper, self).__init__()
        layers = []
        input_dim = y_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, z_dim))
        self.fc_layers = nn.Sequential(*layers)
        self.z_dim = z_dim
        self.z = torch.tensor([])
        self.margin = contrastive_margin
        self.similarity = contrastive_similarity
        
    def forward(self, y):
        self.z = self.fc_layers(y)
        return self.z

    def contrastive_loss(self, labels):
        # Compute pairwise similarity matrix
        if self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(self.z.unsqueeze(1), self.z.unsqueeze(0), dim=-1)
        else:
            raise NotImplementedError("Other similarity metrics not implemented yet")

        # Create mask for positive and negative pairs
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        mask.fill_diagonal_(0)  # Exclude diagonal elements (self-similarity)

        # Compute contrastive loss
        positive_pairs = similarity_matrix * mask
        negative_pairs = similarity_matrix * (1 - mask)
#         print('SIMILARITY MATRIX:\n',similarity_matrix)
#         print('MASK\n',mask)
        loss_contrastive = torch.mean((1 - positive_pairs) + torch.clamp(negative_pairs - self.margin, min=0))

        return loss_contrastive
    
class NET(nn.Module):
    def __init__(self, vae, latent_mapper, num_classes):
        super(NET, self).__init__()
        self.vae = vae
        self.latent_mapper = latent_mapper
        self.classifier = nn.Linear(self.latent_mapper.z_dim, num_classes) 
        
    def forward(self, x):
        x_hat, mean, log_var = self.vae(x)
        y = self.vae.sample(log_var, mean)
        # x = self.vae.fc(y)
        z = self.latent_mapper(y)
        logits  = self.classifier(z)
        return x_hat, mean, log_var, logits
    
    def loss(self, x_hat, x, mean, log_var, logits, labels, vae_weight, cls_weight, cnt_weight):
        
        vae_loss = self.vae.loss(x_hat, x, mean, log_var)
        if cls_weight==0 and cnt_weight==0:
            return vae_weight * vae_loss
        classification_loss = nn.CrossEntropyLoss()(logits, labels)
        contrastive_loss = self.latent_mapper.contrastive_loss(labels)            
        # print('VAE LOSS: ',vae_loss ,'\nCLS LOSS: ', classification_loss,'\nCNT LOSS: ', contrastive_loss)
        return  vae_weight * vae_loss + cls_weight * classification_loss + cnt_weight * contrastive_loss