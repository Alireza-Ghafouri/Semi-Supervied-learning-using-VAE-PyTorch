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
    
    def loss(self, reconstructed_images, images, mean, log_var):
        ce = F.binary_cross_entropy(reconstructed_images, images, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = self.rec_weight * ce + self.kl_weight* kld
        num_pixels = images.size(0) * images.size(1) * images.size(2) * images.size(3)
        return loss / num_pixels

class LatentMapper(nn.Module):
    def __init__(self, y_dim, z_dim, hidden_dims, contrastive_temperature):
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
        self.temperature = contrastive_temperature
        
    def forward(self, y):
        self.z = self.fc_layers(y)
        return self.z
    
    def contrastive_loss(self, embeddings, indexes):
        """
        embeddings: Tensor of shape [batch_size * num_augmentations, embedding_dim]
        num_augmentations: Number of augmentations per image
        temperature: Temperature scaling factor
        """
        # batch_size = embeddings.size(0) // num_augmentations
        embeddings = F.normalize(embeddings, dim=1).to('cpu')
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        similarity_matrix.to('cpu')
        # Create labels
        # labels = torch.arange(batch_size).repeat_interleave(num_augmentations).to(embeddings.device)

        # Create mask to ignore self-similarity
        mask = torch.eye(indexes.shape[0], dtype=torch.bool).to('cpu')
        indexes = (indexes.unsqueeze(0) == indexes.unsqueeze(1)).float().to('cpu')
        indexes = indexes * ~mask  # Zero out the diagonal
        indexes = indexes / indexes.sum(dim=1, keepdim=True)  # Normalize
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, indexes, reduction='sum') / (embeddings.size(0))
        
        return loss
    
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
        return x_hat, mean, log_var, z, logits
    
    def loss(self, x_hat, x, mean, log_var, logits, labels, embeddings, indexes, vae_weight, cls_weight, cnt_weight):

        vae_loss = self.vae.loss(x_hat, x, mean, log_var) if vae_weight!=0 else 0
        contrastive_loss = self.latent_mapper.contrastive_loss(embeddings=embeddings, indexes=indexes) if cnt_weight!=0 else 0      
        classification_loss = nn.CrossEntropyLoss()(logits, labels) if cls_weight!=0 else 0
        
        return  vae_weight * vae_loss + cls_weight * classification_loss + cnt_weight * contrastive_loss