import torch
from torch import nn
from .utils import ResNet18Enc,ResNet18Dec
import torch.nn.functional as F

class VAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)
        
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.mu = None
        self.logvar = None
        self.z = None

    def forward(self, x):
        self.mu, self.logvar = self.encoder(x)
        #z = self.reparameterize(mean, logvar)

        # sample z from q
        std = torch.exp(self.logvar / 2)
        q = torch.distributions.Normal(self.mu, std)
        self.z = q.rsample()
        
        x_recon = self.decoder(self.z)   
        
        return x_recon
    
    def gaussian_likelihood(self, x_hat, x, logscale):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))
    
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def loss(self, x_hat, x):
        
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, x, self.log_scale)

        # kl
        std = torch.exp(self.logvar / 2)
        kl = self.kl_divergence(self.z, self.mu, std)

        # elbo
        #print(kl,recon_loss)
        elbo = (kl - recon_loss)
        elbo = elbo.mean()
        return elbo
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean


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
#         self.alpha = vae_loss_weight
#         self.beta = classification_loss_weight
#         self.gamma = contrastive_loss_weight
        
    def forward(self, x):
#         mu, logvar = self.vae.encode(x)
#         y = self.vae.reparameterize(mu, logvar)
        x_hat = self.vae(x)
        z = self.latent_mapper(self.vae.z)
        logits  = self.classifier(z)
        return x_hat, logits
    
    def loss(self, x_hat, x, logits, labels, alpha, beta, gamma):
        
        vae_loss = self.vae.loss(x_hat, x)
        if beta==0 and gamma==0:
            return alpha * vae_loss
        classification_loss = nn.CrossEntropyLoss()(logits, labels)
        contrastive_loss = self.latent_mapper.contrastive_loss(labels)            
#         print(vae_loss , classification_loss, contrastive_loss)
        return  alpha * vae_loss + beta * classification_loss + gamma * contrastive_loss