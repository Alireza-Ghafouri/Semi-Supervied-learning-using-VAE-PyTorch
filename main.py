import torch
import os
from omegaconf import OmegaConf
from model.model import VAE, LatentMapper, NET
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from training.training import Trainer
from dataloader.dataset import SVHNDataset
from dataloader.utils import split_data
from torch.utils.data import DataLoader


config = OmegaConf.load("config\config.yaml")

os.makedirs(config.paths.weights_root, exist_ok = True)
os.makedirs(config.paths.report_root, exist_ok = True)
os.makedirs(config.paths.rec_results, exist_ok = True)


device = torch.device(config.learning.device)
# print('Training is on: ', device)

full_trainset = SVHNDataset(mode='train')
testset = SVHNDataset(mode='test')

labeled_trainset, unlabeled_trainset = split_data(full_trainset = full_trainset,
                                                  labeled_ratio = config.data.labeled_ratio)


testloader = DataLoader(dataset = testset, 
                        batch_size = config.learning.batch_size, 
                        shuffle = True)

full_trainloader = DataLoader(dataset = full_trainset, 
                              batch_size = config.learning.batch_size, 
                              shuffle = True)

labeled_trainloader = DataLoader(dataset = labeled_trainset,
                                 batch_size = config.learning.batch_size, 
                                 shuffle = True)

unlabeled_trainloader = DataLoader(dataset = unlabeled_trainset, 
                                   batch_size = config.learning.batch_size, 
                                   shuffle = True)

vae = VAE(z_dim = config.model.vae_latent_dim,
          rec_weight= config.loss.reconstruction_term_weight,
          kl_weight= config.loss.kl_term_weight
          )

vae.to(device)


latent_mapper = LatentMapper(y_dim = config.model.vae_latent_dim,
                             z_dim = config.model.cls_latent_dim,
                             hidden_dims = config.model.lm_hidden_dims,
                             contrastive_margin = config.loss.contrastive_margin, 
                             contrastive_similarity = config.loss.contrastive_similarity
                             )
latent_mapper.to(device)


net = NET(vae, latent_mapper, num_classes=config.data.num_classes)
net.to(device)

optimizer = Adam(net.parameters())
scheduler = ExponentialLR(optimizer = optimizer,
                          gamma = config.learning.schd_gamma)

vae_trainer = Trainer(net= net, 
                      train_dataloader= labeled_trainloader, 
                      test_dataloader= testloader,
                      optimizer= optimizer, 
                      scheduler= scheduler, 
                      device= device
                    )

vae_trainer.load_weights(path= os.path.join( config.paths.weights_root,'vae_net.pth' ), full_net= True)
vae_trainer.train(num_epochs= config.learning.num_epochs,
                  vae_weight= config.loss.vae_term_weight, 
                  cls_weight= config.loss.classification_term_weight, 
                  cnt_weight= config.loss.contrastive_term_weight)

# vae_trainer.save_weights(path= os.path.join( config.paths.weights_root,'vae_net.pth' ) )

# vae_trainer.save_loss_plot(path = os.path.join( config.paths.report_root,'test.png' ) )

# vae_trainer.save_rec_images(mean= full_trainset.mean,
#                             std= full_trainset.std,
#                             path=config.paths.rec_results,
#                             show_imgs=True
#                             )