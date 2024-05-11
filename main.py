import torch
import os
from omegaconf import OmegaConf
from model.model import DCVAE, LatentMapper, NET
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from training.training import Trainer
from dataloader.dataset import SVHNDataset, CIFAR10Dataset
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

# full_trainset= CIFAR10Dataset(is_train= True)
# testset = CIFAR10Dataset(is_train= False)

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

vae = DCVAE(image_channels= config.data.in_channels,
            image_size= config.data.image_dim,
            hidden_size= config.model.vae_hidden_dim,
            latent_size= config.model.vae_latent_dim,
            rec_weight= config.loss.reconstruction_term_weight,
            kl_weight= config.loss.kl_term_weight
          )

vae.to(device)

# vae_optimizer = Adam(vae.parameters(),
#                      lr=config.learning.learning_rate)

# vae_scheduler = ExponentialLR(optimizer = vae_optimizer,
#                               gamma = config.learning.schd_gamma)

# vae_trainer = Trainer(net= vae, 
#                       train_dataloader= full_trainloader, 
#                       test_dataloader= testloader,
#                       optimizer= vae_optimizer, 
#                       scheduler= vae_scheduler, 
#                       device= device
#                     )

# vae_trainer.train(num_epochs= config.learning.vae_num_epochs,
#                   vae_weight= 1, 
#                   cls_weight= 0, 
#                   cnt_weight= 0,
#                   save_rec_path= config.paths.rec_results,
#                   )

# vae_trainer.save_weights(path= os.path.join( config.paths.weights_root,'vae_svhn.pth' ) )

# vae.load_state_dict(torch.load(os.path.join( config.paths.weights_root,'vae_svhn.pth' )))
# print("vae weights loaded...\n")


latent_mapper = LatentMapper(y_dim = config.model.vae_latent_dim,
                             z_dim = config.model.cls_latent_dim,
                             hidden_dims = config.model.lm_hidden_dims,
                             contrastive_margin = config.loss.contrastive_margin, 
                             contrastive_similarity = config.loss.contrastive_similarity
                             )
latent_mapper.to(device)


net = NET(vae, latent_mapper, num_classes=config.data.num_classes)
net.to(device)

net_optimizer = Adam(net.parameters(),
                    lr=config.learning.learning_rate
                 )

net_scheduler = ExponentialLR(optimizer = net_optimizer,
                              gamma = config.learning.schd_gamma)

net_trainer = Trainer(net= net, 
                      train_dataloader= full_trainloader, 
                      test_dataloader= testloader,
                      optimizer= net_optimizer, 
                      scheduler= net_scheduler, 
                      device= device
                    )

net_trainer.train(num_epochs= config.learning.net_num_epochs,
                  vae_weight= config.loss.vae_term_weight, 
                  cls_weight= config.loss.classification_term_weight, 
                  cnt_weight= config.loss.contrastive_term_weight,
                  )

net_trainer.get_accuracy()

net_trainer.save_loss_plot(path = os.path.join( config.paths.report_root,'step2_try1.png' ) )

# vae_trainer.save_rec_images(path= config.paths.rec_results, filename= 0, mode='test')