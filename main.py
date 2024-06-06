import torch
import os
from omegaconf import OmegaConf
from model.model import DCVAE, LatentMapper, NET
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from training.training import Trainer
from dataloader.datasets import SVHNDataset, CIFAR10Dataset, MyDataset
from dataloader.utils import split_data, create_pseudo_labeled_dataset, selective_collate
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from dataloader.transforms import svhn_transform_base, svhn_transform_1, svhn_transform_2, svhn_transform_3, svhn_transform_4, svhn_transform_5

config = OmegaConf.load(os.path.join('config' , 'config.yaml'))

os.makedirs(config.paths.weights_root, exist_ok = True)
os.makedirs(config.paths.report_root, exist_ok = True)
os.makedirs(config.paths.rec_results, exist_ok = True)


device = torch.device(config.learning.device)
# print('Training is on: ', device)

full_trainset = SVHNDataset(mode='train')
testset = SVHNDataset(mode='test', transform= svhn_transform_base)

# full_trainset= CIFAR10Dataset(is_train= True)
# testset = CIFAR10Dataset(is_train= False)

svhn_transforms=[svhn_transform_base, svhn_transform_1, svhn_transform_2, svhn_transform_3, svhn_transform_4, svhn_transform_5]

labeled_trainset, unlabeled_trainset = split_data(full_trainset = full_trainset,
                                                  labeled_ratio = config.data.labeled_ratio)

testloader = DataLoader(dataset = testset, 
                        batch_size = config.learning.batch_size, 
                        shuffle = True,
                        )

full_trainloader = DataLoader(dataset = full_trainset, 
                              batch_size = config.learning.batch_size, 
                              shuffle = False,
                              collate_fn= selective_collate,
                              )

labeled_trainloader = DataLoader(dataset = labeled_trainset,
                                 batch_size = config.learning.batch_size, 
                                 shuffle = True,
                                 collate_fn= selective_collate,
                                 )

unlabeled_trainloader = DataLoader(dataset = unlabeled_trainset, 
                                   batch_size = config.learning.batch_size, 
                                   shuffle = True,
                                   collate_fn= selective_collate,
                                   )

#`````````````````````````````````````Phase 1: Train/Load VAE:````````````````````````````````````` 

vae = DCVAE(image_channels= config.data.in_channels,
            image_size= config.data.image_dim,
            hidden_size= config.model.vae_hidden_dim,
            latent_size= config.model.vae_latent_dim,
            rec_weight= config.loss.reconstruction_term_weight,
            kl_weight= config.loss.kl_term_weight
          )

vae.to(device)

# vae_optimizer = Adam(vae.parameters(),
#                      lr=config.learning.vae.learning_rate)

# vae_scheduler = ExponentialLR(optimizer = vae_optimizer,
#                               gamma = config.learning.vae.schd_gamma)

# vae_trainer = Trainer(net= vae, 
#                       train_dataloader= full_trainloader, 
#                       test_dataloader= testloader,
#                       optimizer= vae_optimizer, 
#                       scheduler= vae_scheduler, 
#                       device= device
#                     )

# vae_trainer.train(num_epochs= config.learning.vae.num_epochs,
#                   vae_weight= 1, 
#                   cls_weight= 0, 
#                   cnt_weight= 0,
#                   save_rec_path= config.paths.rec_results,
#                   )

# vae_trainer.save_weights(path= os.path.join( config.paths.weights_root,'vae_svhn.pth' ) )
# vae_trainer.save_loss_plot(path = os.path.join( config.paths.report_root,'vae_svhn.png' ) )

vae.load_state_dict(torch.load(os.path.join( config.paths.weights_root,'vae_svhn.pth' )))
print("\n VAE weights loaded...\n")


#`````````````````````````````````````Phase 2: Train/Load Block2:````````````````````````````````````` 

latent_mapper = LatentMapper(y_dim = config.model.vae_latent_dim,
                             z_dim = config.model.cls_latent_dim,
                             hidden_dims = config.model.lm_hidden_dims,
                             contrastive_temperature= config.loss.contrastive_temperature
                             )
latent_mapper.to(device)

# for param in vae.parameters():
#     param.requires_grad = False

net_init = NET(vae, latent_mapper, num_classes=config.data.num_classes)
net_init.to(device)

for param in net_init.vae.parameters():
    param.requires_grad = False

# latent_mapper_optimizer = Adam(net_init.parameters(),
#                                lr=config.learning.latent_mapper.learning_rate)

# latent_mapper_scheduler = ExponentialLR(optimizer = latent_mapper_optimizer,
#                                         gamma = config.learning.latent_mapper.schd_gamma)

# latent_mapper_trainer = Trainer(net= net_init, 
#                                 train_dataloader= full_trainloader, 
#                                 test_dataloader= testloader,
#                                 optimizer= latent_mapper_optimizer, 
#                                 scheduler= latent_mapper_scheduler, 
#                                 device= device
#                                 )

# latent_mapper_trainer.train(num_epochs= config.learning.latent_mapper.num_epochs,
#                             vae_weight= 0, 
#                             cls_weight= 0, 
#                             cnt_weight= 1,
#                             transforms= svhn_transforms,
#                             )

# latent_mapper_trainer.save_weights(path= os.path.join( config.paths.weights_root,'VAE_latent_mapper_svhn.pth' ) )
# latent_mapper_trainer.save_loss_plot(path = os.path.join( config.paths.report_root,'VAE_latent_mapper_svhn.png' ) )
 
#``````````````````````````````````Phase 3: Train/Load Classifier:```````````````````````````````` 

net= NET(vae, latent_mapper, num_classes=config.data.num_classes)
net.to(device)

net_optimizer = Adam(net.parameters(),
                    lr=config.learning.net.learning_rate
                 )

net_scheduler = ExponentialLR(optimizer = net_optimizer,
                              gamma = config.learning.net.schd_gamma)

net_trainer = Trainer(net= net, 
                      train_dataloader= labeled_trainloader, 
                      test_dataloader= testloader,
                      optimizer= net_optimizer, 
                      scheduler= net_scheduler, 
                      device= device
                    )

net_trainer.train(num_epochs= config.learning.net.num_epochs,
                  vae_weight= config.loss.vae_term_weight, 
                  cls_weight= config.loss.classification_term_weight, 
                  cnt_weight= config.loss.contrastive_term_weight,
                  )

net_trainer.save_weights(path= os.path.join( config.paths.weights_root,'svhn_net_labeled.pth' ))
net_trainer.get_accuracy()

# net_trainer.save_loss_plot(path = os.path.join( config.paths.report_root,'step2_try1.png' ) )

 
#```````````````````````````````Phase 3: labelling unlabeled data:```````````````````````````````` 

pseudo_labeled_trainset = create_pseudo_labeled_dataset(net= net,
                                                        unlabeled_trainloader= unlabeled_trainloader,
                                                        mannual_dataset= MyDataset,
                                                        device= device
                                                        )

# Combine labeled and pseudo-labeled datasets
combined_trainset = ConcatDataset([labeled_trainset, pseudo_labeled_trainset])

# Create DataLoader for combined dataset
combined_trainloader = DataLoader(dataset = combined_trainset, 
                                  batch_size = config.learning.batch_size, 
                                  shuffle = True
                                  )

net2 = NET(vae, latent_mapper, num_classes=config.data.num_classes)
net2.to(device)
net2.load_state_dict(torch.load(os.path.join( config.paths.weights_root,'svhn_net_labeled.pth' )))


net2_optimizer = Adam(net2.parameters(),
                      lr=config.learning.net.learning_rate
                      )

net2_scheduler = ExponentialLR(optimizer = net2_optimizer,
                                gamma = config.learning.net.schd_gamma)


net_trainer2 = Trainer(net= net2, 
                       train_dataloader= combined_trainloader, 
                       test_dataloader= testloader,
                       optimizer= net2_optimizer, 
                       scheduler= net2_scheduler, 
                       device= device,
                      )

net_trainer2.train(num_epochs= 30,
                   vae_weight= config.loss.vae_term_weight, 
                   cls_weight= config.loss.classification_term_weight, 
                   cnt_weight= config.loss.contrastive_term_weight,
                   )

net_trainer2.get_accuracy()
