import torch
import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from .utils import imshow, apply_transformations

class Trainer:
    def __init__(self, net, train_dataloader, test_dataloader, optimizer, scheduler, device):

        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epoch_losses = [np.nan]

    def train(self, num_epochs, vae_weight, cls_weight, cnt_weight, transforms, save_rec_path=None):
        
        self.epoch_losses = [np.nan]
        self.net.to(self.device)
        for epoch in range(num_epochs): 
            self.net.train()
            running_loss = 0.0
            epoch_loss=0.0
            for i, data in tqdm(enumerate(self.train_dataloader, 0), desc="Training Progress"):
                
                inputs, labels, indexes = apply_transformations(data, transforms_list= transforms)
                inputs= inputs.to(self.device)
                labels= labels.to(self.device)
                indexes= indexes.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                if cls_weight==0 and cnt_weight==0:    # VAE model
                    reconstructed_images, mean, log_var = self.net(inputs)
                    loss= self.net.loss(reconstructed_images, inputs, mean, log_var)
                else:   # full model
                    recon_images, mean, log_var, z, logits = self.net(inputs)
                    loss = self.net.loss(recon_images, inputs, mean, log_var, logits, labels, z, indexes, vae_weight, cls_weight, cnt_weight)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item() 
                epoch_loss += loss.item() 

                if i % 100 == 99:    # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] running loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            epoch_loss /= len(self.train_dataloader)

            self.epoch_losses.append(epoch_loss) 
            print(f'Epoch {epoch + 1}/{num_epochs} loss: {epoch_loss:.5f}')
            self.scheduler.step()
            if save_rec_path is not None:
                self.save_rec_images(path= save_rec_path, filename= epoch+1, mode='train')
                self.save_rec_images(path= save_rec_path, filename= epoch+1, mode='test' )
                self.net.to(self.device)

        print()
        print('Training Finished...\n')

    def save_weights(self, path):
        torch.save(self.net.state_dict(), path)
        print('Model weights saved at: ', path)

    def save_loss_plot(self, path):
        plt.figure(figsize=(16,6))
        plt.plot(self.epoch_losses,color='red')
        plt.xticks(np.arange(0, len(self.epoch_losses), step=1))
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch loss values during training')
        plt.savefig(path)
        print('Plot saved at: ',path)
        plt.close()

    def save_rec_images(self, path, filename, mode, show_imgs=False):
        
        torch.manual_seed(42)

        if mode=='train':
            dataloader= self.train_dataloader
        elif mode=='test':
            dataloader= self.test_dataloader
        else:
            raise NotImplementedError("Wrong mode selected!")

        dataiter = iter(dataloader)
        images, labels = next(dataiter)

        self.net.to('cpu')
        self.net.eval()
        rec_images = self.net(images)[0]

        if filename <= 1:
            sample = images[:40]
            imshow(make_grid(sample))
            plt.savefig( os.path.join( path, mode + '_input_images.png' ) )

        imshow(make_grid(rec_images[:40]) )
        plt.savefig( os.path.join( path, mode + str(filename) + '.png' ) )

        if show_imgs:
            plt.show()
        
        plt.close()

    def get_accuracy(self):

        torch.manual_seed(42)

        correct = 0
        total = 0
        self.net.to(self.device)
        self.net.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for images, labels, _ in tqdm(self.test_dataloader,desc='Testing Progress'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # calculate outputs by running images through the network
                logits = self.net(images)[-1]
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network: {correct} / {total} = {round(100 * correct / total,2)} %')


