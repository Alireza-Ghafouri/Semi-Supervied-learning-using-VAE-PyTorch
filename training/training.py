import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from .utils import imshow

class Trainer:
    def __init__(self, net, train_dataloader, test_dataloader, optimizer, scheduler, device):

        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epoch_losses = [np.nan]
        self.recon_images = None

    def train(self, num_epochs, alpha, beta, gamma):
        
        self.epoch_losses = [np.nan]
        self.net.to(self.device)
        for epoch in range(num_epochs): 
            self.net.train()
            running_loss = 0.0
            epoch_loss=0.0
            for i, data in tqdm(enumerate(self.train_dataloader, 0), desc="Training Progress"):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                if beta==0 and gamma==0:    # VAE model
                    self.recon_images = self.net(inputs)
                    loss = self.net.loss(self.recon_images, inputs)
                else:   # full model
                    self.recon_images, logits = self.net(inputs)
                    loss = self.net.loss(self.recon_images, inputs, logits, labels, alpha, beta, gamma)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item() 
                epoch_loss += loss.item() 

                if i % 1000 == 999:    # print every 1000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] running loss: {running_loss / 1000:.1f}')
                    running_loss = 0.0
            epoch_loss /= len(self.train_dataloader) * self.train_dataloader.batch_size
            self.epoch_losses.append(epoch_loss) 
            print(f'Epoch {epoch + 1} loss: {epoch_loss:.3f}')
            self.scheduler.step()

        print()
        print('Training Finished...\n')
            
        
    def save_weights(self, PATH):
        torch.save(self.net.state_dict(), PATH)
        print('Model weights saved at: ', PATH)

    def save_loss_plot(self, PATH):
        plt.figure(figsize=(16,6))
        plt.plot(self.epoch_losses,color='red')
        plt.xticks(np.arange(0, len(self.epoch_losses), step=1))
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch loss values during training')
        plt.savefig(PATH)
        print('Plot saved at: ',PATH)
        plt.close()

    def show_rec_images(self, mean, std, show_gt=False):
        dataiter = iter(self.test_dataloader)
        images, labels = next(dataiter)

        self.net.to('cpu')
        rec_images = self.net(images)
        imshow(make_grid(rec_images), mean, std)

        if show_gt:
            imshow(make_grid(images), mean, std)
        plt.close()

    def get_accuracy(self):
        
        correct = 0
        total = 0
        
        self.net.to(self.device)
        self.net.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for images, labels in tqdm(self.test_dataloader,desc='Testing Progress'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # calculate outputs by running images through the network
                _, logits = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {len(self.test_dataloader)} test images: {100 * correct // total} %')


