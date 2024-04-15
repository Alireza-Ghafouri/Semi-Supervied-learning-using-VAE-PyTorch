import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def train(net, dataloader, EPOCHS, optimizer, scheduler, device, PATH, alpha, beta, gamma):
    
    epoch_losses=[np.nan]
    net.to(device)
    for epoch in range(EPOCHS): 
        net.train()
        running_loss = 0.0
        epoch_loss=0.0
        for i, data in tqdm(enumerate(dataloader, 0), desc="Training Progress"):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            recon_images, logits = net(inputs)
            loss = net.loss(recon_images, inputs, logits, labels, alpha, beta, gamma)
            loss.backward()
            optimizer.step()


            # print statistics
            running_loss += loss.item() 
            epoch_loss += loss.item() 

            if i % 1000 == 999:    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] running loss: {running_loss / 1000:.1f}')
                running_loss = 0.0
        epoch_loss /= len(dataloader) * dataloader.batch_size
        epoch_losses.append(epoch_loss) 
        print(f'Epoch {epoch + 1} loss: {epoch_loss:.3f}')
        scheduler.step()

    print()
    print('Training Finished...\n')
    
    torch.save(net.state_dict(), PATH)
    print('Model weights saved at: ', PATH)
    
    return epoch_losses


def plot_losses(loss_values, PATH):
    plt.figure(figsize=(16,6))
    plt.plot(loss_values,color='red')
    plt.xticks(np.arange(0, len(loss_values), step=1))
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch loss values during training')
    plt.savefig(PATH)
    print('Plot saved at: ',PATH)


def evaluation(net, test_dataloader, device):
    
    correct = 0
    total = 0
    
    net.to(device)
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader,desc='Testing Progress'):
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            _, logits = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(test_dataloader)} test images: {100 * correct // total} %')


