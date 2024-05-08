import numpy as np
import matplotlib.pyplot as plt

def imshow(img, 
        #    mean, 
        #    std,
           ):

    # Unnormalize the images
    # img = img * std[:, None, None] + mean[:, None, None]

    # Convert torch tensor to numpy array
    npimg = img.numpy()

    # Transpose the image array to match matplotlib format
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # Display the image
    # plt.show()