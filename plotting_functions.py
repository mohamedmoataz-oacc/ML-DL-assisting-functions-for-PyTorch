import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_random_images(image_data: torch.utils.data.Dataset, num_images = 5):
    """
    Visualizes `num_images` random images from the given dataset with their labels as titles.
    """
    if num_images > 10:
        print("Maximum number of images to visualize is 10 images.")
        num_images = 10
    
    to_visualize = np.random.randint(0, len(image_data.data), num_images)
    classes = image_data.classes

    fig, axs = plt.subplots(1, num_images)
    fig.set_size_inches(20, 5)
    c = 0
    for i in to_visualize:
        axs[c].imshow(image_data[i][0].permute(1,2,0))
        axs[c].set_title(classes[image_data.targets[i]])
        axs[c].axis(False)
        c+=1

def visualize_truth_vs_predicted_image_labels(model, image_data: torch.utils.data.Dataset, num_images = 5, model_device = 'cpu'):
    """
    Visualizes `num_images` random images from the given dataset.
    It uses the given model to predict the image label and sets the title to be the image's true vs predicted labels.
    """
    if num_images > 10:
        print("Maximum number of images to visualize is 10 images.")
        num_images = 10
    
    to_visualize = np.random.randint(0, len(image_data), num_images)
    classes = image_data.classes

    fig, axs = plt.subplots(1, num_images)
    fig.set_size_inches(20, 5)
    c = 0
    for i in to_visualize:
        axs[c].imshow(image_data[i][0].permute(1,2,0))
        predicted_num = torch.argmax(model(image_data[i][0].unsqueeze(0).type(torch.float32).to(model_device)), dim=1)
        title = f"Truth: {classes[image_data.targets[i]]} | Predicted: {classes[predicted_num]}"
        axs[c].set_title(title, c = 'g' if classes[image_data.targets[i]] == classes[predicted_num] else 'r', fontsize=10)
        axs[c].axis(False)
        c+=1