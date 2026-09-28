from glob import glob
import os
import torch
from animal_face_dataset import *
from torch.utils.data.sampler import SubsetRandomSampler
from Animal_Classification_Network import *
import matplotlib.pyplot as plt
import numpy as np
#######################################################################
# TODO (Q1.3): Extend the binary classification to multi-class classification
N_CLASSES = 2 # num of classes
#######################################################################

# !!! DO NOT MAKE ANY CHANGES AFTER THIS LINE !!!
def main():
    # Load test dataset
    #########################################################################################################
    label_map = {0: "Cat", 1: "Dog", 2: "Bear", 3: "Chicken", 4: "Cow", 5: "Deer", 6: "Duck", 7: "Eagle",
                 8: "Elephant", 9: "Human", 10: "Lion", 11: "Monkey", 12: "Mouse", 13: "Panda", 14: "Pigeon",
                 15: "Pig", 16: "Rabbit", 17: "Sheep", 18: "Tiger", 19: "Wolf"}
    main_path = "AnimalFace/test/"
    paths = []
    labels = []

    for i in range(N_CLASSES):
        folder = label_map[i] + 'Head'
        path_i = os.path.join(main_path, folder, "*")
        for each_file in glob(path_i):
            paths.append(each_file)
            labels.append(i)
    dataset = AnimalDataset(paths, labels, (150, 150))

    dataset_indices = list(range(0, len(dataset)))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    print("Number of test samples: ", len(dataset_indices))
    #########################################################################################################



    # Set up device (gpu or cpu), load CNN model
    ######################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(N_CLASSES).to(device)
    # Load your saved model here!
    model.load_state_dict(torch.load("model.pt"))
    ######################################################################



    # Start the test
    ######################################################
    total_true = 0
    total = len(dataset_indices)
    all_images, all_labels, all_preds = [], [], []
    with torch.no_grad():
        model.eval()
        for data_, target_ in test_loader:
            all_images.append(data_[0])
            data_ = data_.to(device)
            target_ = target_.to(device)

            outputs = model(data_)
            _, preds = torch.max(outputs, dim=1)
            true = torch.sum(preds == target_).item()
            total_true += true
            all_labels.append(int(target_[0]))
            all_preds.append(int(preds[0]))

    test_accuracy = round(100 * total_true / total, 2)
    print(f"Test accuracy: {test_accuracy}%")
    ######################################################

    # Visualize some test images with their ground-truth and predicted labels
    show_predictions(all_images, all_labels, all_preds, label_map)

    return
def show_predictions(images, labels, preds, label_map, n_rows=3, n_cols=5):
    n_show = min(n_rows * n_cols, len(images))
    n_rows = (n_show + n_cols - 1) // n_cols
    indices = np.linspace(0, len(images) - 1, n_show).astype(int)
    _, axis = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax in axis.flat:
        ax.axis("off")
    for ax, idx in zip(axis.flat, indices):
        ax.imshow(np.transpose(images[idx].numpy(), (1, 2, 0)))
        color = "green" if labels[idx] == preds[idx] else "red"
        ax.set_title(f"GT: {label_map[labels[idx]]}\nPred: {label_map[preds[idx]]}", color=color)
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()