from PIL import Image
import torchvision.transforms as transforms

# TODO (Q3.2): Add at least two data augmentation transforms for training
# (see https://pytorch.org/vision/stable/transforms.html).
# Notes:
#   - Images are already resized to 150x150 before this transform is applied, and the network
#     expects 150x150 inputs, so the output of your transforms must still be 150x150.
#   - Only the training set uses transform_train; the validation and test sets use `transform` below.
transform_train = transforms.Compose([
    # Your Code
    transforms.ToTensor(),
])

# !!! DO NOT MAKE ANY CHANGES AFTER THIS LINE !!!

transform = transforms.Compose([transforms.ToTensor()])

class InvalidDatasetException(Exception):
    def __init__(self, len_of_paths, len_of_labels):
        super().__init__(
            f"Number of paths ({len_of_paths}) is not compatible with number of labels ({len_of_labels})"
        )

class AnimalDataset():

    def __init__(self, img_paths, img_labels, size_of_images, split=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.size_of_images = size_of_images
        self.transform = transform_train if split == "train" else transform
        if len(self.img_paths) != len(self.img_labels):
            raise InvalidDatasetException(self.img_paths, self.img_labels)

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, index):
        # Make sure the loaded images are RGB
        PIL_IMAGE = Image.open(self.img_paths[index]).convert('RGB').resize(self.size_of_images)

        TENSOR_IMAGE = self.transform(PIL_IMAGE)
        label = self.img_labels[index]

        return TENSOR_IMAGE, label