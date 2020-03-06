from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def check_args(args):
    pass


def calculate_mean_and_std(dataset_path):
    dataset = datasets.ImageFolder(dataset_path)
    dataloader = DataLoader(dataset, batch_size=8)
    mean = 0.
    std = 0.
    for images, _ in dataloader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)
    return mean, std


def train_transform(image_size, mean, std):
    """ Transforms for training images """
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform


def gram_matrix(y):
    """ Returns the gram matrix, used to compute style loss"""
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram
