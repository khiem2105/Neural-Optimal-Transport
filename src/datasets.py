import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision.datasets import CelebA, SVHN, MNIST
import torchvision.transforms as transforms

from src.distributions import Sampler, LoaderSampler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ToyDataset(Dataset):
    def __init__(self, sampler: Sampler, n_samples: int):
        super(ToyDataset, self).__init__()

        self.data = sampler.sample(n_samples)
        self.len = n_samples
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], 0

def load_celeba(img_size, batch_size, root="datasets", num_workers=2, test_ratio=0.1):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])
    
    celeba = CelebA(
        root=root,
        split="all",
        target_type="attr",
        transform=transform,
        download=True
    )

    male_inds = torch.where(celeba.attr[:, 21] == 0)[0]
    female_inds = torch.where(celeba.attr[:, 21] == 1)[0]

    male_test_len = int(test_ratio * len(male_inds))
    male_train_len = len(male_inds) - male_test_len

    female_test_len = int(test_ratio * len(female_inds))
    female_train_len = len(female_inds) - female_test_len

    celeba_male_train, celeba_male_test = random_split(
        Subset(celeba, male_inds),
        [male_train_len, male_test_len]
    )

    celeba_female_train, celeba_female_test = random_split(
        Subset(celeba, female_inds),
        [female_train_len, female_test_len]
    )

    celeba_male_trainloader = LoaderSampler(
        DataLoader(celeba_male_train, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        device=DEVICE
    )
    celeba_male_test_loader = LoaderSampler(
        DataLoader(celeba_male_test, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        device=DEVICE
    )

    celeba_female_trainloader = LoaderSampler(
        DataLoader(celeba_female_train, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        device=DEVICE
    )
    celeba_female_testloader = LoaderSampler(
        DataLoader(celeba_female_test, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        device=DEVICE
    )

    return celeba_male_trainloader, celeba_male_test_loader, celeba_female_trainloader, celeba_female_testloader
    

def load_digit_dataset(batch_size: int, root: str, name: str, img_size: int=28, num_workers: int=2):
    if name == "SVHN":
        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transform.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        train_dataset = SVHN(
            root=root,
            split="train",
            transform=transform,
            download=True
        )

        test_dataset = SVHN(
            root=root,
            split="test",
            transform=transform,
            download=True
        )
    elif name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transform.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_dataset = MNIST(
            root=root,
            train=True,
            transform=transform,
            download=True
        )

        test_dataset = MNIST(
            root=root,
            transform=transform,
            train=False
        )
    else:
        raise Exception("Not implemented datasets")

    train_loader = LoaderSampler(
        DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        device=DEVICE
    )

    test_loader = LoaderSampler(
        DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        device=DEVICE
    )

    return train_loader, test_loader