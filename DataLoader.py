import torch
import torchvision
import torchvision.transforms as transforms


class DataLoader:
    def __init__(self):
        self.train = None
        self.test = None
        self.names = None

    def getCIFAR10(self):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data/CIFAR10', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data/CIFAR10', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

        self.train = trainloader
        self.test = testloader
        self.names = classes

    def getFashionMNIST(self):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.FashionMNIST(
            root='./data/FashionMNIST', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2)

        testset = torchvision.datasets.FashionMNIST(
            root='./data/FashionMNIST', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2)

        classes = ['Top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Shoe']

        self.train = trainloader
        self.test = testloader
        self.names = classes

    def getSTL10(self):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.STL10(
            './data/stl10', split='train', transform=transform, download=True)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2)

        testset = torchvision.datasets.STL10(
            './data/stl10', split='test', transform=transform, download=True)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2)

        classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

        self.train = trainloader
        self.test = testloader
        self.names = classes
