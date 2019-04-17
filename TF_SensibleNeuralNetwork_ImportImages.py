import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import TF_CNN
import torch.optim as optim
import datetime

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# download images
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# My Neural Network.
myNet = TF_CNN.Net() 

# Loss fuction.
criterion = nn.CrossEntropyLoss()

# Optimalization method.
optimizer = optim.SGD(myNet.parameters(), lr=0.001, momentum=0.9)

# Network Training
for epoh in range(3):
    local_loss = 0.0
    for i, data in enumerate(testloader, 0):
        [input, labels] = data

        # ? gradient cleaning. But why?
        optimizer.zero_grad() 

        out = myNet(input)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        local_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (1 + epoh, i + 1, local_loss / 2000))
            local_loss = 0.0

#  Random images.
dataiter = iter(trainloader)
[images, labels] = dataiter.next()

# Neural Network prosess.
out = myNet(images)

# Select the moest popobale anserw
[dummy , predicted] = torch.max(out, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# Real anserw
print('Truth Is:',' '.join('%5s' % classes[labels[j]] for j in range(4)))
imshow(torchvision.utils.make_grid(images))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = myNet(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

time = datetime.datetime.now().strftime("%y%m%d_%H%M")
outFileName = "results_" + time + ".txt"
outFile = open(outFileName, "w+")

for i in range(10):
    outFile.write('Accuracy of %5s : %2d %% \r\n' % (classes[i], 100 * class_correct[i] / class_total[i]))

outFile.close()