import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import TensorfFlowCNN
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt
from DataLoader import DataLoader
import torchvision

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imFeelingLucky():
    #  Selct random images.
    dataiter = iter(loader.train)
    [images, labels] = dataiter.next()

    # Neural Network prosess.
    out = myNet(images)

    # Guess lucky anserw
    [dummy , predicted] = torch.max(out, 1)
    print('Predicted: ', ' '.join('%5s' % loader.names[predicted[j]] for j in range(4)))

    # Real anserw
    print('Truth Is:',' '.join('%5s' % loader.names[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))

def main():
    info = {
        "data": "",
        "loss": "",
        "optim": "",
        "epoh": 2,
        "accuracy":{ },
    }

    loader = DataLoader()
    loader.getCIFAR10()
    info["data"] = "CIFAR10"

    # My Neural Network.
    myNet = TensorfFlowCNN.Net()

    # Loss fuction.
    criterion = nn.CrossEntropyLoss()
    info["loss"] = "CrossEntropyLoss"

    # Optimalization method.
    optimizer = optim.SGD(myNet.parameters(), lr=0.001, momentum=0.9)
    info["optim"] = "SGD"

    # Network Training
    for epoh in range(info["epoh"]):
        local_loss = 0.0
        for i, data in enumerate(loader.test, 0):
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

    # Network Testing
    class_correct = list(0. for i in range(len(loader.names)))
    class_total = list(0. for i in range(len(loader.names)))
    with torch.no_grad():
        for data in loader.test:
            images, labels = data
            outputs = myNet(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    i = 0
    for name in loader.names:
        info["accuracy"].update({name: (100 * class_correct[i] / class_total[i]) })
        i += 1

    # Save results text
    outFile = open(resultFilePath("json"), "w+")
    outFile.write(str(info))
    outFile.close()

    # Save results image
    plt.bar(*zip(*info["accuracy"].items()))
    plt.title(info["loss"]+" "+info["optim"])
    plt.ylabel("Accuracy %")
    plt.ylim(0,100)
    plt.savefig(resultFilePath("png"))


def resultFilePath(filetype):
    time = datetime.datetime.now().strftime("%y%m%d_%H%M")
    return ("results/results_" + time + "."+filetype)



if __name__ == "__main__":
    main()