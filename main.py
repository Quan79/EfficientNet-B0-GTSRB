import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
# pip install efficient_pytorch
from efficientnet_pytorch import EfficientNet
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from load_data import data_loader
from validate import validate_single_batch, validate_model

print(f"torch version: {torch.__version__}")
gup_availability = torch.cuda.is_available()
if gup_availability:
    device = "cuda"
    print("Cuda ia available")
else:
    device = "CPU"
    print("Cuda is not available, use CPU")


input_size = 224
class_number = 43

batch_size = 32
epoch = 10
learning_rate = 0.0001

train_set_loader, test_set_loader = data_loader(input_size, batch_size)

model_ft = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = model_ft._fc.in_features
model_ft._fc = nn.Linear(num_ftrs, class_number)
model_ft.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate)

# dataiter = iter(train_set_loader)
# images, labels = dataiter.__next__()
# result = model_ft.forward(images.to(device))
# print("shape of result:", result.shape)


def train(model, device, train_set_loader, validation_set_loader, optimizer, loss_function, epoch):

    epoch_loss = []
    epoch_accuracy = []

    for epoch in range(1, epoch + 1):
        model.train()
        batch_loss = []
        batch_accuracy = []
        for batch_idx, (batch, label) in enumerate(train_set_loader):
            batch, label = Variable(batch).to(device), Variable(label).to(device)
            optimizer.zero_grad()
            prediction = model(batch)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            batch_accuracy.append(validate_single_batch(model, batch, label))

            if (batch_idx + 1) % 100 == 0:
                print(f"Training epoch {epoch}: [{(batch_idx + 1) * len(batch)}/{len(train_set_loader.dataset)}]"
                      f"({100*((batch_idx + 1) / len(train_set_loader)):.2f}%) "
                      f"Loss= {np.mean(batch_loss)} "
                      f"Accuracy= {(np.mean(batch_accuracy)):.5f}")

        epoch_loss.append(np.mean(batch_loss))
        epoch_accuracy.append(np.mean(batch_accuracy))
        validation_score, validation_loss = validate_model(model, validation_set_loader, loss_function, device)
        print(f"Finished epoch {epoch}: "
              f"Loss= {epoch_loss[-1]} "
              f"Accuracy= {(epoch_accuracy[-1]):.5f} "
              f"Validation loss= {validation_loss} " 
              f"Validation accuracy= {validation_score:.5f}")


train(model_ft, device, train_set_loader, test_set_loader, optimizer, loss_function, epoch)
torch.save(model_ft, "./models/model_epoch_60_lt_0001.pth")
