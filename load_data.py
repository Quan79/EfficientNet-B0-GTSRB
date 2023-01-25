import torch
import torchvision
import torchvision.transforms as transforms


def data_loader(input_size, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize([input_size, input_size])
        ])

    train_set = torchvision.datasets.GTSRB(
        root="data",
        split="train",
        download=True,
        transform=transform
    )
    test_set = torchvision.datasets.GTSRB(
        root="data",
        split="test",
        download=True,
        transform=transform
    )
    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    dataiter = iter(train_set_loader)
    images, labels = dataiter.__next__()
    print(f"length of train_data_loader: {len(train_set_loader)}")
    print(f"shape of image in train_data_loader: {images.shape}")
    print(f"shape of label in train_data_loader: {labels.shape}")
    print(f"shape of flattened item in train_data_loader: {images.flatten(start_dim=1).shape}")

    return train_set_loader, test_set_loader
