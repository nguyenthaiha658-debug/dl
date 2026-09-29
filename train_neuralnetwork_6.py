from cifar_4 import CIFARDataset
from model_6 import SimpleNeuraNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim


if __name__ == '__main__':
    num_epochs = 100 # so lần huấn luyện mô hình
    train_dataset = CIFARDataset(root='./data', train=True)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    test_dataset = CIFARDataset(root='./data', train=True)

    train_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # 3. KHỞI TẠO MÔ HÌNH, LOSS VÀ OPTIMIZER
    model = SimpleNeuraNetwork(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    num_iters = len(train_dataloader)
    for epoch in range(num_epochs):
        model.train()
        for iter, (images, labels) in enumerate(train_dataloader):
            # Bước 1 (Forward)
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            print('so {}/{}. iter{}/{}, loss{}'.format(epoch+1, num_epochs, iter+1, num_iters, loss_value))

            # Bước 2 (Backward)
            optimizer.zero_grad()
            loss_value.backward()
            # Bước 3 (Update): Cập nhật trọng số mô hình
            optimizer.step()
