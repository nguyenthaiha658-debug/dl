from cifar_4 import CIFARDataset
from model_6 import SimpleNeuraNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from sklearn.metrics import classification_report


if __name__ == '__main__':
    # num_epochs = 100 # so lần huấn luyện mô hình
    # train_dataset = CIFARDataset(root='./data', train=True)
    #
    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     num_workers=4,
    #     drop_last=True,
    # )
    #
    # test_dataset = CIFARDataset(root='./data', train=False)
    #
    # test_dataloader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=4,
    #     drop_last=False,
    # )
    #
    # # 3. KHỞI TẠO MÔ HÌNH, LOSS VÀ OPTIMIZER
    # model = SimpleNeuraNetwork(num_classes=10)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # num_iters = len(train_dataloader)
    #
    # if torch.cuda.is_available():
    #     model.cuda()
    #     # print('co cuda')
    #
    # for epoch in range(num_epochs):
    #     model.train()
    #     for iter, (images, labels) in enumerate(train_dataloader):
    #         if torch.cuda.is_available():
    #             images = images.cuda()
    #             labels = labels.cuda()
    #         # Bước 1 (Forward)
    #         outputs = model(images)
    #         loss_value = criterion(outputs, labels)
    #         # print('so {}/{}. iter{}/{}, loss{}'.format(epoch+1, num_epochs, iter+1, num_iters, loss_value))
    #
    #         # Bước 2 (Backward)
    #         optimizer.zero_grad() # soá gradient
    #         loss_value.backward() # tính gradient
    #         # Bước 3 (Update): Cập nhật trọng số mô hình
    #         optimizer.step()
    #
    #     model.eval()  # Chuyển sang chế độ đánh giá
    #     all_predictions = []
    #     all_labels = []
    #     for iter, (images, labels) in enumerate(test_dataloader):
    #         all_labels.extend(labels)
    #         if torch.cuda.is_available():
    #             images = images.cuda()
    #             labels = labels.cuda()
    #         # Tắt tính toán gradient
    #         with torch.no_grad():
    #             predictions = model(images)
    #             indices = torch.argmax(predictions.cpu(), dim=1)
    #             all_predictions.extend(indices)
    #             loss_value = criterion(predictions, labels)
    #             # Chuyển tensor sang list số nguyên chuẩn của Python
    #     all_labels = [label.item() for label in all_labels]
    #     all_predictions = [prediction.item() for prediction in all_predictions]
    #     print("Epoch {}".format(epoch + 1))
    #     print(classification_report(all_labels, all_predictions))
    print('hello')
