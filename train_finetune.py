# encoding:utf-8
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import model.HBPASM_model as HBPASM_model
import data.data as data
import os
import torch.backends.cudnn as cudnn
from PIL import Image
from utils.utils import progress_bar
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

trainset = data.MyDataset('./data/train_images_shuffle.txt', transform=transforms.Compose([
                                                transforms.Resize((600, 600), Image.BILINEAR),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.RandomVerticalFlip(),
                                                transforms.RandomCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=4)

testset = data.MyDataset('./data/test_images_shuffle.txt', transform=transforms.Compose([
                                                transforms.Resize((600, 600), Image.BILINEAR),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=4)
cudnn.benchmark = True


# Modify to your parameter path
model = HBPASM_model.Net('/data/guijun/HBP_finegrained/pth/resnet34.pth')

model.cuda()


pretrained = True
if pretrained:
    pre_dic = torch.load('firststep_batch16.pth')
    model.load_state_dict(pre_dic)

criterion = nn.NLLLoss()
lr = 1e-1

base_params = list(map(id, model.features.parameters()))
class_params = filter(lambda p: id(p) not in base_params, model.parameters())
optimizer = optim.SGD([{'params': model.features.parameters(), 'lr': 0.1*lr},
                       {'params': class_params, 'lr': lr}], momentum=0.9, weight_decay=1e-5)

def train(epoch):
    model.train()
    print('----------------------------------------Epoch: {}----------------------------------------'.format(epoch))
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader), 'loss: ' + str('{:.4f}'.format(loss.data.item())) + ' | train')



def test():
    model.eval()
    print('----------------------------------------Test---------------------------------------------')
    test_loss = 0
    correct = 0
    for batch_idx,(data, target) in enumerate(testloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        progress_bar(batch_idx, len(testloader), 'test')
    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 16., correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))

    if (100.0 * float(correct) / len(testloader.dataset)) >= 87.5:
        maxre = 100.0 * float(correct) / len(testloader.dataset)
        torch.save(model.state_dict(), 'bestScore'+str(maxre) + '.pth')

def adjust_learning_rate(optimizer, epoch):
    if epoch % 40 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

for epoch in range(1, 161):
    adjust_learning_rate(optimizer, epoch)

    train(epoch)
    if epoch % 5 == 0:
        test()
    elif epoch > 60:
        test()
torch.save(model.state_dict(), 'cub_result.pth')

