import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, class_num = 8):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3) # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3) # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3) # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3) # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3) # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 7 * 7
        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, class_num)
        # softmax 1 * 1 * 1000

    def forward(self, x):

        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x) # 222
        out = F.relu(out)
        out = self.conv1_2(out) # 222
        out = F.relu(out)
        out = self.maxpool1(out) # 112

        out = self.conv2_1(out) # 110
        out = F.relu(out)
        out = self.conv2_2(out) # 110
        out = F.relu(out)
        out = self.maxpool2(out) # 56

        out = self.conv3_1(out) # 54
        out = F.relu(out)
        out = self.conv3_2(out) # 54
        out = F.relu(out)
        out = self.conv3_3(out) # 54
        out = F.relu(out)
        out = self.maxpool3(out) # 28

        out = self.conv4_1(out) # 26
        out = F.relu(out)
        out = self.conv4_2(out) # 26
        out = F.relu(out)
        out = self.conv4_3(out) # 26
        out = F.relu(out)
        out = self.maxpool4(out) # 14

        out = self.conv5_1(out) # 12
        out = F.relu(out)
        out = self.conv5_2(out) # 12
        out = F.relu(out)
        out = self.conv5_3(out) # 12
        out = F.relu(out)
        out = self.maxpool5(out) # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out