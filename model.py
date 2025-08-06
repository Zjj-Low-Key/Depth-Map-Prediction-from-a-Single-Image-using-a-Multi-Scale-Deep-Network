import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal
import mindspore.ops as ops


class coarseNet(nn.Cell):
    def __init__(self, init_weight= True):
        super(coarseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, pad_mode='valid',padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, pad_mode='pad',padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, pad_mode='pad',padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, pad_mode='pad',padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=2, pad_mode='valid',padding=0)
        self.fc1 = nn.Linear(12288, 4096)
        self.fc2 = nn.Linear(4096, 4070)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout = nn.Dropout(p=0.5)
        if init_weight:
            self._initialize_weights()

    def construct(self, x):
                                #1 3 228 304
        x = self.conv1(x)
        x = ops.relu(x)
        x = self.pool(x)
        if ops.any(ops.isnan(x)):
            print("nan in conv1")
        x = self.conv2(x)       
        x = ops.relu(x)
        x = self.pool(x)
        if ops.any(ops.isnan(x)):
            print("nan in conv2")
        x = self.conv3(x)
        x = ops.relu(x)
        if ops.any(ops.isnan(x)):
            print("nan in conv3")
        x = self.conv4(x)
        x = ops.relu(x)
        if ops.any(ops.isnan(x)):
            print("nan in conv4")
        x = self.conv5(x)
        x = ops.relu(x)
        if ops.any(ops.isnan(x)):
            print("nan in conv5")
        x = x.view(x.shape[0], -1)
        x = ops.relu(self.fc1(x))
        x = self.dropout(x)
        if ops.any(ops.isnan(x)):
            print("nan in fc1")
        x = self.fc2(x)
        if ops.any(ops.isnan(x)):
            print("nan in fc2")
        x = x.view(-1, 1, 55, 74)
        return x
    
    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d) or isinstance(cell, nn.Linear):
                cell.weight.set_data(initializer(Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
                    
class fineNet(nn.Cell):
    def __init__(self, init_weight= True):
        super(fineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 63, kernel_size=9, stride=2,pad_mode='valid',padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, pad_mode='pad',padding=2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, stride=1, pad_mode='pad',padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.concat = ops.Concat(axis=1)
        if init_weight:
            self._initialize_weights()
    
    def construct(self, x, y):
        
        x = ops.relu(self.conv1(x))
        x = self.pool(x)
        x = self.concat((x, y))
        x = ops.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d) or isinstance(cell, nn.Linear):
                cell.weight.set_data(initializer(Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
                    

if __name__ == '__main__':
    net = coarseNet()
    net.set_train()
    x = mindspore.Tensor(mindspore.numpy.randn(8, 3, 228, 304).astype(mindspore.float32))
    y = mindspore.Tensor(mindspore.numpy.randn(8, 1, 55, 74).astype(mindspore.float32))
    z = net(x)
    print(z.shape)