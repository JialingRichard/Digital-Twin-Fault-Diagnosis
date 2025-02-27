import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)  # 第一层 LSTM
        out, _ = self.lstm2(out)  # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])  # 全连接层
        return out


class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes, kernel_size=3, num_filters=64):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 转换为 (batch_size, input_size, sequence_length) 格式
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.mean(x, dim=2)  # 全局平均池化
       
        x = self.fc(x)
        x = F.softmax(x, dim=1) 
        return x
    
class ComplexCNN(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(ComplexCNN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, sequence_length) 格式
        x = self.tcn(x)
        x = torch.mean(x, dim=2)  # 全局平均池化
        x = self.fc(x)
        return x
    

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, nhead=2):
        super(TransformerModel, self).__init__()
        # 输入嵌入层，将输入映射到隐藏空间
        self.embedding = nn.Linear(input_size, hidden_size)
        # 定义Transformer的编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        # 定义Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)    
        # 定义全连接层来映射到分类输出
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 输入维度调整 (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
        x = x.permute(1, 0, 2)
        # 嵌入层
        x = self.embedding(x)
        # Transformer编码器
        x = self.transformer_encoder(x)
        # 取最后一个时间步的输出 (batch_size, hidden_size)
        x = x[-1, :, :]
        x = self.fc(x)
        return x
    

class CNNModelDANN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.2, kernel_size=2, num_channels=128, domain_output_size=2):
        super(CNNModelDANN, self).__init__()
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_channels, num_classes)
        )
        
        # 领域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Linear(num_channels, 100),
            nn.ReLU(),
            nn.Linear(100, domain_output_size)
        )
    
    def forward(self, x, alpha=0):
        """
        Args:
        - x: 输入数据，格式为 (batch_size, sequence_length, input_size)
        - alpha: 梯度反转层的系数，用于对抗性训练
        """
        # 特征提取器 (需要调整输入为 CNN 格式)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, sequence_length)
        features = self.feature_extractor(x)
        features = torch.mean(features, dim=2)  # 全局平均池化

        # 分类预测
        class_output = self.classifier(features)
        
        # 梯度反转
        reverse_features = GradientReversalLayer.apply(features, alpha)
        
        # 领域判别
        domain_output = self.domain_discriminator(reverse_features)
        
        return class_output, domain_output

class AlexNetDANN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.2, kernel_size=2, domain_output_size=2):
        super(AlexNetDANN, self).__init__()
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 1024),                          # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),                                 # FC2
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)                           # FC3
        )
        
        # 领域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 1024),                          # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),                                 # FC2
            nn.ReLU(inplace=True),
            nn.Linear(1024, domain_output_size)
        )
    
    def forward(self, x, alpha=0):
        """
        Args:
        - x: 输入数据，格式为 (batch_size, sequence_length, input_size)
        - alpha: 梯度反转层的系数，用于对抗性训练
        """
        # 特征提取器 (需要调整输入为 CNN 格式)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, sequence_length)
        features = self.feature_extractor(x)
        features = torch.mean(features, dim=2)  # 全局平均池化

        # 分类预测
        class_output = self.classifier(features)
        
        # 梯度反转
        reverse_features = GradientReversalLayer.apply(features, alpha)
        
        # 领域判别
        domain_output = self.domain_discriminator(reverse_features)
        
        return class_output, domain_output

class GradientReversalLayer(torch.autograd.Function):
    """
    梯度反转层 (Gradient Reversal Layer)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None





# TCN
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)  # 转换为 (batch_size, input_size, sequence_length)
        """Inputs have to have dimension (N, C_in, L_in)"""
        # print("inputs size: ", inputs.size())
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        # print("y1 size: ", y1.size())
        o = self.linear(y1[:, :, -1])
        # print("o size: ", o.size())
        return F.log_softmax(o, dim=1)
