import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block1 = BasicBlock(in_channels, in_channels)
        self.block2 = BasicBlock(in_channels, in_channels)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return x + out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels * scale_factor, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=1, num_features=64, num_blocks=3):
        super(Encoder, self).__init__()
        self.entry = BasicBlock(in_channels, num_features)
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_features) for _ in range(num_blocks)])

    def forward(self, x):
        out = self.entry(x)
        features = []
        for block in self.residual_blocks:
            out = block(out)
            features.append(out)
        return out, features

class Decoder(nn.Module):
    def __init__(self, num_features=64, out_channels=1, scale_factor=2):
        super(Decoder, self).__init__()
        self.upsample = UpsampleBlock(num_features, scale_factor)
        self.exit = nn.Conv1d(num_features, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.exit(out)
        return out

class Estimator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Estimator, self).__init__()
        self.fc_mu = nn.Linear(in_features, out_features)
        self.fc_b = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.fc_mu(x)
        b = self.fc_b(x)
        return mu, b

class TeacherNetwork(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=3, scale_factor=2):
        super(TeacherNetwork, self).__init__()
        self.encoder = Encoder(in_channels, num_features, num_blocks)
        self.decoder = Decoder(num_features, out_channels, scale_factor)

    def forward(self, x):
        encoded, features = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, features

class StudentNetwork(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=3, scale_factor=2, estimator_in_features=64, estimator_out_features=1):
        super(StudentNetwork, self).__init__()
        self.decoder = Decoder(num_features, out_channels, scale_factor)
        self.estimator = Estimator(estimator_in_features, estimator_out_features)

    def forward(self, x, teacher_features):
        decoded = self.decoder(x)
        mu, b = self.estimator(x)
        return decoded, teacher_features, mu, b
