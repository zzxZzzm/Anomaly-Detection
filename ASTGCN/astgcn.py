import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x

class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)

class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("nm,bitm->bitn", self.Lk, x)
        x_gc = torch.einsum("iok,bitn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)

class SpatialAttention(nn.Module):
    def __init__(self, c_in):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv2d(c_in, c_in // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(c_in, c_in // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(c_in, c_in, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return out

class TemporalAttention(nn.Module):
    def __init__(self, c_in, time_steps):
        super(TemporalAttention, self).__init__()
        self.query_conv = nn.Conv2d(c_in, c_in // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(c_in, c_in // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(c_in, c_in, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height * width)
        energy = torch.bmm(key, query)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch, -1, height * width)
        out = torch.bmm(attention, value)
        out = out.view(batch, channels, height, width)
        return out

class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.s_attention = SpatialAttention(c[1])
        self.t_attention = TemporalAttention(c[1], kt)
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.t_attention(x)
        x = self.tconv1(x)
        x = self.s_attention(x)
        x = self.sconv(x)
        x = self.t_attention(x)
        x = self.tconv2(x)
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)

class fully_conv_layer(nn.Module):
    def __init__(self, c):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class output_layer(nn.Module):
    def __init__(self, c, T, n):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = fully_conv_layer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class ASTGCN(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p):
        super(ASTGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk)
        self.output = output_layer(bs[1][2], T - 4 * (kt - 1), n)

    def forward(self, x):
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        return self.output(x_st2)

class QuantileLoss(nn.Module):
    def __init__(self):
        super(QuantileLoss, self).__init__()

    def forward(self, y_preds, y, r):
        d = y - y_preds
        return torch.where(d >= 0, r * d, (r - 1) * d).mean()
