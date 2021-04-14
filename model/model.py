import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class RandlaNet(nn.Module):
    def __init__(self, d_out, n_layers, n_classes):
        super(RandlaNet, self).__init__()
        self.n_classes = n_classes
        dilate_block_in = 8
        self.fc1 = nn.Linear(6, dilate_block_in)
        self.bn1 = nn.BatchNorm1d(dilate_block_in, eps=1e-6, momentum=0.01)
        self.f_encoders = nn.ModuleList()
        decoder_in_list = [d_out[0]*2]
        for i in range(n_layers):
            self.f_encoders.append(DilatedResidualBlock(dilate_block_in, d_out[i]))
            dilate_block_in = d_out[i]*2
            decoder_in_list.append(dilate_block_in)

        self.conv2 = nn.Conv2d(dilate_block_in, dilate_block_in,
                               kernel_size=[1, 1], stride=[1, 1])
        self.bn2 = nn.BatchNorm2d(dilate_block_in, eps=1e-6, momentum=0.01)

        self.f_decoders = nn.ModuleList()
        for i in range(n_layers):
            self.f_decoders.append(FeatureDecoder(decoder_in_list[-i-1] +
                                                  decoder_in_list[-i-2],
                                   decoder_in_list[-i-2]))
        self.conv3 = nn.Conv2d(decoder_in_list[0], 64, kernel_size=[1, 1],
                               stride=[1, 1])
        self.bn3 = nn.BatchNorm2d(64, eps=1e-6, momentum=0.01)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=[1, 1], stride=[1, 1])
        self.bn4 = nn.BatchNorm2d(32, eps=1e-6, momentum=0.01)
        self.drop4 = nn.Dropout2d(p=0.5)
        self.conv5 = nn.Conv2d(32, self.n_classes, kernel_size=[1, 1],
                               stride=[1, 1])

    def forward(self, inputs):
        x = inputs['features']
        x = self.fc1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = x[:, :, :, None]
        encoded_list = []
        for i, encoder in enumerate(self.f_encoders):
            x = encoder(x, inputs['xyz'][i], inputs['neigh_idx'][i])
            if i == 0:
                encoded_list.append(x.clone())
            x = random_sample(x, inputs['sub_idx'][i])
            encoded_list.append(x.clone())
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        for i, decoder in enumerate(self.f_decoders):
            x = decoder(x, encoded_list[-i-2], inputs['interp_idx'][-i-1])
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = self.drop4(x)
        x = self.conv5(x)
        x = x.squeeze(-1).permute(0, 2, 1).reshape([-1, self.n_classes]).contiguous()
        return x


class FeatureDecoder(nn.Module):
    def __init__(self, f_in, f_out):
        super(FeatureDecoder, self).__init__()
        self.trconv1 = nn.ConvTranspose2d(f_in, f_out, kernel_size=[1, 1],
                                          stride=[1, 1])
        self.bn1 = nn.BatchNorm2d(f_out, eps=1e-6, momentum=0.01)

    def forward(self, feature, encoded_feature, interp_idx):
        f_interp_i = nearest_interpolation(feature, interp_idx)
        f_decoded = self.trconv1(torch.cat([encoded_feature, f_interp_i],
                                           dim=1))
        f_decoded = self.bn1(f_decoded)
        return f_decoded


class DilatedResidualBlock(nn.Module):
    def __init__(self, f_in, d_out):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(f_in, d_out//2, kernel_size=[1, 1],
                               stride=[1, 1])
        self.bn1 = nn.BatchNorm2d(d_out//2, eps=1e-6, momentum=0.01)
        self.bb = BuildingBlock(d_out)
        self.conv2 = nn.Conv2d(d_out, d_out*2, kernel_size=[1, 1],
                               stride=[1, 1])
        self.bn2 = nn.BatchNorm2d(d_out*2, eps=1e-6, momentum=0.01)
        self.shortcut = nn.Conv2d(f_in, d_out*2, kernel_size=[1, 1],
                                  stride=[1, 1])
        self.bn_shortcut = nn.BatchNorm2d(d_out*2, eps=1e-6, momentum=0.01)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = F.leaky_relu(self.bn1(self.conv1(feature)), negative_slope=0.2)
        f_pc = self.bb(xyz, f_pc, neigh_idx)
        f_pc = self.bn2(self.conv2(f_pc))
        shortcut = self.bn_shortcut(self.shortcut(feature))
        return F.leaky_relu(f_pc + shortcut)


class BuildingBlock(nn.Module):
    def __init__(self, d_out):
        super(BuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(10, d_out//2, kernel_size=[1, 1], stride=[1, 1])
        self.bn1 = nn.BatchNorm2d(d_out//2, eps=1e-6, momentum=0.01)
        self.attpool1 = AttentivePooling(2*(d_out//2), d_out//2)
        self.conv2 = nn.Conv2d(d_out//2, d_out//2, kernel_size=[1, 1],
                               stride=[1, 1])
        self.bn2 = nn.BatchNorm2d(d_out//2, eps=1e-6, momentum=0.01)
        self.attpool2 = AttentivePooling(2*(d_out//2), d_out)

    def forward(self, xyz, feature, neigh_idx):
        f_xyz = relative_pos_encoding(xyz, neigh_idx)
        f_xyz = F.leaky_relu(self.bn1(self.conv1(f_xyz)), negative_slope=0.2)
        feature = torch.squeeze(feature, dim=-1).permute(0, 2, 1).contiguous()
        f_neighbours = gather_neighbour(feature, neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.attpool1(f_concat)

        f_xyz = F.leaky_relu(self.bn2(self.conv2(f_xyz)),  negative_slope=0.2)
        f_pc_agg = torch.squeeze(f_pc_agg, dim=-1).permute(0, 2, 1).contiguous()
        f_neighbours = gather_neighbour(f_pc_agg, neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.attpool2(f_concat)
        return f_pc_agg


class AttentivePooling(nn.Module):
    def __init__(self, n_feature, d_out):
        super(AttentivePooling, self).__init__()
        self.n_feature = n_feature
        self.fc1 = nn.Linear(n_feature, n_feature, bias=False)
        self.conv1 = nn.Conv2d(n_feature, d_out, kernel_size=[1, 1],
                               stride=[1, 1])
        self.bn1 = nn.BatchNorm2d(d_out, eps=1e-6, momentum=0.01)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[2]
        num_neigh = x.shape[3]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.reshape(x, [-1, num_neigh, self.n_feature])
        att_activation = self.fc1(x)
        att_score = F.softmax(att_activation, dim=1)
        x = x * att_score
        x = torch.sum(x, dim=1)
        x = torch.reshape(x, [batch_size, num_points, self.n_feature])[:, :, :, None].permute(0, 2, 1, 3).contiguous()
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        return x


def relative_pos_encoding(xyz, neighbor_idx):
    neighbor_xyz = gather_neighbour(xyz, neighbor_idx)
    xyz = xyz[:, :, None, :].permute(0, 3, 1, 2).contiguous()
    repeated_xyz = xyz.repeat(1, 1, 1, 16)
    relative_xyz = repeated_xyz - neighbor_xyz
    relative_dist = torch.sqrt(torch.sum(relative_xyz**2, dim=1, keepdim=True))
    relative_feature = torch.cat([relative_dist, relative_xyz, repeated_xyz, neighbor_xyz], dim=1)
    return relative_feature


def gather_neighbour(point_features, neighbor_idx):
    batch_size = point_features.shape[0]
    n_points = point_features.shape[1]
    n_features = point_features.shape[2]
    index_input = torch.reshape(neighbor_idx, shape=[batch_size, -1])
    features = batch_gather(point_features, index_input)
    features = torch.reshape(features, [batch_size,
                                        n_points,
                                        neighbor_idx.shape[-1],
                                        n_features])
    return features.permute(0, 3, 1, 2).contiguous()


def random_sample(feature, pool_idx):
    feature = torch.squeeze(feature, dim=3)
    num_neigh = pool_idx.shape[-1]
    batch_size = pool_idx.shape[0]
    d = feature.shape[1]
    feature = feature.permute(0, 2, 1).contiguous()
    pool_idx = torch.reshape(pool_idx, [batch_size, -1])
    pool_features = batch_gather(feature, pool_idx)
    pool_features = torch.reshape(pool_features, [batch_size, -1, num_neigh, d])
    pool_features = torch.max(pool_features, dim=2, keepdim=True)[0]
    return pool_features.permute(0, 3, 1, 2).contiguous()


def nearest_interpolation(feature, interp_idx):
    feature = torch.squeeze(feature, dim=3)
    batch_size = interp_idx.shape[0]
    up_num_points = interp_idx.shape[1]
    interp_idx = torch.reshape(interp_idx, [batch_size, up_num_points])
    feature = feature.permute(0, 2, 1).contiguous()
    interp_features = batch_gather(feature, interp_idx)
    return interp_features.permute(0, 2, 1)[:, :, :, None].contiguous()


def batch_gather(tensor, indices):
    shape = list(tensor.shape)
    device = tensor.device
    flat_first = torch.reshape(
        tensor, [shape[0] * shape[1]] + shape[2:])
    offset = torch.reshape(
        torch.arange(shape[0], device=device) * shape[1],
        [shape[0]] + [1] * (len(indices.shape) - 1))
    output = flat_first[indices.long() + offset]
    return output
