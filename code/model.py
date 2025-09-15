import torch
import torch.nn as nn
import torch.nn.functional as F
from MiT import MixVisionTransformer

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device='cuda:0'):
        super(DSConv, self).__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)

    def forward(self, f):
        f = f.to(self.device)
        offset = self.offset_conv(f)
        offset = offset.to(self.device)
        offset = self.bn(offset)

        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph, device=self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        deformed_feature = deformed_feature.to(self.device)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
        else:
            x = self.dsc_conv_y(deformed_feature)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope

        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape(-1).float()
        x = x.reshape(-1).float()

        zero = torch.zeros([], device=self.device).int()
        max_y = self.width - 1
        max_x = self.height - 1

        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height
        )
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch, device=self.device) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height], device=self.device).unsqueeze(0).float()
        base = torch.matmul(base, repeat).reshape([-1])

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        index_a0 = base_y0 + x0
        index_c0 = base_y0 + x1
        index_a1 = base_y1 + x0
        index_c1 = base_y1 + x1

        value_a0 = input_feature_flat[index_a0.long()]
        value_c0 = input_feature_flat[index_c0.long()]
        value_a1 = input_feature_flat[index_a1.long()]
        value_c1 = input_feature_flat[index_c1.long()]

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        wa = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        wb = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        wc = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        wd = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * wa + value_c0 * wb + value_a1 * wc + value_c1 * wd)

        if self.morph == 0:
            outputs = outputs.reshape([self.num_batch, self.num_points * self.width, 1 * self.height, self.num_channels])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([self.num_batch, 1 * self.width, self.num_points * self.height, self.num_channels])
            outputs = outputs.permute(0, 3, 1, 2)

        return outputs
    def _coordinate_map_3D(self, offset, if_offset):

        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)


        y_center = torch.arange(0, self.width, device=self.device).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width).permute(1, 0).reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float().to(self.device).unsqueeze(0)

        x_center = torch.arange(0, self.height, device=self.device).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height).permute(0, 1).reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float().to(self.device).unsqueeze(0)

        if self.morph == 0:
            y = torch.linspace(0, 0, 1, device=self.device)
            x = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points), device=self.device)

            y, x = torch.meshgrid(y, x, indexing='ij')
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height]).reshape([self.num_points, self.width, self.height]).unsqueeze(0).to(self.device)
            x_grid = x_spread.repeat([1, self.width * self.height]).reshape([self.num_points, self.width, self.height]).unsqueeze(0).to(self.device)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone().to(self.device)

            if if_offset:
                y_offset = y_offset.to(self.device).permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.to(self.device).permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height]).permute(0, 3, 1, 4, 2).reshape([self.num_batch, self.num_points * self.width, 1 * self.height])
            x_new = x_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height]).permute(0, 3, 1, 4, 2).reshape([self.num_batch, self.num_points * self.width, 1 * self.height])
            return y_new, x_new

        else:
            y = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points), device=self.device)
            x = torch.linspace(0, 0, 1, device=self.device)

            y, x = torch.meshgrid(y, x, indexing='ij')
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height]).reshape([self.num_points, self.width, self.height]).unsqueeze(0).to(self.device)
            x_grid = x_spread.repeat([1, self.width * self.height]).reshape([self.num_points, self.width, self.height]).unsqueeze(0).to(self.device)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            x_offset_new = x_offset.detach().clone().to(self.device)

            if if_offset:
                x_offset = x_offset.to(self.device).permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.to(self.device).permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape([self.num_batch, 1, self.num_points, self.width, self.height]).permute(0, 3, 1, 4, 2).reshape([self.num_batch, 1 * self.width, self.num_points * self.height])
            x_new = x_new.reshape([self.num_batch, 1, self.num_points, self.width, self.height]).permute(0, 3, 1, 4, 2).reshape([self.num_batch, 1 * self.width, self.num_points * self.height])
            return y_new, x_new

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, expansion=4):
        super(InvertedResidualBlock, self).__init__()

        hidden_dim = in_channel * expansion
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.dwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.dwise(out)
        out = self.conv2(out)
        out = x + out
        return out

class AFFM(nn.Module):
    def __init__(self, input1_channels, input2_channels, out_channels, expansion=4, dsconv_params=None, device='cuda:0'):
        super(AFFM, self).__init__()
        if dsconv_params is None:
            dsconv_params = {
                'kernel_size': 15,
                'extend_scope': 1,
                'morph': 0,
                'if_offset': True
            }


        self.irb = InvertedResidualBlock(out_channels, out_channels, expansion=expansion)
        self.dsconv = DSConv(
            in_ch=input2_channels,
            out_ch=input2_channels,
            kernel_size=dsconv_params['kernel_size'],
            extend_scope=dsconv_params['extend_scope'],
            morph=dsconv_params['morph'],
            if_offset=dsconv_params['if_offset'],
            device=device
        )

        concat_channels = input1_channels + 2 * input2_channels
        self.catConv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

        expand_size = max(out_channels // expansion, 8)
        self.conv1d_1 = nn.Conv1d(out_channels, expand_size, kernel_size=3, padding=1, bias=False)
        self.conv1d_2 = nn.Conv1d(expand_size, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.gap_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gmp_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, input1, input2):

        input1_con = self.dsconv(input1)
        if input1.shape[2:] != input2.shape[2:]:
            input2 = F.interpolate(input2, size=input1.shape[2:], mode='bilinear', align_corners=True)
            input1_con = F.interpolate(input1_con, size=input1.shape[2:], mode='bilinear', align_corners=True)

        out = torch.cat((input1, input1_con, input2), dim=1)
        out = self.catConv(out)
        avg_pool = F.adaptive_avg_pool2d(out, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(out, output_size=(1, 1))
        avg_pool = self.gap_conv(avg_pool)
        max_pool = self.gmp_conv(max_pool)
        x = avg_pool + max_pool
        x = self.relu(self.conv1d_1(x.squeeze(-1)).unsqueeze(-1))
        x = self.sig(self.conv1d_2(x.squeeze(-1)).unsqueeze(-1))
        out = self.irb(out)
        out = out * x

        return out

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels

        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []

        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)

        return out_puts

class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=2):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)         # 创建PPM模块实例

        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class DAMiTUNet(nn.Module):
    def __init__(self, num_classes=2, embed_dims=[64, 128, 320, 512], ppm_out_channels=128, device='cuda:0'):
        super(DAMiTUNet, self).__init__()
        self.mit_encoder = MixVisionTransformer(
            in_channels=3,
            embed_dims=64,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1]
        )

        self.ppm = PPMHEAD(in_channels=embed_dims[3], out_channels=ppm_out_channels)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(ppm_out_channels, ppm_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ppm_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ppm_out_channels, ppm_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ppm_out_channels)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(ppm_out_channels, ppm_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ppm_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ppm_out_channels, ppm_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ppm_out_channels)
        )


        self.affm_3 = AFFM(input1_channels=embed_dims[2], input2_channels=embed_dims[2], out_channels=embed_dims[2],
                           device=device)
        self.affm_2 = AFFM(input1_channels=embed_dims[1], input2_channels=embed_dims[1], out_channels=embed_dims[1],
                           device=device)
        self.affm_1 = AFFM(input1_channels=embed_dims[0], input2_channels=embed_dims[0], out_channels=embed_dims[0],
                           device=device)


        self.upConv_3 = nn.Sequential(
            nn.Conv2d(ppm_out_channels, embed_dims[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.upConv_2 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.upConv_1 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.upConv_0 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0] // 2, kernel_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )


        self.final_conv = nn.Conv2d(embed_dims[0] // 2, num_classes, kernel_size=1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):

        features = self.mit_encoder(x)
        x1, x2, x3, x4 = features
        x4 = self.ppm(x4)
        residual = x4
        x4 = self.conv_block1(x4)
        x4 = nn.ReLU(inplace=True)(x4 + residual)
        residual = x4
        x4 = self.conv_block2(x4)
        x4 = nn.ReLU(inplace=True)(x4 + residual)
        out = self.upConv_3(x4)
        out = self.affm_3(x3, out)
        out = self.upConv_2(out)
        out = self.affm_2(x2, out)
        out = self.upConv_1(out)
        out = self.affm_1(x1, out)
        out = self.upConv_0(out)
        out = self.final_conv(out)
        out = self.final_upsample(out)

        return out

    def freeze_backbone(self):
        """
        Freezes the backbone weights by setting their 'requires_grad' to False.
        """
        for param in self.mit_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the backbone weights by setting their 'requires_grad' to True.
        """
        for param in self.mit_encoder.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = DAMiTUNet(num_classes=2, device=device).to(device)
    input_image = torch.randn(1, 3, 512, 512).to(device)
    output = net(input_image)
    print(output.shape)

