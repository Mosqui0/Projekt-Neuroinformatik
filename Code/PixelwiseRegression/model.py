import torch
from torch.functional import F

from PixelwiseRegression.utils import generate_com_filter, xavier_weights_init

class ResBlock(torch.nn.Module):
    def __init__(self, features, kernel_size=3, norm=torch.nn.BatchNorm2d, inplace=True):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Sequential(
            norm(features, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features, features // 2, 1, stride=1),
            norm(features // 2, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features // 2, features // 2, kernel_size, stride=1, padding=padding),
            norm(features // 2, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features // 2, features, 1, stride=1)
        )

    def forward(self, x):
        return x + self.conv(x)

class Hourglass(torch.nn.Module):
    def __init__(self, features, level=4, kernel_size=3, norm=torch.nn.BatchNorm2d):
        super(Hourglass, self).__init__()
        self.input_conv = ResBlock(features, kernel_size=kernel_size, norm=norm)
        self.down_sample = torch.nn.MaxPool2d(2, stride=2)

        if level > 0:
            self.inner = Hourglass(features, level - 1, kernel_size=kernel_size, norm=norm)
        else:
            self.inner = ResBlock(features, kernel_size=kernel_size, norm=norm)

        self.output_conv = ResBlock(features, kernel_size=kernel_size, norm=norm)

    def forward(self, x):
        x = self.input_conv(x)
        h = self.down_sample(x)

        h = self.inner(h)

        h = self.output_conv(h)
        h = F.interpolate(h, size=x.shape[2:])

        return h + x

class PlaneRegression(torch.nn.Module):
    def __init__(self, features, joints, label_size, kernel_size=3, norm=torch.nn.BatchNorm2d, inplace=True, normalization_method='softmax'):
        self.normalization_method = normalization_method
        super(PlaneRegression, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(features, features, kernel_size, stride=1, padding=padding),
            norm(features, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features, features, kernel_size, stride=1, padding=padding),
            norm(features, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features, features, kernel_size, stride=1, padding=padding),
            norm(features, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features, joints, kernel_size, stride=1, padding=padding)
        )

        com_weight = generate_com_filter(label_size, label_size) # ndarray[label_size, label_size, 2]
        com_weight = torch.from_numpy(com_weight).float()
        com_weight = com_weight.permute(2, 0, 1).contiguous()

        self.register_buffer('filter', com_weight)

        if self.normalization_method == 'softmax':
            self.register_parameter('w', torch.nn.Parameter(torch.ones(joints, 1)))

    def forward(self, f):
        heatmaps = self.conv(f)

        B, J, H, W = heatmaps.shape

        if self.normalization_method == 'softmax':
            # use softmax to normalize heatmap
            heatmaps = heatmaps.view(B, J, -1)
            heatmaps = F.softmax(self.w * heatmaps, dim=2)
            heatmaps = heatmaps.view(B, J, H, W)
        else:
            # use sum to normalize heatmap
            heatmaps = F.relu(heatmaps, inplace=True)
            heatmaps = heatmaps + 1e-14 # prevent all zero heatmap
            heatmaps = heatmaps / heatmaps.sum(dim=(2, 3), keepdim=True)

        u = torch.sum(self.filter[0].view(1, 1, H, W) * heatmaps, dim=(2, 3)).unsqueeze(-1)
        v = torch.sum(self.filter[1].view(1, 1, H, W) * heatmaps, dim=(2, 3)).unsqueeze(-1)

        plane_coordinates = torch.cat([u, v], dim=2)

        return heatmaps, plane_coordinates

class DepthRegression(torch.nn.Module):
    def __init__(self, features, joints, kernel_size=3, norm=torch.nn.BatchNorm2d, inplace=True):
        super(DepthRegression, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(features, features, kernel_size, stride=1, padding=padding),
            norm(features, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features, features, kernel_size, stride=1, padding=padding),
            norm(features, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features, features, kernel_size, stride=1, padding=padding),
            norm(features, affine=True),
            torch.nn.ReLU(inplace),
            torch.nn.Conv2d(features, joints, kernel_size, stride=1, padding=padding)
        )
    
    def forward(self, f, heatmaps, label_img, mask):
        depthmaps = self.conv(f)

        # if the input don't have extra dim
        # label_img = label_img.unsqueeze(1)
        # mask = mask.unsqueeze(1)

        reconstruction = depthmaps + label_img
        masked_reconstruction = mask * reconstruction
        masked_heatmaps = heatmaps * mask

        depth_coordinates = torch.sum(masked_heatmaps * masked_reconstruction, dim=(2, 3)) / (
            torch.sum(masked_heatmaps, dim=(2, 3)) + 1e-14 # prevent all masked heatmap
        )       
        depth_coordinates = depth_coordinates.unsqueeze(-1)

        return depthmaps, depth_coordinates

class PredictionBlock(torch.nn.Module):
    def __init__(self, in_dim, joints, label_size=64, features=256, level=4, kernel_size=3, norm=torch.nn.BatchNorm2d, heatmap_method='softmax'):
        super(PredictionBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_dim, features, 1, stride=1, padding=0)

        self.hourglass = Hourglass(features, level, norm=norm)

        self.plane_regression = PlaneRegression(features, joints, label_size, kernel_size=kernel_size, norm=norm, normalization_method=heatmap_method)
        self.depth_regression = DepthRegression(features, joints, kernel_size=kernel_size, norm=norm)

    def forward(self, x, label_img, mask):
        f = self.hourglass(self.conv(x))

        heatmaps, plane_coordinates = self.plane_regression(f)

        depthmaps, depth_coordinates = self.depth_regression(f, heatmaps, label_img, mask)

        return f, heatmaps, depthmaps, torch.cat([plane_coordinates, depth_coordinates], dim=2)

class PixelwiseRegression(torch.nn.Module):
    def __init__(self, joints, stage=2, label_size=64, features=256, level=4, kernel_size=3, norm_method='batch', heatmap_method='softmax'):
        super(PixelwiseRegression, self).__init__()

        if norm_method == 'batch':
            norm = torch.nn.BatchNorm2d
        elif norm_method == 'instance':
            norm = torch.nn.InstanceNorm2d

        padding = kernel_size // 2

        init_conv = [
            torch.nn.Conv2d(1, 32, kernel_size, stride=1, padding=padding),
            norm(32, affine=True),
            torch.nn.ReLU(True)
        ]

        conv_features = 32
        while conv_features < features:
            next_features = min(2 * conv_features, features)
            init_conv.extend([
                torch.nn.Conv2d(conv_features, next_features, kernel_size, stride=1, padding=padding),
                norm(next_features, affine=True),
                torch.nn.ReLU(True)
            ])
            # conv_features *= 2
            conv_features = next_features

        init_conv.extend([
            torch.nn.Conv2d(features, features, kernel_size, stride=2, padding=padding),
            norm(features, affine=True),
            torch.nn.ReLU(True)
        ])

        self.conv = torch.nn.Sequential(*init_conv)

        # concat_dim = features + 2 * joints + 1
        concat_dim = 2 * joints + 1
        stage_list = [
            PredictionBlock(features if i == 0 else concat_dim, joints, label_size, \
                features, level, kernel_size=kernel_size, heatmap_method=heatmap_method, norm=norm) for i in range(stage)
        ]

        self.stages = torch.nn.ModuleList(stage_list)

        self.apply(xavier_weights_init)

    def forward(self, img, label_img, mask):
        f = self.conv(img)

        results = []
        for stage in self.stages:
            f, heatmaps, depthmaps, uvd = stage(f, label_img, mask)
            results.append((heatmaps, depthmaps, uvd))
            # f = torch.cat([f, heatmaps, depthmaps, label_img], dim=1)
            f = torch.cat([heatmaps, depthmaps, label_img], dim=1)

        return results # list of tuple

# --------------------------------------------------------------------------------- #
#                        Below code is only for ablation                            #
# --------------------------------------------------------------------------------- #
class FullRegressionBlock(torch.nn.Module):
    def __init__(self, in_dim, joints, label_size=64, features=256, level=4, norm=torch.nn.BatchNorm2d):
        super(FullRegressionBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_dim, features, 1, stride=1, padding=0)

        self.hourglass = Hourglass(features, level, norm=norm)

        self.flatten_dim = label_size ** 2 * features // 64
        self.joints = joints

        self.downsampling = torch.nn.Sequential(
            torch.nn.Conv2d(features, features, 3, stride=2, padding=1),
            norm(features, affine=True),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(features, features, 3, stride=2, padding=1),
            norm(features, affine=True),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(features, features, 3, stride=2, padding=1),
            norm(features, affine=True),
            torch.nn.ReLU(True)
        )

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(self.flatten_dim, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, joints * 3)
        )

    def forward(self, x, label_img, mask):
        f = self.hourglass(self.conv(x))

        downsampled_f = self.downsampling(f)
        downsampled_f = downsampled_f.view(-1, self.flatten_dim)

        coordinates = self.regression(downsampled_f)
        # _f = f.view(-1, self.flatten_dim)
        # coordinates = self.regression(_f)

        coordinates = coordinates.view(-1, self.joints, 3)

        return f, coordinates

class FullRegression(torch.nn.Module):
    def __init__(self, joints, stage=2, label_size=64, features=256, level=4, norm_method='batch'):
        super(FullRegression, self).__init__()

        if norm_method == 'batch':
            norm = torch.nn.BatchNorm2d
        elif norm_method == 'instance':
            norm = torch.nn.InstanceNorm2d

        init_conv = [
            torch.nn.Conv2d(1, 32, 3, stride=1, padding=1),
            norm(32, affine=True),
            torch.nn.ReLU(True)
        ]

        conv_features = 32
        while conv_features < features:
            init_conv.extend([
                torch.nn.Conv2d(conv_features, 2 * conv_features, 3, stride=1, padding=1),
                norm(2 * conv_features, affine=True),
                torch.nn.ReLU(True)
            ])
            conv_features *= 2

        init_conv.extend([
            torch.nn.Conv2d(features, features, 3, stride=2, padding=1),
            norm(features, affine=True),
            torch.nn.ReLU(True)
        ])

        self.conv = torch.nn.Sequential(*init_conv)

        concat_dim = features + 1
        stage_list = [
            FullRegressionBlock(features if i == 0 else concat_dim, joints, label_size, features, norm=norm) for i in range(stage)
        ]

        self.stages = torch.nn.ModuleList(stage_list)

        self.apply(xavier_weights_init)

    def forward(self, img, label_img, mask):
        f = self.conv(img)

        results = []
        for stage in self.stages:
            f, uvd = stage(f, label_img, mask)
            results.append(uvd)
            f = torch.cat([f, label_img], dim=1)

        return results # list of tuple