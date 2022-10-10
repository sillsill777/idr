import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size=256,  # 256
            d_in=3,  # 3
            d_out=1,  # 1
            dims=[512,512,512,512,512,512,512,512],  # [512,512,512,512,512,512,512,512]
            geometric_init=True,
            bias=1.0,  # 0.6
            skip_in=(4,),  # [4]
            weight_norm=True,
            multires=6  # 6
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]  # [3,512,512...,512,257]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)  # embed_fn, 39
            self.embed_fn = embed_fn
            dims[0] = input_ch  # [39, 512,512,...,512, 257]

        self.num_layers = len(dims)  # 10
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        """
        input size = Batch, 3
        output size = Batch, 257

        At the last layer, no activation function.
        """
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size=256,
            mode='idr',
            d_in=9,
            d_out=3,
            dims=[512,512,512,512],
            weight_norm=True,
            multires_view=4
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]  # [256+9, 512,512,512,512, 3]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)  # input_ch=27
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        """
        At the final output tanh is used.
        """
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        rendering_input=None
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x

class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view)

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals


if __name__=='__main__':
    a=ImplicitNetwork()

    x=torch.arange(30).reshape(10,3).to(torch.float32)
    print(a(x).shape)
    out=a(x).sum()
    out.backward()
    print(out.requires_grad)
    print('-'*20)

    print(x.requires_grad)  # False
    print(x.grad)  # None
    x2=x.requires_grad_()
    out2=a(x2).sum()
    print(x2.requires_grad)  # True
    out2.backward()
    print(x2.grad)  # grad with shape (10,3)

    print('*'*30)
    b=RenderingNetwork()
    print(a.gradient(x).shape)

    print('*'*40)
    x = torch.rand(3, 4)
    x.requires_grad_()
    print(x)
    y=x**2
    z=torch.sum(x)
    gradZ=torch.autograd.grad(outputs=z,inputs=x)[0]
    print(gradZ)
    grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    print(grad)
    print(x.grad)
    grad2 = torch.autograd.grad(outputs=grad, inputs=x, grad_outputs=torch.ones_like(grad))[0]
    print(grad2)
    t=torch.sum(x**2)
    gradT=torch.autograd.grad(outputs=t, inputs=x, create_graph=True)[0]
    print(gradT)
    t.backward()
    print(x.grad)