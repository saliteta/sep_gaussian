from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch
import math
from dataclasses import dataclass, field
import numpy as np
from plyfile import PlyData

@dataclass
class CameraModel:
    FoVx: float  # radiance of the field of view along x-axis
    FoVy: float  # radiance of the field of view along y-axis
    world_view_transform: torch.Tensor  # World Coordinates to Camera Coordinates (Extrinsic)
    full_proj_transform: torch.Tensor  # World Coordinates Project to 2D plane (Extrinsic * Intrinsic)
    camera_center: torch.Tensor  # Camera center (x,y,z)
    image_height: int = field(default=1960)
    image_width: int = field(default=1080)

class GaussianModel:

    def __init__(self, ply_path) -> None:
        
        self.max_sh_degree = 3
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        
        plydata = PlyData.read(ply_path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = (torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._features_dc = (torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
        self._features_rest = (torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
        self._opacity = (torch.tensor(opacities, dtype=torch.float, device="cuda"))
        self._scaling = (torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = (torch.tensor(rots, dtype=torch.float, device="cuda"))

        self.active_sh_degree = self.max_sh_degree

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)




class Render:
    
    def __init__(self, pc: GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, default_camera_pose = None) -> None:
        self.model = pc
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.camera_model = default_camera_pose

        # Set up rasterization configuration
        tanfovx = math.tan(self.camera_model.FoVx * 0.5)
        tanfovy = math.tan(self.camera_model.FoVy * 0.5)

        self.gs_raster_setting = GaussianRasterizationSettings(
            image_height=int(self.camera_model.image_height),
            image_width=int(self.camera_model.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=self.camera_model.world_view_transform,
            projmatrix=self.camera_model.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=self.camera_model.camera_center,
            prefiltered=False,
            debug=False,
    )
        
        self.rasterizer = GaussianRasterizer(raster_settings=self.gs_raster_setting)
        self.means2D = torch.zeros_like(pc._xyz, dtype=pc._xyz.dtype, device="cuda")

    def forward(self, viewpoint_camera:CameraModel,  scaling_modifier = 1.0) -> torch.Tensor:
        """
        Render the scene. 

        Background tensor (bg_color) must be on GPU!
        """
    
        self.gs_raster_setting = self.gs_raster_setting._replace(
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            campos=viewpoint_camera.camera_center
        )

        self.rasterizer.raster_settings = self.gs_raster_setting


        scales = self.model.get_scaling
        rotations = self.model.get_rotation
        means3D = self.model.get_xyz
        opacity = self.model.get_opacity
        shs = self.model.get_features


        rendered_image, radii = self.rasterizer(
            means3D = means3D,
            means2D = self.means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None)

        return rendered_image


def getProjectionMatrix(fovX, fovY, znear=0.01, zfar=100):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4).cuda()

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def qvec2rotmat_tensor(qvec: torch.Tensor) -> torch.Tensor:
    return torch.Tensor([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]]).cuda()


def getWorld2View2(R, t, translate=torch.Tensor([.0, .0, .0]).cuda(), scale=1.0)->torch.Tensor:
    Rt = torch.zeros((4, 4)).cuda()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return torch.tensor(Rt, dtype=torch.float32)


def load_camera_model(
                      position: torch.Tensor, orientation:torch.Tensor,
                      h, w, fx, fy, cx, cy):
    orientation = torch.Tensor([orientation[3], orientation[0], orientation[1], orientation[2]]).cuda()
    R = qvec2rotmat_tensor(orientation)
    T = position
    
    # Construct world to view transform (extrinsic matrix 4x4)
    world_view_transform = getWorld2View2(R=R, t=T).transpose(0,1)
    
    2*math.atan(w/(2*fx))
    FoVx=2 * math.atan2(cx, fx)
    FoVy=2 * math.atan2(cy, fy)

    projection_matrix = getProjectionMatrix(fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    
    return CameraModel(
        FoVx=FoVx,
        FoVy=FoVy,
        image_height=h,
        image_width=w,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center
    )