"""
    Now we need to construct a automatically evaluation script

    The input is output folder, and we will iterate each submodel's center for evaluation. 
    The reason for us to do so is because, if we use the large scale reconstruction result after filtering and merge, 
    the peripheral might lose information. For fair comparison, we need to use the submodel. 

    The reason for using center is that the peripheral are actually overlap with other model, and in the end will be filtered out, 
    which does not represent the over all evaluation result 

    We will select 2 centered images 
"""




import os
import torch
import torchvision.transforms.functional
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import pycolmap
import torchvision
from typing import Tuple, List
from render import CameraModel, GaussianModel, load_camera_model, Render
import numpy as np
from colorama import Fore, Back
from scipy.spatial import cKDTree
import pandas as pd
import PIL.Image as Image


def load_camera_intrinsics(reconstruction: pycolmap.Reconstruction):
    intrinsics = {}
    for camera_id, camera in reconstruction.cameras.items():
        camera = camera.todict()
        model = camera['model']
        width = camera['width']
        height = camera['height']
        params = camera['params']
        
        if model.name == 'PINHOLE':
            fx, fy, cx, cy = params
        elif model.name == 'SIMPLE_PINHOLE':
            f, cx, cy = params
            fx = fy = f
        else:
            # Handle other models as needed
            continue
        
        intrinsics[camera_id] = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'width': width,
            'height': height
        }
        
    assert len(intrinsics) >= 1, "at least have one camera, we have zero"
    return intrinsics


def get_comparison_images(reconstruction: pycolmap.Reconstruction, K = 2) -> List[Tuple[CameraModel, str]]:
    """
    This function will takes in an pycolmap  reconstruction, 
    and select the centered K images and return it's related 
    information
        Args: 
            reconstruction (pycolmap.Reconstruction): A colmap reconstruction 
            for us to select centered K image
            K(int): how many image we will select from center
        Returns:
            cameraList: it will return a camera List contain a tuple, the first 
            element of it is the camera pose and the size of image, determined by
            colmap information, and the second element contain the gt image location
    """
    translation = []
    quat = []
    name = []
    intrinsic = load_camera_intrinsics(reconstruction=reconstruction)[1]
    fx, fy, cx, cy, w, h = intrinsic['fx'], intrinsic['fy'], intrinsic['cx'], \
        intrinsic['cy'], intrinsic['width'], intrinsic['height']
    for _, image in reconstruction.images.items():
        img_dict = image.cam_from_world.todict()
        quat.append(img_dict['rotation']['quat'])
        translation.append(img_dict['translation'])
        name.append(image.name)
        
    translation = np.array(translation)
    mean_points = translation.mean(axis=0)

    tree = cKDTree(translation)

    _, indices = tree.query(mean_points, k = K)

    cameraList = []
    for i in indices:
        cameraList.append(
            (load_camera_model(
                position=torch.Tensor(translation[i]).cuda(),
                orientation=torch.Tensor(quat[i]).cuda(),
                h = h,
                w = w,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy
            ) , name[i])
        )
    return cameraList


def render(render: Render, cameraList: List[Tuple[CameraModel, str]], output_path: str, original_image_folder: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    We render the cameraList according to its camera position, and pass
    that to gs_model for render, we will save the rendered result to some
    place, but and the same time, we will return a list of CPU images 
    torch.tensor for further comparison. And that list of CPU images, should contain
    first the rendered result, and second gt images

        Args: 
            gs_model(gs_model): The model trained for rendering
            cameraList(List[Tuple[CameraModel, str]]): CameraModel and string of path
            output_path(str): Evaluated Result Location
        Return: 
            image_pair(List[torch.Tensor, torch.Tensor]): Rendered image and GT images pair images
    """ 


    image_list = []
    for (cam, name) in cameraList:
        rendered_image = render.forward(
            cam
        )

        rendered_image = rendered_image.cpu()
        
        torchvision.utils.save_image(rendered_image, os.path.join(output_path, name))
        gt_image = torchvision.transforms.functional.to_tensor(Image.open(os.path.join(original_image_folder, name)))
        image_list.append((rendered_image, gt_image))


    return image_list


def metrics(imageList: List[Tuple[torch.Tensor, torch.Tensor]], cameraList: List[Tuple[CameraModel, str]], output_path) -> None:
    """
    We use PSNR, LPIPS, and SSIM to get a measured result of each image pair,
    and we use the images name to generate a CSV file using pandas. It should contain
    each image's result, and we should also have overall, and average result.
    
    Args:
        imageList (List[Tuple[torch.Tensor, torch.Tensor]]): List of rendered image and GT image pairs.
        cameraList (List[Tuple[CameraModel, str]]): List of CameraModel and path string.
        output_path (str): Evaluated Result Location, metrics.csv.
    """    

    # Initialize a list to collect rows for each image
    rows = []

    for i, (rendered, gt) in enumerate(tqdm(imageList)):
        name = cameraList[i][1]
        gt = gt.unsqueeze(0).cuda()
        rendered = rendered.unsqueeze(0).cuda()
        new_row = {
            'name': name,
            'psnr': psnr(gt, rendered),
            'ssim': ssim(gt, rendered),
            'lpips': lpips(gt, rendered, net_type='vgg')
        }
        rows.append(new_row)
    
    # Create DataFrame from the list of rows
    df = pd.DataFrame(rows)

    # Calculate mean values for PSNR, SSIM, and LPIPS
    psnr_mean = df['psnr'].mean()
    ssim_mean = df['ssim'].mean()
    lpips_mean = df['lpips'].mean()
    mean_row = {'name': 'means', 'psnr': psnr_mean, 'ssim': ssim_mean, 'lpips': lpips_mean}

    # Append the mean row to the DataFrame using concat
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(output_path, 'report.csv'), index=False)


def parser():
    parser = ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("--model_location", help="the place where we store models, and seperations")
    parser.add_argument("--images", help="The place where we store the large scale dataset")

    args = parser.parse_args()
    return args


def main():
    args = parser()

    model_location = args.model_location
    model_folders = os.path.join(model_location, 'models')
    colmap_folders = os.path.join(model_location, 'seperation')
    output_folders = os.path.join(model_location, "eval")

    image_folder = args.images

    if os.path.exists(colmap_folders) and os.path.exists(model_folders):
        print(Fore.GREEN, "Your seperation and model is succssefully found")
        os.makedirs(output_folders, exist_ok=True)
        print(Fore.GREEN, "Your evaluation and folder is succssefully create")
    else:
        print(Fore.RED, f"no models folder and colmap folder, we expect there should be a folder named as {model_location}/models after training")
        exit()

    imageList = []
    cameraList_sum = []

    print(Fore.RESET, "START EVALUATION")
    for models in tqdm(os.listdir(model_folders)):
        ply_location = os.path.join(model_folders, models, 'point_cloud/iteration_30000/point_cloud.ply')
        colmap_location = os.path.join(colmap_folders, models)

        reconstruction = pycolmap.Reconstruction(colmap_location)
        camearList = get_comparison_images(reconstruction=reconstruction)
        cameraList_sum = cameraList_sum + camearList

        gs_model = GaussianModel(ply_path=ply_location)
        renderer = Render(pc=gs_model, bg_color=torch.Tensor([0,0,0]).cuda(), default_camera_pose=camearList[0][0])
        imageList = imageList + render(render=renderer, cameraList=camearList, output_path=output_folders, original_image_folder=image_folder)

    print(Fore.RESET, "Data Preparation Accomlished")
    metrics(imageList=imageList, cameraList=cameraList_sum,output_path=output_folders)

    print(Fore.GREEN, "Evaluation Done")
    print(Fore.GREEN, f"Result Store {output_folders}. report store {output_folders}/report.csv")



def get_particular(image_name, reconstruction):
    intrinsic = load_camera_intrinsics(reconstruction=reconstruction)[1]
    fx, fy, cx, cy, w, h = intrinsic['fx'], intrinsic['fy'], intrinsic['cx'], \
        intrinsic['cy'], intrinsic['width'], intrinsic['height']
    quat = None
    translation = None
    for _, image in reconstruction.images.items():
        img_dict = image.cam_from_world.todict()
        quat=(img_dict['rotation']['quat'])
        translation=(img_dict['translation'])

        if image.name == image_name:
            print(quat)
            print(translation)
            print(intrinsic)
            exit()
        

if __name__ == "__main__":
    #reconstruction = pycolmap.Reconstruction('/data2/butian/GauUscene/CUHK_UPPER_CAMPUS_COLMAP/sparse/0')
    #get_particular('DJI_20231219112144_0002_Zenmuse-L1-mission.JPG', reconstruction)

    main()