
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np


def gradcamplpl_mask(model, target_layer, img_path, cuda_device):
    target_layers = [target_layer]
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(cuda_device)
    targets = None
    methods = {
        "gradcam++": GradCAMPlusPlus
    }
    cam_algorithm = methods["gradcam++"]

    with cam_algorithm(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True
                            )

        grayscale_cam = grayscale_cam[0, :]

        #cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        #cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

    return grayscale_cam
