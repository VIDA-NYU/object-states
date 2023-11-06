import torch
import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
import os
import numpy as np
import torch.hub

import pathtrees as pt
from IPython import embed


device = 'cuda'

def load_clip():
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    class CLIP(torch.nn.Module):
        def forward(self, x):
            return model.encode_image(x)
    return CLIP(), preprocess

def load_dino():
    # Load the DINOv2 model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').eval().cuda()

    # Define a transformation for preprocessing images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


@torch.no_grad()
@torch.inference_mode()
def run_model(model, transform, imagenet_directory, compute_image_features=True, batch_size=32, num_workers=12):
    tree = pt.tree(imagenet_directory, {
        '{split}/{object}__{state}/{video_name}__{frame_index:d}__{track_id:d}.JPEG': 'img'
    })

    # Create an ImageFolder dataset
    dataset = ImageFolder(root=imagenet_directory, transform=transform)#s.ToTensor()

    # Create a data loader to iterate through the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Initialize lists to store embeddings, image paths, frame numbers, and video basenames
    embeddings = []
    # img_features = []
    data = []

    # # Iterate through the dataset and extract embeddings
    with torch.no_grad():
        for i, (images, labels) in tqdm.tqdm(enumerate(data_loader), total=len(dataset)/batch_size):
            # embed()
            # print(images.shape, images.max(), images.min())
            # images_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            # print(images_np.shape, images_np.max(), images_np.min())
            # if model is not None:
                
            #     images = torch.stack([transform(Image.fromarray(im)) for im in images_np])
            features = model(images.cuda())
            embeddings.extend(features.cpu().numpy())
            # if compute_image_features:
            #     for im in images_np:
            #         img_features.append(calc_image_features(im))
            
            # Process each batch element and collect the relevant information
            for j in range(len(labels)):
                path, _ = dataset.imgs[i * data_loader.batch_size + j]
                data.append({**tree.img.parse(path), 'path': path})
            # break

    df = pd.DataFrame(data)
    df['vector'] = embeddings
    # if compute_image_features:
    #     df['color_features'] = img_features
    return df


import cv2
def calc_image_features(image, bins=8):
    image = (image*255).astype(np.uint8)
    # print(image.shape, image.max(), image.min())
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the 2D histogram for the Hue and Saturation channels
    hist = cv2.calcHist([hsv_image], [0, 1], None, [bins, bins], [0, 180, 0, 256])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    bgr_avg = image.mean((0,1))
    hsv_avg = image[:,:,:2].mean((0,1))
    return np.concatenate([hist, bgr_avg, hsv_avg])


def main(imagenet_directory, batch_size=32):
    model, transform = load_dino()
    df = run_model(model, transform, imagenet_directory, batch_size=batch_size)
    df.to_pickle(imagenet_directory + f'_dino.pkl')




if __name__ == '__main__':
    import fire
    fire.Fire(main)