import os
from glob import glob
import numpy as np
import argparse
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument("--image_folder_path", help="Path to image folder, where folder contains a set of jpg or png files", type=str)
parser.add_argument("--resolution", help="Image resolution", type=int)
parser.add_argument("--output_path", help="data path target (NPZ file)", type=str)
args = parser.parse_args()
exts = ['.jpg', '.png', '.jpeg']
filenames = [filename for ext in exts for filename in glob(os.path.join(args.image_folder_path, f'*{ext}'))]
imgs = []
for filename in filenames:
    print(f"Processing {filename}...")
    try:
        img = Image.open(filename)
    except Exception as ex:
        print(f"Exception when reading: {filename}, pass.")
        print(f"Exception is:{ex}")
        continue
    img = img.convert("RGB")
    img = img.resize((args.resolution, args.resolution))
    img = np.array(img)
    assert img.shape[2] == 3
    imgs.append(img)
imgs = np.array(imgs).astype("uint8")
print(imgs.shape)
np.savez(args.output_path, images=imgs)
