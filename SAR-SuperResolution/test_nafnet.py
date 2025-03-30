import os
import torch
import numpy as np
import torch.nn.functional as F
from nafnet.NAFNet_arch import NAFNet
from train_nafnet import sanitize_image, compute_metrics
from PIL import Image
import torchvision.transforms.functional as TF
import csv

def create_gaussian_weight(patch_size, sigma_ratio=0.125):
    sigma = patch_size * sigma_ratio
    ax = np.linspace(-(patch_size - 1) / 2., (patch_size - 1) / 2., patch_size)
    xx, yy = np.meshgrid(ax, ax)
    gaussian = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    gaussian /= np.max(gaussian)
    return gaussian.astype(np.float32)

def stitch_patches(sr_patches, indices, image_shape, patch_size, overlap):
    stride = patch_size - overlap
    reconstructed_image = np.zeros(image_shape, dtype=np.float32)
    weight_matrix = np.zeros(image_shape, dtype=np.float32)
    gaussian_weight = create_gaussian_weight(patch_size)

    for patch, (i, j) in zip(sr_patches, indices):
        patch = patch.squeeze()
        h_end = min(i + patch_size, image_shape[0])
        w_end = min(j + patch_size, image_shape[1])
        patch_h = h_end - i
        patch_w = w_end - j
        weight_crop = gaussian_weight[:patch_h, :patch_w]

        reconstructed_image[i:h_end, j:w_end] += patch[:patch_h, :patch_w] * weight_crop
        weight_matrix[i:h_end, j:w_end] += weight_crop

    reconstructed_image /= np.maximum(weight_matrix, 1e-6)
    return reconstructed_image

def extract_patches(image, patch_size, overlap):
    h, w = image.shape[-2:]
    stride = patch_size - overlap
    patches = []
    indices = []

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            patch = image[:, i:i + patch_size, j:j + patch_size]
            pad_h = max(0, patch_size - patch.shape[-2])
            pad_w = max(0, patch_size - patch.shape[-1])
            if pad_h > 0 or pad_w > 0:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
            patches.append(patch)
            indices.append((i, j))

    return patches, indices

def store_data_and_plot(im, threshold, filename):
    im = np.abs(im)
    im = np.clip(im, 0, threshold)
    im = (im / threshold * 255).astype(np.uint8)
    filename = filename if filename.endswith('.png') else filename + '.png'
    Image.fromarray(im).convert('L').save(filename)

def save_sar_images(denoised, noisy, imagename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    threshold = np.mean(np.abs(noisy)) + 3 * np.std(np.abs(noisy))
    denoisedfilename = os.path.join(save_dir, imagename)
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename.replace('.npy', ''))

def load_model(model_path, device):
    model = NAFNet(img_channel=1, width=96, middle_blk_num=12, enc_blk_nums=[2, 4, 6, 8], dec_blk_nums=[2, 4, 6, 8])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def process_all_images(model, lr_dir, hr_dir, sr_dir, patch_size, overlap, device, csv_path):
    results = []
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.npy')])

    for filename in hr_files:
        hr_path = os.path.join(hr_dir, filename)
        lr_path = os.path.join(lr_dir, filename)

        if not os.path.exists(lr_path):
            print(f"LR file not found for {filename}, skipping.")
            continue

        print(f"Processing {filename}...")
        hr = sanitize_image(np.load(hr_path).real)
        lr = sanitize_image(np.load(lr_path).real)

        hr_tensor = torch.tensor(hr).unsqueeze(0).float().to(device)
        lr_tensor = torch.tensor(lr).unsqueeze(0).float().to(device)

        lr_resized = TF.resize(lr_tensor, hr_tensor.shape[-2:], interpolation=TF.InterpolationMode.BICUBIC)

        patches, indices = extract_patches(lr_resized, patch_size, overlap)
        sr_patches = []

        with torch.no_grad():
            for patch in patches:
                sr_patch = model(patch.unsqueeze(0)).cpu().numpy()
                sr_patches.append(sr_patch)

        sr_image = stitch_patches(sr_patches, indices, hr.shape, patch_size, overlap)
        save_sar_images(sr_image, sr_image, filename, save_dir=sr_dir)

        # Metrics
        metrics = compute_metrics(sr_image, hr)
        results.append([filename, metrics["PSNR"], metrics["SSIM"]])

    # Save CSV
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "PSNR", "SSIM"])
        writer.writerows(results)

    print(f"\n✅ Finished. Results saved to {csv_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/nafnet_sr_best.pth"
    nafnet_sr = load_model(model_path, device)

    process_all_images(
        model=nafnet_sr,
        lr_dir="./data/Test/LR",
        hr_dir="./data/Test/HR",
        sr_dir="./data/SR",
        patch_size=512,
        overlap=128,
        device=device,
        csv_path="results_sr_metrics.csv"
    )

if __name__ == '__main__':
    main()