# Image Deblurring with Deep Learning

A complete PyTorch project for restoring sharp images from blurred photos using deep learning.

## 📁 Project Structure

```
image-deblurring/
├── environment.yml          # Environment setup
├── requirements.txt         # Python packages
├── README.md               # This file
├── download_dataset.py      # Automatic dataset downloader
├── train.py                # Train your own model
├── inference.py            # Use trained model on images
├── models/
│   └── unet.py             # Model architectures
└── utils/                  # Helper functions
    ├── dataset.py
    └── metrics.py
```

## 🛠️ Quick Start

### 1. Setup Environment
```bash
# Download this project
git clone https://github.com/getmokshshah/image-deblurring.git
cd image-deblurring

# Create environment (choose one method)
conda env create -f environment.yml && conda activate deblur
# OR
pip install -r requirements.txt
```

### 2. Get Training Data
```bash
# Download dataset and create training pairs (takes ~10 minutes)
python download_dataset.py --prepare-blur
```

This downloads 900 high-quality images and creates 1,200+ blurred/sharp training pairs automatically.

### 3. Train Model
```bash
# Start training (takes 1-3 hours depending on your GPU)
python train.py \
    --train_path ./data/processed/train \
    --val_path ./data/processed/val \
    --epochs 50 \
    --batch_size 4
```

### 4. Use Your Model
```bash
# Process a single image
python inference.py \
    --model_path ./models/best_model.pth \
    --input_path blurry_photo.jpg \
    --output_path sharp_photo.jpg \
    --save_comparison

# Process multiple images
python inference.py \
    --model_path ./models/best_model.pth \
    --input_path ./blurry_photos/ \
    --output_path ./results/ \
    --batch_mode \
    --save_comparison
```

## 📊 What Dataset is Used?

**DIV2K High-Resolution Dataset**
- **Content**: 900 diverse, high-quality photos (people, nature, objects, scenes)
- **Quality**: Professional 2K resolution images
- **Size**: ~2GB after processing
- **Blur Creation**: I apply realistic motion and focus blur onto these images

## 🧠 Model Options

### U-Net (Recommended)
- **Architecture**: Standard encoder-decoder with skip connections
- **Quality**: Highest image restoration quality
- **Speed**: Moderate
- **Memory**: Requires ~6GB GPU memory
- **Best for**: Getting the best possible results

### Lightweight Model
- **Architecture**: Custom lightweight encoder-decoder
- **Quality**: Good image restoration
- **Speed**: 3x faster training and inference
- **Memory**: Works with 2GB GPU memory
- **Best for**: Quick results or limited hardware

Choose the lightweight model by adding `--model_type deblurnet` to training commands. (Note: `deblurnet` is just the internal name for the lightweight model)

## 📈 Understanding Results

The models are measured using standard metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**
  - Higher = better (20-35+ is good for deblurring)
  - Measures overall image quality

- **SSIM (Structural Similarity)**
  - Range 0-1, closer to 1 = better
  - Measures how similar images look to humans

## 🔧 Training Options

### Basic Training
```bash
python train.py \
    --train_path ./data/processed/train \
    --val_path ./data/processed/val \
    --epochs 50 \
    --batch_size 4
```

### Advanced Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 100 | How long to train (50-100 usually enough) |
| `--batch_size` | 8 | Images per batch (reduce if out of memory) |
| `--lr` | 0.001 | Learning rate |
| `--model_type` | `unet` | Architecture: `deblurnet` for lightweight, `unet` for standard |
| `--image_size` | 256 | Input image size (smaller = faster) |

### Monitor Training Progress
```bash
# View training charts in browser
tensorboard --logdir ./models/logs
# Then open http://localhost:6006
```

## 🔮 Using Your Trained Model

### Single Image
```bash
python inference.py \
    --model_path ./models/best_model.pth \
    --input_path my_blurry_photo.jpg \
    --output_path my_sharp_photo.jpg \
    --save_comparison
```

### Folder of Images
```bash
python inference.py \
    --model_path ./models/best_model.pth \
    --input_path ./my_blurry_photos/ \
    --output_path ./my_results/ \
    --batch_mode \
    --save_comparison
```

The `--save_comparison` option creates side-by-side before/after images so you can see the improvement.

## 🐛 Potential Issues & Solutions

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --batch_size 2

# Use lightweight model (uses less memory)
python train.py --model_type deblurnet

# Use smaller images
python train.py --image_size 128
```

### "No matching image pairs found"
- Make sure you ran `python download_dataset.py --prepare-blur` first
- Check that `./data/processed/train/` has both `blurred/` and `sharp/` folders

### Slow training
- Use a GPU if available (much faster than CPU)
- Reduce `--num_workers` if you get data loading errors
- Make sure dataset is on fast storage (SSD preferred)

### Downloads failing
```bash
# Retry the download
python download_dataset.py --prepare-blur

# Use different storage location
python download_dataset.py --data_dir /different/path --prepare-blur
```

## 💡 Tips for Best Results

- **For highest quality**: Use U-Net with default settings
- **For speed**: Use the lightweight model with `--model_type deblurnet` and `--batch_size 2`
- **Limited memory**: Reduce `--image_size` to 128 or 192
- **Custom data**: Replace images in `./data/processed/train/` with your own blur/sharp pairs

## 📂 Using Your Own Images

If you have your own blurred and sharp image pairs, organize them like this:

```
data/processed/
├── train/
│   ├── blurred/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── sharp/
│       ├── image1.jpg  # Same filename as blurred version
│       ├── image2.jpg
│       └── ...
└── val/
    ├── blurred/
    └── sharp/
```

## 📄 License

This project is licensed under the MIT License - feel free to use for research or commercial projects.

## 🙏 Credits

- **Dataset**: DIV2K dataset by Agustsson & Timofte (NTIRE 2017)
- **U-Net Architecture**: Based on Ronneberger et al.
- **Built with**: PyTorch, OpenCV, and other open-source libraries