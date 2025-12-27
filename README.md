# Wallpaper Generator 

High-quality AI image upscaling tool using OpenCV’s **DNN Super-Resolution (FSRCNN/EDSR)**. Designed for ultra-high-resolution wallpaper generation with enhancements like denoising, sharpening, CLAHE, and gamma correction.

This version is **single-threaded and robust**, optimized for macOS, Linux, and Windows.

---

## Features

* AI-powered super-resolution using **FSRCNN** (or any OpenCV-compatible model).
* Tile-based upscaling for **large images** to prevent memory issues.
* Automatic **denoising** and **sharpening** for professional-quality results.
* CLAHE-based **contrast enhancement**.
* Gamma correction and HSV color enhancement for vivid colors.
* Batch processing of entire folders.
* Supports common image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
* Reliable output with automatic folder creation.
* Single-threaded, avoiding complex concurrency issues.

---

## Folder Structure

```
SuperResolutionApp/
├── CMakeLists.txt
├── src/
│   └── main.cpp
├── models/
│   └── FSRCNN_x4.pb      # Example SR model
├── input/                # Input images folder
├── output/               # Processed images folder (auto-created)
└── README.md
```

---

## Prerequisites

* **C++17 compiler** (`clang++`, `g++`, or MSVC).
* **CMake ≥ 3.18**.
* **OpenCV ≥ 4.5** with modules:

  * `core`, `imgproc`, `highgui`, `imgcodecs`, `dnn`, `dnn_superres`, `photo`.

**Optional:** CUDA build for GPU acceleration if you modify the backend (current version uses CPU).

---

## Setup Instructions

### 1. Clone repository

```bash
git clone <repo_url>
cd SuperResolutionApp
```

### 2. Place model

Download a compatible FSRCNN or EDSR model (OpenCV `.pb` format) and put it into `models/`.

Example:

```
models/FSRCNN_x4.pb
```

---

### 3. Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

> On macOS, `nproc` can be replaced with `sysctl -n hw.ncpu`.

---

### 4. Run

```bash
./super_resolution <input_folder> <output_folder> <target_width> <target_height> <model_path>
```

**Example:**

```bash
./super_resolution ../input ../output 3840 2160 ../models/FSRCNN_x4.pb
```

* Input folder: folder containing images to process.
* Output folder: will be created if it doesn’t exist.
* Target width/height: final output resolution (e.g., 3840×2160 for 4K).
* Model path: path to the `.pb` super-resolution model.

**Output filenames:** Original name + `_upscaled` suffix.

```
input.png → input_upscaled.png
photo.jpg → photo_upscaled.jpg
```

---

## Example Workflow

1. Put your input images into `input/`.
2. Ensure the SR model is in `models/`.
3. Run the program to upscale images to 4K or 8K.
4. Check `output/` folder for enhanced wallpapers.

The terminal shows a **progress bar per image** and outputs a confirmation for each saved image.

---

## Enhancements Applied

* **Denoising:** OpenCV `fastNlMeansDenoisingColored`.
* **Tile-based Super-Resolution:** Avoids memory issues on large images.
* **High-quality resizing:** `INTER_LANCZOS4`.
* **CLAHE:** Contrast Limited Adaptive Histogram Equalization on L-channel.
* **Sharpening:** Unsharp masking with Gaussian blur.
* **HSV saturation/brightness boost** for vivid colors.
* **Gamma correction** for perceptually better output.

---

## Notes

* Single-threaded design avoids complex concurrency bugs on macOS arm64.
* Tile size can be modified in `tileUpscale()` (default 1024).
* CPU is used by default. GPU acceleration is possible if OpenCV DNN is built with CUDA.
* Works for images of any resolution, including ultra HD and 8K.

---

## Troubleshooting

* **Image not written:** Ensure `output/` folder exists and is writable.
* **Model fails to load:** Verify `.pb` model path and format. Use OpenCV’s supported SR models (FSRCNN/EDSR).
* **Memory issues on huge images:** Reduce tile size in `tileUpscale()`.

---

## License

MIT License – free to use and modify.

