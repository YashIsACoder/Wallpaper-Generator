#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::dnn_superres;

// ============================
// Simple terminal progress
// ============================
void printProgress(int current, int total, const std::string& filename) {
    int percent = int(float(current) / total * 100);
    std::cout << "\rProcessing " << filename << " [" << percent << "%]" << std::flush;
    if (current == total) std::cout << std::endl;
}

// ============================
// Tile-based SR for large images
// ============================
void tileUpscale(DnnSuperResImpl& sr, const Mat& src, Mat& dst, int tileSize = 1024) {
    dst = Mat::zeros(src.rows * sr.getScale(), src.cols * sr.getScale(), src.type());
    int scale = sr.getScale();
    for (int y = 0; y < src.rows; y += tileSize) {
        for (int x = 0; x < src.cols; x += tileSize) {
            Rect roi(x, y, std::min(tileSize, src.cols - x), std::min(tileSize, src.rows - y));
            Mat tile = src(roi);
            Mat tile_sr;
            sr.upsample(tile, tile_sr);
            tile_sr.copyTo(dst(Rect(x*scale, y*scale, tile_sr.cols, tile_sr.rows)));
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_folder> <output_folder> <target_width> <target_height> <model_path>\n";
        return -1;
    }

    std::string inputFolder  = argv[1];
    std::string outputFolder = argv[2];
    int targetW = std::stoi(argv[3]);
    int targetH = std::stoi(argv[4]);
    std::string modelPath = argv[5];

    if (!fs::exists(outputFolder)) fs::create_directories(outputFolder);

    // Load model
    DnnSuperResImpl sr;
    try {
        sr.readModel(modelPath);
    } catch (const cv::Exception& e) {
        std::cerr << "❌ Could not load model: " << e.what() << std::endl;
        return -1;
    }
    sr.setModel("fsrcnn", 4);
    sr.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    sr.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Collect images
    std::vector<fs::path> images;
    for (auto& p: fs::directory_iterator(inputFolder)) {
        if (!p.is_regular_file()) continue;
        std::string ext = p.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext==".jpg" || ext==".jpeg" || ext==".png" || ext==".bmp" || ext==".tiff" || ext==".webp")
            images.push_back(p.path());
    }

    int total = images.size();
    if(total == 0) { std::cerr << "No images found in " << inputFolder << std::endl; return -1; }
    int count = 0;

    for (auto& imgPath : images) {
        std::string filename = imgPath.filename().string();
        printProgress(count, total, filename);

        Mat input = imread(imgPath.string(), IMREAD_COLOR);
        if (input.empty()) {
            std::cerr << "\n❌ Failed to read " << imgPath << std::endl;
            ++count;
            continue;
        }

        // Denoise
        Mat denoised;
        cv::fastNlMeansDenoisingColored(input, denoised, 3, 3, 7, 21);

        // AI upscale
        Mat sr_img;
        tileUpscale(sr, denoised, sr_img);

        // Resize
        Mat resized;
        resize(sr_img, resized, Size(targetW, targetH), 0, 0, INTER_LANCZOS4);

        // CLAHE
        Mat lab;
        cvtColor(resized, lab, COLOR_BGR2Lab);
        std::vector<Mat> planes;
        split(lab, planes);
        Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8));
        clahe->apply(planes[0], planes[0]);
        merge(planes, lab);
        cvtColor(lab, resized, COLOR_Lab2BGR);

        // Sharpen
        Mat blurred, sharpened;
        GaussianBlur(resized, blurred, Size(0,0), 3);
        sharpened = resized*1.5 - blurred*0.5;
        sharpened.convertTo(sharpened, CV_8U);

        // Save
        std::string outPath = outputFolder + "/" + imgPath.stem().string() + "_upscaled" + imgPath.extension().string();
        if (!imwrite(outPath, sharpened))
            std::cerr << "\n❌ Failed to write " << outPath << std::endl;
        else
            std::cout << "\n✅ Saved: " << outPath << std::endl;

        ++count;
    }

    printProgress(total, total, "Done");
    std::cout << "✅ All images processed!\n";
    return 0;
}
