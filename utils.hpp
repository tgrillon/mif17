#pragma once
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>

std::chrono::high_resolution_clock::time_point start;
std::chrono::high_resolution_clock::time_point stop;

std::chrono::microseconds duration;

#define MEASURE_TIME(func) \
        start = std::chrono::high_resolution_clock::now(); \
        func; \
        stop = std::chrono::high_resolution_clock::now(); \
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
        std::cout << "Time taken by " << #func << ": " << (duration.count() / 1000.0) << "ms" << std::endl; \

float radians(float a) { return M_PI / 180 * a; }

float degrees(float a) { return 180 / M_PI * a; }

cv::Mat calcHistCumul(const cv::Mat &src, int histSize) {
  cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
  dst.at<float>(0) = src.at<float>(0);

  for (int i = 1; i < histSize; ++i) {
    dst.at<float>(i) = dst.at<float>(i - 1) + src.at<float>(i);
  }
  return dst;
}

cv::Mat etirement(const cv::Mat &image, int Nmin, int Nmax) {
  int height = image.rows;
  int width = image.cols;
  cv::Mat image2(height, width, CV_8UC1);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      image2.at<uchar>(r, c) =
          cv::saturate_cast<uchar>(255 * ((image.at<uchar>(r, c) - Nmin) /
                                          static_cast<double>(Nmax - Nmin)));
    }
  }

  return image2;
}

cv::Mat egalisation(const cv::Mat &inputImage, const cv::Mat &inputHist,
                    int histSize) {
  cv::Mat histoCumul = calcHistCumul(inputHist, histSize);

  histoCumul /= inputImage.total();

  cv::Mat outputImage = inputImage.clone();

  for (int i = 0; i < outputImage.rows; ++i) {
    for (int j = 0; j < outputImage.cols; ++j) {
      outputImage.at<uchar>(i, j) = cv::saturate_cast<uchar>(
          255 * histoCumul.at<float>(inputImage.at<uchar>(i, j)));
    }
  }
  return outputImage;
}

float convolution(const cv::Mat &img, const cv::Mat &h, int x, int y) {
  float sum = 0.0;
  for (int u = -1; u <= 1; ++u) {
    for (int v = -1; v <= 1; ++v) {
      sum += h.at<float>(1 + u, 1 + v) * img.at<uchar>(y + u, x + v);
    }
  }
  return sum;
}

template <typename T> void print_mat(cv::Mat const &mat) {
  for (int r = 0; r < mat.rows; ++r) {
    for (int c = 0; c < mat.cols; ++c) {
      std::cout << (float)mat.at<T>(r, c) << " ";
    }
    std::cout << std::endl;
  }
}

void filter(const cv::Mat &src, cv::Mat &dst, const cv::Mat &h) {
  assert(h.rows == 3 && h.cols == 3);
  int height = src.rows;
  int width = src.cols;
  dst = cv::Mat::zeros(src.size(), src.type());

  for (int r = 1; r < height - 1; ++r) {
    for (int c = 1; c < width - 1; ++c) {
      dst.at<uchar>(r, c) = cv::saturate_cast<uchar>(convolution(src, h, c, r));
    }
  }
}

void thresholding(cv::Mat const &src, cv::Mat &dst, uchar ths) {
  assert(src.type() == CV_8UC1);
  dst = src.clone();
  int rows = src.rows;
  int cols = src.cols;
  for (int r = 1; r < rows - 1; ++r) {
    for (int c = 1; c < cols - 1; ++c) {
      uchar val = src.at<uchar>(r, c);
      // if (val < ths) val = (val - ths*.5 < 0) ? 0 : val - ths*.5;
      // else val = (val + ths*.5 > 255) ? 255 : val + ths*.5;
      if (val < ths)
        val = 0;
      else
        val = 255;
      dst.at<uchar>(r, c) = val;
    }
  }
}

void minmax(const cv::Mat &img, double *min, double *max) {
  cv::Point empty;
  cv::minMaxLoc(img, min, max, &empty, &empty);
}

struct TimeFunction {
  long long microseconds = 0;

  TimeFunction() {}

  void operator()(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  }

  TimeFunction(std::function<void()> func) { (*this)(func); }

  void print(std::string name = "function") {
    std::cout << "Time taken for " << name << " : " << microseconds
              << " micro seconds ";
  }
};
