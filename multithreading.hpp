#pragma once
#include "applications.hpp"
#include "gradient.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <thread>

// Hough lines from bin, multithreading
// 1. divide the image by n*n squares
// 2.

struct MutexMatrix3D {
  std::vector<std::mutex> pixelMutexes;
  int size_a, size_b, size_c;

  MutexMatrix3D(int size_a, int size_b, int size_c)
      : size_a(size_a), size_b(size_b), size_c(size_c),
        pixelMutexes(size_a * size_b * size_c) {}

  std::mutex &get_mutex(int a, int b, int c) {
    int index = a + b * size_a + c * size_a * size_b;
    if (index > size_a * size_b * size_c || index < 0) {
      std::cerr << std::endl << std::endl << "Invalid......." << std::endl;
    }
    return pixelMutexes[index];
  }
};

struct ThreadStruct {
  ThreadStruct(MutexMatrix3D &mut, cv::Mat acc, cv::Point range, int max_a,
               int max_b)
      : mutexes(mut), accumulator(acc), range_cols(range), max_a(max_a),
        max_b(max_b) {}

  MutexMatrix3D &mutexes;
  cv::Mat accumulator;
  cv::Point range_cols;
  int max_a;
  int max_b;
};

void circle_accumulator(ThreadStruct thread, const cv::Mat &bin,
                        uchar binThresh) {
  for (int y = 0; y < bin.rows; y++) {
    for (int x = thread.range_cols.x; x < thread.range_cols.y; x++) {
      if (bin.at<uchar>(y, x) < binThresh)
        continue;

      for (int a = 0; a < thread.max_a; a++) {
        for (int b = 0; b < thread.max_b; b++) {
          float da = a - x;
          float db = b - y;
          // Calculer directement r
          float r = sqrt(da * da + db * db);

          std::unique_lock lock(thread.mutexes.get_mutex(a, b, r));
          thread.accumulator.at<float>(a, b, r) += 1;
        }
      }
    }
  }
}
//
// void houghCircleDirection(ThreadStruct thread, const cv::Mat bin, cv::Mat
// &acc, cv::Mat const &dirs,
//                           uchar th) {
//   for (int y = 0; y < bin.rows; y++) {
//     for (int x = 0; x < bin.cols; x++) {
//       if (bin.at<uchar>(y, x) < th)
//         continue;
//
//       float theta = dirs.at<float>(y, x);
//
//       incLineDir(acc, theta, x, y, max_a, max_b);
//       incLineDir(acc, theta, x, y, max_a, max_b, -1);
//     }
//   }
// }

HoughCirclesResult HoughCirclesFromBinMT(const int nb_threads,
                                         const cv::Mat &img, uchar binThresh,
                                         uchar circleThresh) {
  HoughCirclesResult result;

  int offset = img.cols / nb_threads;

  int max_r = std::max(
      img.cols,
      std::max(img.rows, (int)sqrt(img.cols * img.cols + img.rows * img.rows)) +
          1);
  int max_a = img.cols;
  int max_b = img.rows;

  int sizes[]{max_a, max_b, max_r};

  cv::Mat accumulator = cv::Mat::zeros(3, sizes, CV_32F);
  MutexMatrix3D mutexes = MutexMatrix3D(max_a, max_b, max_r);

  std::vector<std::thread> threads;
  for (int i = 0; i < nb_threads; i++) {
    int max_cols = offset * (i + 1) < img.cols ? offset * (i + 1) : img.cols;

    cv::Point range_cols = {offset * i, max_cols};

    ThreadStruct data(std::ref(mutexes), accumulator, range_cols, max_a, max_b);

    threads.push_back(
        std::thread([&]() { circle_accumulator(data, img, binThresh); }));

    std::cout << "Thread " << i << "started ! " << std::endl;
  }

  for (int i = 0; i < nb_threads; i++) {
    std::cout << "Thread " << i << "joined ! " << std::endl;
    threads[i].join();
  }

  result.edges = img;
  result.circles = img;

  // result.final = img;//cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  std::cout << "end" << std::endl;

  drawCircles(img, result.circles, accumulator, circleThresh);

  return result;
}
//
// HoughCirclesResult HoughCirclesFromGradMT(const int nb_threads, uchar sh,
//                                           uchar sb, const cv::Mat &img,
//                                           uchar binThresh, uchar
//                                           circleThresh, Dimension dim =
//                                           MULTI_DIM) {
//   HoughCirclesResult result;
//
//   // ------------ grad ----------
//   //
//
//   cv::Mat h(3, 3, CV_32F, const_cast<float *>(kernel::kirsch));
//
//   std::vector<cv::Mat> grads;
//   grads = computeGradients(img, h, dim);
//
//   cv::Mat mags, uc_mags, dirs, fnl;
//   if (dim == 2) {
//     magnitudeBD(grads, mags, dirs);
//   } else {
//     magnitudeMD(grads, mags, dirs);
//   }
//
//   double max;
//   cv::minMaxLoc(mags, nullptr, &max);
//   mags.convertTo(uc_mags, CV_8UC1);
//   hysteresis(uc_mags, fnl, sh, sb);
//
//   //
//   // ------------------------------
//
//   int offset = img.cols / nb_threads;
//
//   int max_r = std::min(img.cols, img.rows);
//   int max_a = img.cols;
//   int max_b = img.rows;
//
//   int sizes[]{max_a, max_b, max_r};
//
//   cv::Mat accumulator = cv::Mat::zeros(3, sizes, CV_32F);
//   MutexMatrix3D mutexes = MutexMatrix3D(max_a, max_b, max_r);
//
//   std::vector<std::thread> threads;
//   for (int i = 0; i < nb_threads; i++) {
//     int max_cols = offset * (i + 1) < img.cols ? offset * (i + 1) : img.cols;
//
//     cv::Point range_cols = {offset * i, max_cols};
//
//     ThreadStruct data(std::ref(mutexes), accumulator, range_cols, max_b);
//
//     threads.push_back(
//         std::thread([&]() { circle_accumulator(data, img, binThresh); }));
//
//     std::cout << "Thread " << i << "started ! " << std::endl;
//   }
//
//   for (int i = 0; i < nb_threads; i++) {
//     std::cout << "Thread " << i << "joined ! " << std::endl;
//     threads[i].join();
//   }
//
//   result.edges = fnl;
//   result.circles = cv::Mat::zeros(img.size(), CV_8UC3);
//
//   // result.final = img;//cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
//   std::cout << "end" << std::endl;
//
//   drawCircles(img, result.circles, accumulator, circleThresh);
//
//   return result;
// }
