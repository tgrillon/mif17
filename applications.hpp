#pragma once
#include "gradient.hpp"
#include "hough.hpp"
#include "kernel.hpp"
struct HoughLinesResult {
  cv::Mat inter, regimg, hough_lines, final;
};

void houghLinesFromBin(HoughLinesResult &result, const cv::Mat &img,
                       uchar houghThresh = 170) {
  cv::Mat intersect;
  houghLines(img, intersect, houghThresh);

  result.hough_lines = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  result.regimg = cv::Mat::zeros(intersect.rows, intersect.cols, CV_8UC3);
  result.final = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

  auto regions = get_regions(intersect);
  draw_local_maximums(regions, result.regimg);
  draw_lines(regions, result.final);

  double min, max;
  minmax(intersect, &min, &max);
  intersect.convertTo(result.inter, CV_8UC1, 255 / max);
}

void houghLinesWithGradient(HoughLinesResult &result, const cv::Mat &img,
                            uchar houghThresh = 170,
                            Dimension dim = MULTI_DIM) {
  // Unused because blur var wasn't used anywhere
  // int i = 11;
  // cv::bilateralFilter(img, blur, i, i * 2, i / 2);

  cv::Mat h(3, 3, CV_32F, const_cast<float *>(kernel::kirsch));

  std::vector<cv::Mat> grds;
  grds = compute_gradients(img, h, dim);

  cv::Mat mgs, drs;
  magnitude(grds, mgs, drs);

  houghLinesFromBin(result, mgs, houghThresh);
}
