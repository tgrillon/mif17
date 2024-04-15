#pragma once
#include "gradient.hpp"
#include "hough.hpp"
#include "kernel.hpp"

struct HoughResult {
  cv::Mat img, flt, edg, acc, shapes;
};

void processGradient(
  const cv::Mat &img,
  cv::Mat & fnl, 
  cv::Mat & dirs,
  int kernel, 
  uchar sh, uchar sb,
  Dimension dim) 
{
  float* k;
  switch(kernel) {
  case 0:
    k = const_cast<float *>(kernel::prewitt);
    break; 
  case 1:
    k = const_cast<float *>(kernel::sobel);
    break; 
  case 2:
    k = const_cast<float *>(kernel::kirsch);
    break; 
  }
  cv::Mat h(3, 3, CV_32F, k);

  std::vector<cv::Mat> grads;
  grads = computeGradients(img, h, dim);

  cv::Mat mags, uc_mags;
  if (dim == Dimension::TWO_DIM) {
    magnitudeBD(grads, mags, dirs);
  } else {
    magnitudeMD(grads, mags, dirs);
  }

  double max;
  minmax(mags, nullptr, &max);
  mags.convertTo(uc_mags, CV_8UC1);
  hysteresis(uc_mags, fnl, sh, sb);
}

HoughResult houghLinesFromBin(
  cv::Mat const& img, 
  cv::Mat const& flt,
  cv::Mat const& edges,
  int thickness,
  uchar bin_thresh,
  float line_thresh,
  float grouping_thresh,
  bool use_dirs, 
  bool canny = false,
  cv::Mat dirs = cv::Mat()) 
{
  HoughResult result;
  cv::Mat acc;

  result.img = img.clone();
  result.flt = flt.clone();
  result.edg = edges.clone();
  if (canny) {
    cv::Canny(result.flt, result.edg, 200, 50);
  }

  if (dirs.empty() || !use_dirs) {
    houghLines(result.edg, acc, bin_thresh);
  } else {
    houghLines(result.edg, acc, dirs, bin_thresh);
  }

  double max;
  minmax(acc, nullptr, &max);
  acc.convertTo(result.acc, CV_8UC1, 255 / max);
  cv::cvtColor(result.acc, result.acc, cv::COLOR_GRAY2BGR);

  auto lines = getLines(acc, line_thresh, grouping_thresh);
  drawLocalExtrema(lines, result.acc);

  cv::Mat temp = result.img.clone();
  cv::cvtColor(temp, result.shapes, cv::COLOR_GRAY2BGR);
  drawLines(lines, result.shapes, thickness);

  return result;
}

HoughResult houghLinesWithGradient(
  cv::Mat const& img,
  cv::Mat const& flt,
  int kernel, 
  int thickness,
  uchar sh, uchar sb,
  uchar bin_thresh,
  float line_thresh,
  float grouping_thresh,
  bool use_dirs,
  Dimension dim) 
{
  // Unused because blur var wasn't used anywhere
  // int i = 11;
  // cv::bilateralFilter(img, blur, i, i * 2, i / 2);
  cv::Mat dirs, fnl;
  processGradient(flt, fnl, dirs, kernel, sh, sb, dim);

  return houghLinesFromBin(
    img, flt, fnl, thickness, bin_thresh, line_thresh, grouping_thresh, use_dirs, false, dirs
  );
}

HoughResult houghCirclesFromBin(
  cv::Mat const& img, 
  cv::Mat const& flt, 
  cv::Mat const& edges,
  int thickness,
  uchar bin_thresh,
  float circle_thresh,
  float grouping_thresh,
  bool use_dirs, 
  bool canny = false,
  cv::Mat dirs = cv::Mat()) 
{
  HoughResult result;
  cv::Mat acc;

  result.img = img.clone();
  result.flt = flt.clone();
  result.edg = edges.clone();
  if (canny) {
    cv::Canny(result.flt, result.edg, 200, 50);
  }

  if (dirs.empty() || !use_dirs) {
    houghCircles(result.edg, acc, bin_thresh);
  } else {
    houghCircles(result.edg, acc, dirs, bin_thresh);
  }

  auto circles = getCircles(acc, circle_thresh, grouping_thresh);

  cv::Mat temp = result.img.clone();
  cv::cvtColor(temp, result.shapes, cv::COLOR_GRAY2BGR);
  drawCircles(circles, result.shapes, thickness);

  return result;
}

HoughResult houghCirclesWithGradient(
  cv::Mat const& img, 
  cv::Mat const& flt, 
  int kernel, 
  int thickness,
  uchar sh, uchar sb, 
  uchar bin_thresh,
  float circle_thresh,
  float grouping_thresh,
  bool use_dirs,
  Dimension dim = MULTI_DIM) 
{
  cv::Mat dirs, fnl;
  processGradient(flt, fnl, dirs, kernel, sh, sb, dim);

  return houghCirclesFromBin(
    img, flt, fnl, thickness, bin_thresh, circle_thresh, grouping_thresh, use_dirs, false, dirs
  );
}
