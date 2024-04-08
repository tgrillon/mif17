#pragma once
#include "gradient.hpp"
#include "hough.hpp"
#include "kernel.hpp"
struct HoughLinesResult {
  cv::Mat inter, regimg, hough_lines, final;
};

HoughLinesResult houghLinesFromBin(const cv::Mat &img,
                                    uchar houghThresh = 170,
                                   float regthresh1 = 0.4f,
                                   float regThresh2 = 0.1f) 
{
  HoughLinesResult result;
  cv::Mat accumulator;

  houghLines(img, accumulator, houghThresh);  

  result.hough_lines = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  result.regimg = cv::Mat::zeros(accumulator.rows, accumulator.cols, CV_8UC3);
  result.final = img;//cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

  auto lines = getLines(accumulator, regthresh1, regThresh2);
  drawLocalExtrema(lines, result.regimg);
  drawLines(lines, result.hough_lines);

  double min, max;
  minmax(accumulator, &min, &max);
  accumulator.convertTo(result.inter, CV_8UC1, 255 / max);

  // intersectImg(img, result.hough_lines, result.final);

  return result;
}

HoughLinesResult houghLinesWithGradient(const cv::Mat &img,
                                        uchar sh, uchar sb,
                                        uchar houghThresh = 170,
                                        Dimension dim = MULTI_DIM,
                                        float regthresh1 = 0.4f,
                                        float regThresh2 = 0.1f) {
  // Unused because blur var wasn't used anywhere
  // int i = 11;
  // cv::bilateralFilter(img, blur, i, i * 2, i / 2);
  cv::Mat h(3, 3, CV_32F, const_cast<float *>(kernel::kirsch));

  std::vector<cv::Mat> grds;
  grds = computeGradients(img, h, dim);

  cv::Mat mgs, drs, fnl;
  magnitude(grds, mgs, drs);

  hysteresis(mgs, fnl, sh, sb);

  return houghLinesFromBin(fnl, houghThresh, regthresh1, regThresh2);
}
