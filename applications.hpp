#pragma once
#include "gradient.hpp"
#include "hough.hpp"
#include "kernel.hpp"
struct HoughLinesResult {
  cv::Mat inter, edges, regimg, lines, final;
};

struct HoughCirclesResult {
  cv::Mat inter, edges, circles, final;
};

HoughLinesResult houghLinesFromBin(const cv::Mat &img,
                                    uchar binThresh  = 170,
                                   float regthresh1 = 0.4f,
                                   float regThresh2 = 0.1f,
                                   cv::Mat dirs = cv::Mat()) 
{
  HoughLinesResult result;
  cv::Mat acc;

  if (dirs.empty())
    houghLines(img, acc, binThresh);  
  else 
    houghLines(img, acc, dirs, binThresh);  

  result.lines = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  result.regimg = cv::Mat::zeros(acc.rows, acc.cols, CV_8UC3);
  result.final = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  result.edges = img;

  auto lines = getLines(acc, regthresh1, regThresh2);
  drawLocalExtrema(lines, result.regimg);
  drawLines(lines, result.lines);

  double min, max;
  minmax(acc, &min, &max);
  acc.convertTo(result.inter, CV_8UC1, 255 / max);

  intersectImg(img, result.lines, result.final);

  return result;
}

HoughLinesResult houghLinesWithGradient(const cv::Mat &img,
                                        uchar sh, uchar sb,
                                        uchar binThresh  = 170,
                                        Dimension dim = MULTI_DIM,
                                        float regthresh1 = 0.4f,
                                        float regThresh2 = 0.1f) 
{
  // Unused because blur var wasn't used anywhere
  // int i = 11;
  // cv::bilateralFilter(img, blur, i, i * 2, i / 2);
  cv::Mat h(3, 3, CV_32F, const_cast<float *>(kernel::kirsch));

  std::vector<cv::Mat> grads;
  grads = computeGradients(img, h, dim);

  cv::Mat mags, uc_mags, dirs, fnl;
  if (dim==2) {
    magnitudeBD(grads, mags, dirs);
  } else {
    magnitudeMD(grads, mags, dirs);
  }

  double min, max;
  minmax(mags, &min, &max);
  mags.convertTo(uc_mags, CV_8UC1);
  hysteresis(uc_mags, fnl, sh, sb);

  return houghLinesFromBin(fnl, binThresh , regthresh1, regThresh2, dirs);
}

HoughCirclesResult houghCirclesFromBin(const cv::Mat &img,
                                      uchar binThresh,
                                      uchar circleThresh) 
{
  HoughCirclesResult result;
  cv::Mat acc;

  houghCircle(img, acc, binThresh);
  result.edges = img;
  result.circles = img;

  // result.final = img;//cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  std::cout << "end" << std::endl;

  drawCircles(img, result.circles, acc, circleThresh);

  return result;
}

HoughCirclesResult houghCirclesWithGradient(const cv::Mat &img,
                                        uchar sh, uchar sb,
                                        uchar binThresh  = 170,
                                        uchar circleThresh = 5,
                                        Dimension dim = MULTI_DIM) 
{
  cv::Mat h(3, 3, CV_32F, const_cast<float *>(kernel::kirsch));

  std::vector<cv::Mat> grads;
  grads = computeGradients(img, h, dim);

  cv::Mat mags, uc_mags, dirs, fnl;
  if (dim==2) {
    magnitudeBD(grads, mags, dirs);
  } else {
    magnitudeMD(grads, mags, dirs);
  }

  double max;
  cv::minMaxLoc(mags, nullptr, &max);
  mags.convertTo(uc_mags, CV_8UC1);
  hysteresis(uc_mags, fnl, sh, sb);

  HoughCirclesResult result;
  cv::Mat acc;

  houghCircleDirection(fnl, acc, dirs, binThresh );

  result.circles = cv::Mat::zeros(img.size(), CV_8UC3);

  // for (int r = 0; r < img.rows; ++r) {
  //   for (int c = 0; c < img.cols; ++c) {
  //     if (fnl.at<uchar>(r, c) < 255) continue;
  //     float theta = dirs.at<float>(r, c);
  //     float a = cos(theta);
  //     float b = sin(theta);

  //     int x1 = c + 50*a;
  //     int y1 = r + 50*b;
  //     cv::line(result.circles, {c, r}, {x1, y1}, cv::Scalar(255, 0, 0), 1, cv::LINE_8); 

  //     int x2 = c - 50*a;
  //     int y2 = r - 50*b;
  //     cv::line(result.circles, {c, r}, {x2, y2}, cv::Scalar(255, 0, 0), 1, cv::LINE_8); 
  //   }
  // }

  // result.final = img;//cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  result.edges = fnl;

  drawCircles(img, result.circles, acc, circleThresh);

  return result;
}