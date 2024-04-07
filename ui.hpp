#pragma once
#include "applications.hpp"
#include "gradient.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

class Viewer {
  virtual void process() = 0;
  virtual void configure_window() = 0;
  virtual void window() = 0;

public:
  void show() {
    // process();
    configure_window();
  };
};

class DemoHoughLinesBase : public Viewer {
protected:
  HoughLinesResult result;
  cv::Mat img;

public:
  DemoHoughLinesBase(const cv::Mat &img) : img(img) {}

  void window() override {
    cv::imshow("img", img);
    cv::imshow("intersect", result.inter);
    cv::imshow("regimg", result.regimg);
    cv::imshow("hough", result.hough_lines);
    cv::imshow("final", result.final);
  }
};

// class DemoHoughLinesBin : public DemoHoughLinesBase {
// public:
//   DemoHoughLinesBin(const cv::Mat &img) : DemoHoughLinesBase(img) {}
//
//   void process() override { houghLinesFromBin(result, img); }
//
//   void configure_window() override {}
// };
//
class DemoHoughLinesGrad : public DemoHoughLinesBase {
private:
  int hough_thresh = 0, multi_dim = 1, compute = 0;
  int regThresh1 = 4, regThresh2 = 1;

public:
  DemoHoughLinesGrad(const cv::Mat &img) : DemoHoughLinesBase(img) {}

  void process() override {
    if (compute) {
      float regThresh1 = ((float)this->regThresh1) / 100;
      float regThresh2 = ((float)this->regThresh2) / 100;
      result = houghLinesWithGradient(img, hough_thresh,
                                      multi_dim ? Dimension::MULTI_DIM
                                                : Dimension::TWO_DIM,
                                      regThresh1, regThresh2);
      window();
    }
  }

  void configure_window() override {
    cv::namedWindow("base");

    auto compute_fn = [](int, void *user) {
      DemoHoughLinesGrad *bthis = static_cast<DemoHoughLinesGrad *>(user);
      bthis->process();
    };

    cv::createTrackbar("hough_thresh", "base", &hough_thresh, 255, compute_fn,
                       this);
    cv::createTrackbar("grad_multi_dim", "base", &multi_dim, 1, compute_fn,
                       this);
    cv::createTrackbar("ref_thresh_initial", "base", &regThresh1, 100,
                       compute_fn, this);
    cv::createTrackbar("reg_thresh_neigh", "base", &regThresh2, 100, compute_fn,
                       this);
    cv::createTrackbar("Compute on changes:", "base", &compute, 1, compute_fn,
                       this);

    cv::waitKey();
  }
};
