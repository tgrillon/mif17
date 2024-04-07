#pragma once
#include "applications.hpp"
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
    process();
    configure_window();
    window();
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
    cv::waitKey(0);
  }
};

class DemoHoughLinesBin : public DemoHoughLinesBase {
public:
  DemoHoughLinesBin(const cv::Mat &img) : DemoHoughLinesBase(img) {}

  void process() override { houghLinesFromBin(result, img); }

  void configure_window() override {}
};

class DemoHoughLinesGrad : public DemoHoughLinesBase {
public:
  DemoHoughLinesGrad(const cv::Mat &img) : DemoHoughLinesBase(img) {}

  void process() override { houghLinesWithGradient(result, img); }

  void configure_window() override {}
};
