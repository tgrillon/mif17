#pragma once
#include "applications.hpp"
#include "gradient.hpp"
#include "multithreading.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include <opencv2/core.hpp>
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
    cv::imshow("edges", result.edges);
    cv::imshow("regimg", result.regimg);
    cv::imshow("hough", result.lines);
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
  int bin_thresh = 255, multi_dim = 1, compute = 0, invert = 0;
  int regThresh1 = 50, regThresh2 = 1, grad = 1;
  int sh = 24, sb = 4;

public:
  DemoHoughLinesGrad(const cv::Mat &img) : DemoHoughLinesBase(img) {}

  void process() override {
    TimeFunction time([&]() {
      float regThresh1 = ((float)this->regThresh1) / 100;
      float regThresh2 = ((float)this->regThresh2) / 100;
      cv::Mat img;
      if (invert && !grad)
        cv::bitwise_not(this->img, img);
      else
        img = this->img;

      if (grad) {
        result = houghLinesWithGradient(img, sh, sb, bin_thresh,
                                        multi_dim ? Dimension::MULTI_DIM
                                                  : Dimension::TWO_DIM,
                                        regThresh1, regThresh2);
      } else {
        cv::Mat edge;
        cv::Canny(img, edge, 200, 50);
        result = houghLinesFromBin(img, bin_thresh, regThresh1, regThresh2);
      }
    });

    time.print("process");
    window();
  }

  void configure_window() override {
    std::string title = "Hough Line Demo Config";
    cv::namedWindow(title);

    auto compute_fn = [](int, void *user) {
      DemoHoughLinesGrad *bthis = static_cast<DemoHoughLinesGrad *>(user);
      if (bthis->compute) {
        bthis->process();
      }
    };

    cv::createTrackbar("bin_thresh ", title, &bin_thresh, 255, compute_fn,
                       this);
    cv::createTrackbar("bin only / with grad", title, &grad, 1, compute_fn,
                       this);
    cv::createTrackbar("bin_only_invert_input", title, &invert, 1, compute_fn,
                       this);
    cv::createTrackbar("grad_multi_dim", title, &multi_dim, 1, compute_fn,
                       this);
    cv::createTrackbar("reg_thresh_initial", title, &regThresh1, 100,
                       compute_fn, this);
    cv::createTrackbar("reg_thresh_neigh", title, &regThresh2, 100, compute_fn,
                       this);
    cv::createTrackbar("sh hysteresis", title, &sh, 255, compute_fn, this);
    cv::createTrackbar("sb hysteresis", title, &sb, 255, compute_fn, this);
    cv::createTrackbar("Compute on changes (key 'R' = compute)", title,
                       &compute, 1, compute_fn, this);

    while (true) {
      switch (cv::waitKey()) {
      case 'r':
        this->process();
        std::cout << "Done!" << std::endl;
        break;

      case 27:
        return;
      }
    }
  }
};

class DemoHoughCirclesBase : public Viewer {
protected:
  HoughCirclesResult result;
  cv::Mat img;

public:
  DemoHoughCirclesBase(const cv::Mat &img) : img(img) {}

  void window() override {
    cv::imshow("img", img);
    cv::imshow("edges", result.edges);
    cv::imshow("hough", result.circles);
  }
};

class DemoHoughCirclesGrad : public DemoHoughCirclesBase {
private:
  int bin_thresh = 255, circle_thresh = 5;
  int multi_dim = 1, compute = 0, invert = 0, grad = 1;
  int sh = 24, sb = 4;

public:
  DemoHoughCirclesGrad(const cv::Mat &img) : DemoHoughCirclesBase(img) {}

  void process() override {
    TimeFunction time([&]() {
      cv::Mat img;
      if (invert && !grad)
        cv::bitwise_not(this->img, img);
      else
        img = this->img;

      if (grad) {
        result = houghCirclesWithGradient(
            img, sh, sb, bin_thresh, circle_thresh,
            multi_dim ? Dimension::MULTI_DIM : Dimension::TWO_DIM);
      } else {
        // cv::Mat edge;
        // cv::Canny(img, edge, 200, 50);
        result = houghCirclesFromBin(img, bin_thresh, circle_thresh);
        // result = HoughCirclesFromBinMT(4, img, bin_thresh, circle_thresh);
      }
    });
    time.print("process");
    window();
  }

  void configure_window() override {
    std::string title = "Hough Circle Demo Config";
    cv::namedWindow(title);

    auto compute_fn = [](int, void *user) {
      DemoHoughCirclesGrad *bthis = static_cast<DemoHoughCirclesGrad *>(user);
      if (bthis->compute) {
        bthis->process();
      }
    };

    cv::createTrackbar("bin_thresh ", title, &bin_thresh, 255, compute_fn,
                       this);
    cv::createTrackbar("bin only / with grad", title, &grad, 1, compute_fn,
                       this);
    cv::createTrackbar("bin_only_invert_input", title, &invert, 1, compute_fn,
                       this);
    cv::createTrackbar("grad_multi_dim", title, &multi_dim, 1, compute_fn,
                       this);
    cv::createTrackbar("circle thresh", title, &circle_thresh, 100, compute_fn,
                       this);
    cv::createTrackbar("sh hysteresis", title, &sh, 255, compute_fn, this);
    cv::createTrackbar("sb hysteresis", title, &sb, 255, compute_fn, this);
    cv::createTrackbar("Compute on changes (key 'R' = compute)", title,
                       &compute, 1, compute_fn, this);

    while (true) {
      switch (cv::waitKey()) {
      case 'r':
        this->process();
        std::cout << "Done!" << std::endl;
        break;

      case 27:
        return;
      }
    }
  }
};
