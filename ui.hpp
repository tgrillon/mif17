#pragma once
#include "applications.hpp"
#include "gradient.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

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
  HoughLinesResult m_result;
  cv::Mat m_img;

public:
  DemoHoughLinesBase(const cv::Mat &img) : m_img(img) {}

  void window() override {
    cv::imshow("img", m_img);
    cv::imshow("intersect", m_result.inter);
    cv::imshow("edges", m_result.edges);
    cv::imshow("regimg", m_result.regimg);
    cv::imshow("hough", m_result.lines);
    cv::imshow("final", m_result.final);
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
  int m_bin_thresh  = 255, m_multi_dim = 1, m_compute = 0, m_invert = 0;
  int m_line_thresh = 50, m_grouping_thresh = 1, m_grad = 1;
  int m_sh = 24, m_sb = 4;
  int m_use_dirs = true;

public:
  DemoHoughLinesGrad(const cv::Mat &img) : DemoHoughLinesBase(img) {}

  void process() override {
    float line_thresh = ((float)this->m_line_thresh) * 0.01;
    float grouping_thresh = ((float)this->m_grouping_thresh) * 0.01;
    cv::Mat img;
    if (m_invert && !m_grad)
      cv::bitwise_not(this->m_img, img);
    else
      img = this->m_img;

    if (m_grad) {
      m_result = houghLinesWithGradient(img, m_sh, m_sb, m_bin_thresh ,
                                      line_thresh, grouping_thresh,
                                      m_use_dirs,
                                      m_multi_dim ? Dimension::MULTI_DIM
                                                : Dimension::TWO_DIM);
    } else {
      cv::Mat edge;
      cv::Canny(img, edge, 200, 50);
      m_result = houghLinesFromBin(img, m_bin_thresh, line_thresh, grouping_thresh);
    }
    window();
  }

  void configure_window() override {
    std::string title = "Hough Lines Detection Configs";
    cv::namedWindow(title);

    auto compute_fn = [](int, void *user) {
      DemoHoughLinesGrad *bthis = static_cast<DemoHoughLinesGrad *>(user);
      if (bthis->m_compute) {
        bthis->process();
      }
    };

    cv::createTrackbar("Edge detection threshold ", title, &m_bin_thresh , 255, compute_fn,
                       this);
    cv::createTrackbar("With gradient ? no -> 0 | yes -> 1 ", title, &m_grad, 1, compute_fn,
                       this);
    cv::createTrackbar("Bidirectionnal -> 0 | Multidirectionnal -> 1", title, &m_multi_dim, 1, compute_fn,
                       this);
    cv::createTrackbar("Use direction in computation", title, &m_use_dirs, 1, compute_fn,
                      this);
    cv::createTrackbar("Invert binary image", title, &m_invert, 1, compute_fn,
                       this);
    cv::createTrackbar("Line detection threshold", title, &m_line_thresh, 100,
                       compute_fn, this);
    cv::createTrackbar("Grouping threshold", title, &m_grouping_thresh, 100, compute_fn,
                       this);
    cv::createTrackbar("Hysteresis : Upper bound (sh)", title, &m_sh, 255, compute_fn,
                      this);
    cv::createTrackbar("Hysteresis : Lower bound (sb)", title, &m_sb, 255, compute_fn,
                      this);
    cv::createTrackbar("Compute with key 'R'", title,
                       &m_compute, 1, compute_fn, this);

    while (true) {
      switch (cv::waitKey()) {
      case 'r':
        std::cout << "\x1B[2J\x1B[H";
        std::cout << "Computing..." << std::endl;
        MEASURE_TIME(this->process());
        std::cout << "Done." << std::endl;
        break;
      case 27:
        return;
      }
    }
  }
};

class DemoHoughCirclesBase : public Viewer {
protected:
  HoughCirclesResult m_result;
  cv::Mat m_img;

public:
  DemoHoughCirclesBase(const cv::Mat &img) : m_img(img) {}

  void window() override {
    cv::imshow("img", m_img);
    cv::imshow("edges", m_result.edges);
    cv::imshow("hough", m_result.circles);
  }
};

class DemoHoughCirclesGrad : public DemoHoughCirclesBase {
private:
  int m_bin_thresh  = 255, m_circle_thresh = 50, m_grouping_thesh = 1;
  int m_multi_dim = 1, m_compute = 0, m_invert = 0, m_grad = 1;
  int m_sh = 24, m_sb = 4;
  int m_use_dirs = 1;

public:
  DemoHoughCirclesGrad(const cv::Mat &img) : DemoHoughCirclesBase(img) {}

  void process() override {
    cv::Mat img;
    float circle_thresh = this->m_circle_thresh;
    if (m_invert && !m_grad)
      cv::bitwise_not(this->m_img, img);
    else
      img = this->m_img;

    if (m_grad) {
      m_result = houghCirclesWithGradient(img, m_sh, m_sb, 
                                      m_bin_thresh, circle_thresh,
                                      m_use_dirs, 
                                      m_multi_dim ? Dimension::MULTI_DIM
                                                : Dimension::TWO_DIM
                                     );
    } else {
      // cv::Mat edge;
      // cv::Canny(img, edge, 200, 50);
      m_result = houghCirclesFromBin(img, m_bin_thresh, circle_thresh);
    }
    window();
  }

  void configure_window() override {
    std::string title = "Hough Circle Demo Config";
    cv::namedWindow(title);

    auto compute_fn = [](int, void *user) {
      DemoHoughCirclesGrad *bthis = static_cast<DemoHoughCirclesGrad *>(user);
      if (bthis->m_compute) {
        bthis->process();
      }
    };

    cv::createTrackbar("Edge detection threshold ", title, &m_bin_thresh , 255, compute_fn,
                       this);
    cv::createTrackbar("With gradient ? no -> 0 | yes -> 1 ", title, &m_grad, 1, compute_fn,
                       this);
    cv::createTrackbar("Bidirectionnal -> 0 | Multidirectionnal -> 1", title, &m_multi_dim, 1, compute_fn,
                       this);
    cv::createTrackbar("Use direction in computation", title, &m_use_dirs, 1, compute_fn,
                      this);
    cv::createTrackbar("Invert binary image", title, &m_invert, 1, compute_fn,
                       this);
    cv::createTrackbar("Circle detection threshold", title, &m_circle_thresh, 100, compute_fn,
                       this);
    cv::createTrackbar("Grouping threshold", title, &m_grouping_thesh, 100, compute_fn,
                       this);
    cv::createTrackbar("Hysteresis : Upper bound (sb)", title, &m_sh, 255, compute_fn,
                      this);
    cv::createTrackbar("Hysteresis : Lower bound (sb)", title, &m_sb, 255, compute_fn,
                      this);
    cv::createTrackbar("Compute with key 'R'", title,
                       &m_compute, 1, compute_fn, this);

    while (true) {
      switch (cv::waitKey()) {
      case 'r':
        std::cout << "\x1B[2J\x1B[H";
        std::cout << "Computing..." << std::endl;
        MEASURE_TIME(this->process());
        std::cout << "Done." << std::endl;
        break;

      case 27:
        return;
      }
    }
  }
};