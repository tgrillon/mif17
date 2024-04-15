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
  HoughResult m_result;
  cv::Mat m_img;

public:
  DemoHoughLinesBase(const cv::Mat &img) : m_img(img) {}

  void window() override {
    cv::imshow("Input image", m_result.img);
    cv::imshow("Filtered image", m_result.flt);
    cv::imshow("Edges", m_result.edg);
    cv::imshow("Accumulator", m_result.acc);
    cv::imshow("Final result", m_result.shapes);
  }
};

class DemoHoughLinesGrad : public DemoHoughLinesBase {
private:
  int m_multi_dim = 1, m_compute = 0, m_invert = 0, m_grad = 1;
  int m_bin_thresh  = 255, m_line_thresh = 50, m_grouping_thresh = 1;
  int m_sh = 24, m_sb = 4;
  int m_canny = 0, m_use_dirs = 1, m_kernel = 2;
  int m_thickness = 1;
  int m_bf_d = 27, m_bf_sigma_color = 27, m_bf_sigma_space = 27;

public:
  DemoHoughLinesGrad(const cv::Mat &img) : DemoHoughLinesBase(img) {}

  void process() override {
    float line_thresh = ((float)this->m_line_thresh) * 0.01;
    float grouping_thresh = ((float)this->m_grouping_thresh) * 0.01;
    cv::Mat img, gray, flt;
    if (m_invert && !m_grad)
      cv::bitwise_not(this->m_img, img);
    else
      img = this->m_img;

    cv::cvtColor(m_img, gray, cv::COLOR_BGR2GRAY);
    cv::bilateralFilter(gray, flt, m_bf_d, m_bf_sigma_color, m_bf_sigma_space);

    if (m_grad) {
      m_result = houghLinesWithGradient(
        m_img, flt, m_kernel, m_thickness, m_sh, m_sb, m_bin_thresh, line_thresh, grouping_thresh,
        m_use_dirs, m_multi_dim ? Dimension::MULTI_DIM : Dimension::TWO_DIM
      );
    } else {
      m_result = houghLinesFromBin(
        m_img, flt, img, m_thickness, m_bin_thresh, line_thresh, grouping_thresh, false, m_canny
      );
    }
    window();
  }

  void configure_window() override {
    std::string w_title = "[Configuration panel] Hough Line Detection";
    cv::namedWindow(w_title);

    auto compute_fn = [](int, void *user) {
      DemoHoughLinesGrad *bthis = static_cast<DemoHoughLinesGrad *>(user);
      if (bthis->m_compute) {
        bthis->process();
      }
    };

    cv::createTrackbar("[Input] Bilateral filter d", w_title, &m_bf_d, 255, compute_fn,
                       this);
    cv::createTrackbar("[Input] Bilateral filter sigma color", w_title, &m_bf_sigma_color, 255, compute_fn,
                       this);
    cv::createTrackbar("[Input] Bilateral filter sigma space", w_title, &m_bf_sigma_space, 255, compute_fn,
                       this);
    cv::createTrackbar("[Binary] Invert binary image", w_title, &m_invert, 1, compute_fn,
                       this);
    cv::createTrackbar("[Binary] Opencv edge detection", w_title, &m_canny, 1, compute_fn,
                       this);
    cv::createTrackbar("[Hough] Use gradient ? 0 : no  | 1 : yes ", w_title, &m_grad, 1, compute_fn,
                       this);
    cv::createTrackbar("[Gradient] Kernel (0: prewitt | 1: sobel | 2: kirsch)", w_title, &m_kernel , 2, compute_fn,
                       this);
    cv::createTrackbar("[Gradient] 0 : Bidirectionnal | 1 : Multidirectionnal", w_title, &m_multi_dim, 1, compute_fn,
                       this);
    cv::createTrackbar("[Gradient] Hysteresis : Upper bound (sh)", w_title, &m_sh, 255, compute_fn,
                      this);
    cv::createTrackbar("[Gradient] Hysteresis : Lower bound (sb)", w_title, &m_sb, 255, compute_fn,
                      this);
    cv::createTrackbar("[Hough + Gradient] Use direction in computation", w_title, &m_use_dirs, 1, compute_fn,
                      this);
    cv::createTrackbar("[Hough] Edge detection threshold ", w_title, &m_bin_thresh , 255, compute_fn,
                       this);
    cv::createTrackbar("[Hough] Line detection threshold (% of max)", w_title, &m_line_thresh, 100,
                       compute_fn, this);
    cv::createTrackbar("[Hough] Grouping threshold (% of max)", w_title, &m_grouping_thresh, 100, compute_fn,
                       this);
    cv::createTrackbar("[Hough] Shape thickness", w_title, &m_thickness, 10, compute_fn,
                       this);
    cv::createTrackbar("Compute with key 'R'", w_title,
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
  HoughResult m_result;
  cv::Mat m_img;

public:
  DemoHoughCirclesBase(const cv::Mat &img) : m_img(img) {}

  void window() override {
    cv::imshow("Input image", m_result.img);
    cv::imshow("Filtered image", m_result.flt);
    cv::imshow("Edges", m_result.edg);
    cv::imshow("Final result", m_result.shapes);
  }
};

class DemoHoughCirclesGrad : public DemoHoughCirclesBase {
private:
  int m_bin_thresh  = 255, m_circle_thresh = 50, m_grouping_thresh = 1;
  int m_multi_dim = 1, m_compute = 0, m_invert = 0, m_grad = 1;
  int m_sh = 24, m_sb = 4;
  int m_canny = 0, m_use_dirs = 1, m_kernel = 2;
  int m_thickness = 1;
  int m_bf_d = 27, m_bf_sigma_color = 27, m_bf_sigma_space = 27;

public:
  DemoHoughCirclesGrad(const cv::Mat &img) : DemoHoughCirclesBase(img) {}

  void process() override {
    cv::Mat img, gray, flt;
    float circle_thresh = this->m_circle_thresh * 0.01;
    float grouping_thresh = this->m_grouping_thresh * 0.01;
    if (m_invert && !m_grad)
      cv::bitwise_not(this->m_img, img);
    else
      img = this->m_img;

    cv::cvtColor(m_img, gray, cv::COLOR_BGR2GRAY);
    cv::bilateralFilter(gray, flt, m_bf_d, m_bf_sigma_color, m_bf_sigma_space);

    if (m_grad) {
      m_result = houghCirclesWithGradient(
        img, flt, m_kernel, m_thickness, m_sh, m_sb, m_bin_thresh, circle_thresh, grouping_thresh, 
        m_use_dirs, m_multi_dim ? Dimension::MULTI_DIM : Dimension::TWO_DIM
      );
    } else {
      m_result = houghCirclesFromBin(
        m_img, flt, img, m_thickness, m_bin_thresh, circle_thresh, grouping_thresh, false, m_canny
      );
    }
    window();
  }

  void configure_window() override {
    std::string w_title = "[Configuration panel] Hough Circle Detection";
    cv::namedWindow(w_title);

    auto compute_fn = [](int, void *user) {
      DemoHoughCirclesGrad *bthis = static_cast<DemoHoughCirclesGrad *>(user);
      if (bthis->m_compute) {
        bthis->process();
      }
    };

    cv::createTrackbar("[Input] Bilateral filter d", w_title, &m_bf_d, 255, compute_fn,
                       this);
    cv::createTrackbar("[Input] Bilateral filter sigma color", w_title, &m_bf_sigma_color, 255, compute_fn,
                       this);
    cv::createTrackbar("[Input] Bilateral filter sigma space", w_title, &m_bf_sigma_space, 255, compute_fn,
                       this);
    cv::createTrackbar("[Binary] Invert binary image", w_title, &m_invert, 1, compute_fn,
                       this);
    cv::createTrackbar("[Binary] Opencv edge detection", w_title, &m_canny, 1, compute_fn,
                       this);
    cv::createTrackbar("[Hough] Use gradient ? no -> 0 | yes -> 1 ", w_title, &m_grad, 1, compute_fn,
                       this);
    cv::createTrackbar("[Gradient] Bidirectionnal -> 0 | Multidirectionnal -> 1", w_title, &m_multi_dim, 1, compute_fn,
                       this);
    cv::createTrackbar("[Gradient] Kernel (0: prewitt | 1: sobel | 2: kirsch)", w_title, &m_kernel , 2, compute_fn,
                       this);
    cv::createTrackbar("[Gradient] Hysteresis : Upper bound (sb)", w_title, &m_sh, 255, compute_fn,
                      this);
    cv::createTrackbar("[Gradient] Hysteresis : Lower bound (sb)", w_title, &m_sb, 255, compute_fn,
                      this);
    cv::createTrackbar("[[Hough + Gradient] Use direction in computation", w_title, &m_use_dirs, 1, compute_fn,
                      this);
    cv::createTrackbar("[Hough] Edge detection threshold ", w_title, &m_bin_thresh , 255, compute_fn,
                       this);
    cv::createTrackbar("[Hough] Circle detection threshold", w_title, &m_circle_thresh, 100, compute_fn,
                       this);
    cv::createTrackbar("[Hough] Grouping threshold", w_title, &m_grouping_thresh, 100, compute_fn,
                       this);
    cv::createTrackbar("[Hough] Shape thickness", w_title, &m_thickness, 10, compute_fn,
                       this);
    cv::createTrackbar("Compute with key 'R'", w_title,
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
