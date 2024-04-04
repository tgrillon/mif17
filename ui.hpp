#pragma once
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include "applications.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>


class Viewer {
    virtual void process();
    virtual void configure_window();
    virtual void window();
public:
    void show() {
        process();
        configure_window();
        window();
    };
};

class DemoHoughLinesBase : public Viewer{ 
protected:
  HoughLinesResult result;
  cv::Mat img;
public: 
    DemoHoughLinesBase(const cv::Mat& img)
        : img(img) {} 

    void window() override {
        cv::imshow("img", img);
        cv::imshow("intersect", result.inter);
        cv::imshow("regimg", result.regimg);
        cv::imshow("hough", result.hough_lines);
        cv::imshow("final", result.final);
        cv::waitKey(0);
    }
};

class DemoHoughLinesBin : public DemoHoughLinesBase{ 
public: 
    DemoHoughLinesBin(const cv::Mat& img)
        : DemoHoughLinesBase(img) {} 

    void process() override {
      houghLinesFromBin(result, img);
    }

    void configure_window() override {

    }
};
