#pragma once
#include "opencv2/imgproc.hpp"

namespace kernel
{
    const float prewitt[9] = {
        -1.f/3, 0, 1.f/3,
        -1.f/3, 0, 1.f/3,
        -1.f/3, 0, 1.f/3
    };

    const float sobel[9] = {
        -1.f/4, 0, 1.f/4,
        -2.f/4, 0, 2.f/4,
        -1.f/4, 0, 1.f/4
    };

    const float kirsch[9] = {
        -3.f/15, -3.f/15, 5.f/15,
        -3.f/15,  0,      5.f/15,
        -3.f/15, -3.f/15, 5.f/15
    };

    const float gaussian[9] = {
        1.f/16, 1.f/8, 1.f/16,
        1.f/8,  1.f/4, 1.f/8,
        1.f/16, 1.f/8, 1.f/16
    };

    cv::Mat rotate(cv::Mat const& h)
    {
        cv::Mat rt = h.clone();
        float bff = rt.at<float>(0, 0);
        rt.at<float>(0, 0) = rt.at<float>(1, 0);
        rt.at<float>(1, 0) = rt.at<float>(2, 0);
        rt.at<float>(2, 0) = rt.at<float>(2, 1);
        rt.at<float>(2, 1) = rt.at<float>(2, 2);
        rt.at<float>(2, 2) = rt.at<float>(1, 2);
        rt.at<float>(1, 2) = rt.at<float>(0, 2);
        rt.at<float>(0, 2) = rt.at<float>(0, 1);
        rt.at<float>(0, 1) = rt.at<float>(0, 0);
        rt.at<float>(0, 1) = bff;

        return rt;
    }
}
