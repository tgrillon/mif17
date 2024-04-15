#pragma once
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include "kernel.hpp"

enum Dimension {
    TWO_DIM=2,
    MULTI_DIM=4
};

std::vector<cv::Mat> computeGradients(const cv::Mat& src, const cv::Mat& h, Dimension dim=MULTI_DIM)
{
    assert(h.rows == 3 && h.cols == 3);
    int height = src.rows;
    int width = src.cols;

    std::vector<cv::Mat> krns(dim);
    krns[0] = h;
    for (int i = 1; i < dim; ++i) {
        krns[i] = kernel::rotate(krns[i-1]);
        if (dim == 2) {
            krns[i] = kernel::rotate(krns[i]);
        }
    }

    std::vector<cv::Mat> grads(dim);
    for (int k = 0; k < dim; ++k) {
        grads[k] = cv::Mat::zeros(src.size(), CV_32F);
    }

    for (int r = 1; r < height - 1; ++r) {
        for (int c = 1; c < width - 1; ++c) {
            for (int k = 0; k < dim; ++k) {
                float val = convolution(src, krns[k], c, r);
                grads[k].at<float>(r, c) = val;
            }
        }
    }

    return grads;
}

void magnitudeBD(std::vector<cv::Mat> const& grads, cv::Mat & mags, cv::Mat & dirs)
{
    mags = cv::Mat::zeros(grads[0].size(), CV_32F);
    dirs = cv::Mat::zeros(grads[0].size(), CV_32F);
    int rows = grads[0].rows;
    int cols = grads[0].cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            float gx = grads[0].at<float>(r, c);
            float gy = grads[1].at<float>(r, c);
            float mag = sqrt(gx*gx+gy*gy);
            float dir = atan2(gy, gx);

            mags.at<float>(r, c) = mag;
            dirs.at<float>(r, c) = dir;
        }
    }
}

void magnitudeMD(std::vector<cv::Mat> const& grads, cv::Mat & mags, cv::Mat & dirs)
{
    mags = cv::Mat::zeros(grads[0].size(), CV_32F);
    dirs = cv::Mat::zeros(grads[0].size(), CV_32F);
    int rows = grads[0].rows;
    int cols = grads[0].cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            float sup = abs(grads[0].at<float>(r, c));
            float dir = 0.f;
            for (int k = 1; k < grads.size(); ++k) {
                float tmp = abs(grads[k].at<float>(r, c));
                if (tmp > sup) {
                    sup = tmp;
                    dir = k;
                }
            }
            float val = sup;
            mags.at<float>(r, c) = val;
            dirs.at<float>(r, c) = dir*M_PI_4;
        }
    }
}

bool checkNeighbors(cv::Mat const& img, unsigned int r, unsigned int c) 
{
    for (int i = -1; i < 1; ++i) {
        for (int j = -1; j < 1; ++j) {
            if (i == 0 && j == 0) continue;
            if (img.at<uchar>(r-i,c-j) > 0) 
                return true;
        }
    }

    return false;
}

void hysteresis(cv::Mat const& src, cv::Mat & dest, uchar sh, uchar sb) 
{
    assert(src.type() == CV_8UC1);
    dest = cv::Mat::zeros(src.size(), src.type());
    int rows = src.rows;
    int cols = src.cols;

    // Premier parcours
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            if (src.at<uchar>(r, c) > sh) {
                dest.at<uchar>(r, c) = 255;
            }
        }
    }

    // Second parcour
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            uchar val = src.at<uchar>(r, c);
            if (val <= sh && val > sb) {
                if (checkNeighbors(dest, r, c)) {
                    dest.at<uchar>(r, c) = 255;
                }
            }
        }
    }
}

void direction(cv::Mat const& mags, cv::Mat & dest, cv::Mat const& dirs)
{
    assert(mags.type() == CV_8UC1);
    dest = cv::Mat::zeros(mags.size(), CV_8UC3);
    int rows = mags.rows;
    int cols = mags.cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            uchar mg = mags.at<uchar>(r, c);
            float dir = dirs.at<float>(r, c)*100.f;
            dest.at<cv::Vec3b>(r, c) = {mg, 0, 0};
            if (dir > 0) {
                if (dir < 80) {
                    dest.at<cv::Vec3b>(r, c) = {0, mg, 0};
                } else if (dir < 160) {
                    dest.at<cv::Vec3b>(r, c) = {0, 0, mg};
                } else {
                    dest.at<cv::Vec3b>(r, c) = {0, mg, mg};
                }
            }
        }
    }
}
