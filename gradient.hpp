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

    std::vector<cv::Mat> grds(dim);
    for (int k = 0; k < dim; ++k) {
        grds[k] = cv::Mat::zeros(src.size(), src.type());
    }

    for (int r = 1; r < height - 1; ++r) {
        for (int c = 1; c < width - 1; ++c) {
            for (int k = 0; k < dim; ++k) {
                uchar val = cv::saturate_cast<uchar>(abs(convolution(src, krns[k], c, r)));
                grds[k].at<uchar>(r, c) = val;
            }
        }
    }

    return grds;
}

void magnitudeBD(std::vector<cv::Mat> const& grds, cv::Mat & mgs, cv::Mat & drs)
{
    mgs = cv::Mat::zeros(grds[0].size(), CV_32F);
    drs = cv::Mat::zeros(grds[0].size(), CV_32F);
    int rows = grds[0].rows;
    int cols = grds[0].cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            float gx = float(grds[0].at<uchar>(r, c));
            float gy = float(grds[1].at<uchar>(r, c));
            float mag = sqrt(gx*gx+gy*gy);
            float dir = atan2(gy, gx);

            mgs.at<float>(r, c) = mag;
            drs.at<float>(r, c) = dir;
        }
    }
}

void magnitudeMD(std::vector<cv::Mat> const& grds, cv::Mat & mgs, cv::Mat & drs)
{
    mgs = cv::Mat::zeros(grds[0].size(), CV_32F);
    drs = cv::Mat::zeros(grds[0].size(), CV_32F);
    int rows = grds[0].rows;
    int cols = grds[0].cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            float sup = float(grds[0].at<uchar>(r, c));
            float dir = 0.f;
            for (int k = 1; k < grds.size(); ++k) {
                float tmp = float(grds[k].at<uchar>(r, c));
                if (tmp > sup) {
                    sup = tmp;
                    dir = k;
                }
            }
            float val = sup;
            mgs.at<float>(r, c) = val;
            drs.at<float>(r, c) = dir*M_PI_4;
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

void hysteresis(cv::Mat const& src, cv::Mat & dst, uchar sh, uchar sb) 
{
    assert(src.type() == CV_8UC1);
    dst = cv::Mat::zeros(src.size(), src.type());
    int rows = src.rows;
    int cols = src.cols;

    // Premier parcours
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            if (src.at<uchar>(r, c) > sh) {
                dst.at<uchar>(r, c) = 255;
            }
        }
    }

    // Second parcour
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            uchar val = src.at<uchar>(r, c);
            if (val <= sh && val > sb) {
                if (checkNeighbors(dst, r, c)) {
                    dst.at<uchar>(r, c) = 255;
                }
            }
        }
    }
}

void direction(cv::Mat const& mgs, cv::Mat & dst, cv::Mat const& drs)
{
    assert(mgs.type() == CV_8UC1);
    dst = cv::Mat::zeros(mgs.size(), CV_8UC3);
    int rows = mgs.rows;
    int cols = mgs.cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            uchar mg = mgs.at<uchar>(r, c);
            float dir = drs.at<float>(r, c)*100.f;
            dst.at<cv::Vec3b>(r, c) = {mg, 0, 0};
            if (dir > 0) {
                if (dir < 80) {
                    dst.at<cv::Vec3b>(r, c) = {0, mg, 0};
                } else if (dir < 160) {
                    dst.at<cv::Vec3b>(r, c) = {0, 0, mg};
                } else {
                    dst.at<cv::Vec3b>(r, c) = {0, mg, mg};
                }
            }
        }
    }
}
