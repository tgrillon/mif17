#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>


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
        rt.at<float>(0, 0) = rt.at<float>(0, 1);
        rt.at<float>(0, 1) = rt.at<float>(0, 2);

        rt.at<float>(0, 2) = rt.at<float>(1, 2);
        rt.at<float>(1, 2) = rt.at<float>(2, 2);
        rt.at<float>(2, 2) = rt.at<float>(2, 1);

        rt.at<float>(2, 1) = rt.at<float>(2, 0);
        rt.at<float>(2, 0) = rt.at<float>(1, 0);
        rt.at<float>(1, 0) = bff;

        return rt;
    }
}

void gnuPlot(const cv::Mat& hist, const std::string& fileName, const int histSize) {
    std::ofstream dataFile("../ressources/" + fileName + ".txt");
    for (int i = 0; i < histSize; i++) {
        dataFile << i << " " << hist.at<float>(i) << std::endl;
    }
    dataFile.close();

    FILE* gnuplotPipe = popen("gnuplot -persistent", "w");
    if (gnuplotPipe) {
        fprintf(gnuplotPipe, "set title 'Histogramme de %s'\n", fileName.c_str());
        fprintf(gnuplotPipe, "plot '../ressources/%s.txt' with boxes\n", fileName.c_str());
        fflush(gnuplotPipe);
        getchar(); 
        pclose(gnuplotPipe);
    } else {
        std::cerr << "Erreur lors de l'ouverture de Gnuplot." << std::endl;
    }
}

cv::Mat calcHistCumul(const cv::Mat& src, int histSize) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    dst.at<float>(0) = src.at<float>(0);

    for (int i=1; i<histSize; ++i) {
        dst.at<float>(i) = dst.at<float>(i-1) + src.at<float>(i); 
    }
    return dst;
}

cv::Mat etirement(const cv::Mat& image, int Nmin, int Nmax) 
{
    int height = image.rows;
    int width = image.cols;
    cv::Mat image2(height, width, CV_8UC1);
    for (int r=0; r<height; ++r) {
        for (int c=0; c<width; ++c) {
            image2.at<uchar>(r, c) = cv::saturate_cast<uchar>(255 * ((image.at<uchar>(r, c) - Nmin) / static_cast<double>(Nmax - Nmin)));
        }
    }

    return image2;
}

cv::Mat egalisation(const cv::Mat & inputImage, const cv::Mat& inputHist, int histSize) {
    cv::Mat histoCumul = calcHistCumul(inputHist, histSize);

    histoCumul /= inputImage.total();

    cv::Mat outputImage = inputImage.clone();
   
    for (int i = 0; i < outputImage.rows; ++i) {
        for (int j = 0; j < outputImage.cols; ++j) {
            outputImage.at<uchar>(i, j) = cv::saturate_cast<uchar>(255 * histoCumul.at<float>(inputImage.at<uchar>(i, j)));
        }
    }
    return outputImage;
}

float convolution(const cv::Mat& img, const cv::Mat& h, int x, int y)
{
    float sum = 0.0;
    for (int u = -1; u <= 1; ++u) {
        for (int v = -1; v <= 1; ++v) {
            sum += h.at<float>(1 + u, 1 + v) * img.at<uchar>(y + u, x + v);
        }
    }
    return sum;
}

void print_mat(cv::Mat const& mat)
{
    for (int r = 0; r < mat.rows; ++r) {
        for (int c = 0; c < mat.cols; ++c) {
            std::cout << float(mat.at<uchar>(r, c)) << " ";
        }
        std::cout << std::endl;
    }
}

enum Dimension {
    TWO_DIM=2,
    MULTI_DIM=4
};

std::vector<cv::Mat> compute_gradients(const cv::Mat& src, const cv::Mat& h, Dimension dim=MULTI_DIM)
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

void magnitude(std::vector<cv::Mat> const& grds, cv::Mat & mgs, cv::Mat & drs)
{
    mgs = cv::Mat::zeros(grds[0].size(), grds[0].type());
    drs = cv::Mat::zeros(grds[0].size(), grds[0].type());
    int rows = grds[0].rows;
    int cols = grds[0].cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            uchar sup = grds[0].at<uchar>(r, c);
            int dir = 0;
            for (int k = 1; k < grds.size(); ++k) {
                uchar tmp = grds[k].at<uchar>(r, c);
                if (tmp > sup) {
                    sup = tmp;
                    dir = k;
                }
            }
            uchar val = sup;
            mgs.at<uchar>(r, c) = val;
            drs.at<uchar>(r, c) = dir;
        }
    }
}

void filter(const cv::Mat& src, cv::Mat& dst, const cv::Mat& h)
{
    assert(h.rows == 3 && h.cols == 3);
    int height = src.rows;
    int width = src.cols;
    dst = cv::Mat::zeros(src.size(), src.type());

    for (int r = 1; r < height - 1; ++r) {
        for (int c = 1; c < width - 1; ++c) {
            dst.at<uchar>(r, c) = cv::saturate_cast<uchar>(convolution(src, h, c, r));
        }
    }
}

void direction(cv::Mat const& src, cv::Mat & dst, cv::Mat const& drs)
{
    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    int rows = src.rows;
    int cols = src.cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            uchar val = src.at<uchar>(r, c);
            uchar dir = drs.at<uchar>(r, c);
            dst.at<cv::Vec3b>(r, c) = {val, 0, 0};
            if (dir == 1) dst.at<cv::Vec3b>(r, c) = {0, val, 0};
            else if (dir == 2) dst.at<cv::Vec3b>(r, c) = {0, 0, val};
            else if (dir == 3) dst.at<cv::Vec3b>(r, c) = {val, 0, val};
        }
    }
}


void thresholding(cv::Mat const& src, cv::Mat & dst, uchar ths)
{
    assert(src.type() == CV_8UC1);
    dst = src.clone();
    int rows = src.rows;
    int cols = src.cols;
    for (int r = 1; r < rows-1; ++r) {
        for (int c = 1; c < cols-1; ++c) {
            uchar val = src.at<uchar>(r, c);
            // if (val < ths) val = (val - ths*.5 < 0) ? 0 : val - ths*.5;
            // else val = (val + ths*.5 > 255) ? 255 : val + ths*.5;
            if (val < ths) val = 0;
            else val = 255;
            dst.at<uchar>(r, c) = val;
        }
    }
}

bool check_neighbors(cv::Mat const& img, unsigned int r, unsigned int c) 
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
                if (check_neighbors(dst, r, c)) {
                    dst.at<uchar>(r, c) = 255;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    const char* filepath = (argc > 1) ? argv[1] : "../ressources/lena.png";

    cv::Mat img, blur;
    img = cv::imread(filepath, 0);

    if (!img.data) {
        printf("No image data \n");
        return -1;
    }

    int histSize = 256; // de 0 à 255
    float range[] = { 0, 255 };
    const float* histRange[] = { range };

    int histWidth = 512;
    int histHeight = 400;

    cv::Mat hist;
    // calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);
    // gnuPlot(hist, "Histogramme", histSize);

    // img = etirement(img, 25, 235);
    // img = egalisation(img, hist, histSize);

    int i = 11;
    bilateralFilter(img, blur, i, i*2, i/2);

    cv::Mat h(3, 3, CV_32F, const_cast<float*>(kernel::kirsch));

    Dimension dim = MULTI_DIM;

    std::vector<cv::Mat> grds;
    grds = compute_gradients(blur, h, dim);

    cv::Mat mgs, drs, fnl;
//    for (auto & grd : grds) {
//        threshold(grd, grd, 5);
//    }

    magnitude(grds, mgs, drs);

    calcHist(&mgs, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);
    // gnuPlot(hist, "Histogramme magnitude", histSize);

    /*********Seuillage Unique (par Histogramme)************/

    float pct = (argc > 2) ? float(std::atoi(argv[2]))/100 : 0.1;
    float sum = 0; 
    float imgSize = float(img.rows * img.cols);
    int gs = 0;
    while (sum/imgSize < pct && gs < 256) {
        sum += hist.at<float>(gs,0);
        ++gs;
    }

    uchar gs_8uc = cv::saturate_cast<uchar>(gs);
    thresholding(mgs, mgs, gs_8uc);

    /*******************************************************/
    cv::Mat tmp;
    hysteresis(mgs, tmp, 24, 4);
    direction(tmp, fnl, drs);

    cv::imshow("Magnitude", fnl);
    for (int k = 0; k < dim; ++k) {
        std::string title = std::to_string(k) + " gradient";
        cv::imshow(title, grds[k]);
    }
    cv::waitKey(0);
    return 0;
}