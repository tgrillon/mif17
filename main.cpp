#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <stack>

#include "hough.hpp"
#include "ui.hpp"
#include "kernel.hpp"
#include "gradient.hpp"

#include "applications.hpp"

int main(int argc, char** argv)
{
    const char* filepath = (argc > 1) ? argv[1] : "../ressources/Droites_simples.png";

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
    grds = compute_gradients(img, h, dim);



    cv::Mat mgs, drs, fnl;
//    for (auto & grd : grds) {
//        threshold(grd, grd, 5);
//    }

    magnitude(grds, mgs, drs);

    
    calcHist(&mgs, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);
    // gnuPlot(hist, "Histogramme magnitude", histSize);

    /*********Seuillage Unique (par Histogramme)************/

    // float pct = (argc > 2) ? float(std::atoi(argv[2]))/100 : 0.1;
    // float sum = 0; 
    // float imgSize = float(img.rows * img.cols);
    // int gs = 0;
    // while (sum/imgSize < pct && gs < 256) {
    //     sum += hist.at<float>(gs,0);
    //     ++gs;
    // }

    // uchar gs_8uc = cv::saturate_cast<uchar>(gs);
    // thresholding(mgs, mgs, gs_8uc);

    /*******************************************************/
    cv::Mat tmp;
    hysteresis(mgs, fnl, 24, 4);
    // direction(tmp, fnl, drs);

    // cv::imshow("Magnitude", mgs);
    // cv::waitKey();
    cv::Mat intersect;
    houghLines(fnl, intersect);

    int img_width = fnl.cols;
    int img_height = fnl.rows;
    
    print_mat<float>(intersect);

    cv::Mat hough_lines = cv::Mat::zeros(fnl.rows, fnl.cols, CV_8UC1);

    double max, min;
    cv::Point empty;
    cv::minMaxLoc(intersect, &min, &max, &empty, &empty );
    
    // cv::Mat graph = cv::Mat::zeros(intersect.rows, intersect.cols, CV_8UC1);
    // for (int y = 0; y < graph.cols; y++){
    //     for (int x = 0; x < graph.rows; x++) {
    //         graph.at<uchar>(y,x) = (intersect.at<unsigned int>(y,x) / max) * 255;
    //     }
    // }

    // cv::imshow("test", graph);

    struct Axis {
        Axis(float x1, float x2, float y1, float y2)
        : x1(x1), x2(x2), y1(y1), y2(y2) {}

        float x1, x2, y1, y2;
    };
    std::vector<Axis> droites;

    cv::Mat dest;
    
    auto regions = get_regions(intersect);
    // Afficher les régions 
    cv::Mat regimg = cv::Mat::zeros(intersect.rows, intersect.cols, CV_8UC3);

    for (auto & region : regions) {
        cv::drawMarker(regimg, region.center, {255,0,0}, 1, 10);
        int x = region.center.x;
        int y = region.center.y;

        float rho = x;
        float theta = radians(y * step);

        float a = cos(theta);
        float b = sin(theta);
            
        int x0 = a*rho;
        int y0 = b*rho;
        int x1 = int(x0 + 1000*(-b));
        int y1 = int(y0 + 1000*(a));
        int x2 = int(x0 - 1000*(-b));
        int y2 = int(y0 - 1000*(a));

            
        droites.push_back(Axis(x1,x2,y1,y2));
        cv::line(hough_lines, cv::Point(x1, y1), cv::Point(x2, y2), 255, 1);
    }

    intersect_img(fnl, hough_lines, dest);

    // }

    // for (int k = 0; k < dim; ++k) {
    //     std::string title = std::to_string(k) + " gradient";
    //     cv::imshow(title, grds[k]);
    // }

    cv::Mat showInter;
    intersect.convertTo(showInter, CV_8UC1, 255/max);
    cv::imshow("intersect", showInter);
    cv::imshow("regimg", regimg);
    cv::imshow("img", img);
    cv::imshow("Hough lines", hough_lines);
    cv::imshow("ezaczevr", dest);
    cv::waitKey(0);
    return 0;
}

/*
* 
    struct Segment {
        Segment(cv::Point a, cv::Point b)
        : a(a), b(b) {}
        cv::Point a, b;
    };
    std::vector<Segment> segments;


*
**/
// Iter sur les segments
    // for (auto d : droites) {
    //     const float step = 0.5f;
    //     const int error = 3;

    //     float t = 0;
    //     float count = 0;
    //     float err_count = error;

    //     float dx = d.x2 - d.x1;
    //     float dy = d.y2 - d.y1;
        
    //     cv::Point2f o = 
    //         {std::clamp(d.x1, 0.f, (float)img_width), 
    //         std::clamp(d.y1, 0.f, (float)img_height)};
    //     cv::Vec2f tmp = cv::Vec2f(
    //         std::clamp(dx, 0.f, (float)img_width), 
    //         std::clamp(dy, 0.f, (float)img_height)
    //         );
    //     cv::Point2f dir = cv::normalize(tmp) * step;
    //     cv::Point2f p = o, segment_start; 

    //     do {
    //         p += dir; 

    //         bool is_edge = mgs.at<uchar>(p.y, p.x) > 100 ? true : false;

    //         if (!is_edge){
    //             if (err_count == error - 1){
    //                 segments.push_back(Segment(segment_start, p));
    //             }

    //             err_count++;
    //             continue;
    //         }

    //         if (err_count >= error) {
    //             segment_start = p;
    //         }
            
    //         err_count = 0;

    //     } while (p.x < intersect.cols && p.x >= 0 && p.y < intersect.rows && p.y >= 0);

    //     for (auto s : segments){
    //         cv::line(hough_lines, s.a, s.b, cv::Scalar(255, 0, 0), 1);
    //     }