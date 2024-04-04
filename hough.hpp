#pragma once
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include <stack>

const float step = 1;

struct Region {
    cv::Point2f center={0.f, 0.f};
    int count=0;
};

void houghLines(cv::Mat bin, cv::Mat & intersections, uchar thresh = 170) {
    int max_theta = 360 / step; 
    int max_rho = std::ceil(sqrt(bin.cols*bin.cols + bin.rows*bin.rows));

    intersections = cv::Mat::zeros(max_theta, max_rho, CV_32F);
    
    for (int y = 0; y < bin.rows; y++){
        for(int x = 0; x < bin.cols; x++){
            if (bin.at<uchar>(y,x) < thresh) continue;

            float grad = ((float)(bin.at<uchar>(y,x) - thresh)) / ( 255 - thresh ) ;

            for (int t = 0; t < max_theta; t++){
                float theta = radians(t * step);
                int rho = int(x*cos(theta) + y*sin(theta));
                    
                intersections.at<float>(t, rho) += 1;
            }
        }
    }
}


void houghCircle(cv::Mat bin, cv::Mat & intersections) {
    // a, b, r

    int max_r = std::min(bin.cols, bin.rows);
    int max_a = bin.cols;
    int max_b = bin.rows;
    uchar th = 170;
    
    int sizes[] {max_a, max_b, max_r};
    
    intersections = cv::Mat::zeros(3, sizes, CV_8UC1);
    
    for (int y = 0; y < bin.rows; y++){
        for(int x = 0; x < bin.cols; x++){
            if (bin.at<uchar>(y,x) < th) continue;

            float grad = ((float)(bin.at<uchar>(y,x) - th)) / ( 255 - th ) ;

            for (int a = 0; a < max_a; a++) {
                for (int b = 0; b < max_b; b++) {
                    float da = a - x;
                    float db = b - y;
                    // Calculer directement r
                    float r = sqrt(da * da + db * db);
                    intersections.at<float>(a, b, r)++;
                }
            }
        }
    }
}


void color_pixel_region(cv::Mat & bin, cv::Mat& result, std::stack<cv::Point>& stack, int thresh, unsigned int x, unsigned int y)
{
    unsigned int rows = bin.rows;
    unsigned int cols = bin.cols;
    bin.at<float>(y, x) = 0;
    // result.at<uchar>(y, x) = 255;

    std::vector<cv::Point> neighbors = {{x-1, y}, {x+1, y}, {x, y-1}, {x, y+1}};
    for (auto neigh : neighbors) {
        if (neigh.x >= 0 && neigh.x < cols && neigh.y >= 0 && neigh.y < rows) {
            if (bin.at<float>(neigh.y, neigh.x) > thresh) {
                bin.at<float>(neigh.y, neigh.x) = 0;
                stack.push({neigh.x, neigh.y});
            }
        }
    }
}

std::vector<Region> get_regions(const cv::Mat& bin){
    cv::Mat img = bin.clone();
    cv::Mat result = cv::Mat::zeros(bin.rows, bin.cols, bin.type());

    std::vector<Region> regions;
    // cv::Mat pixel2Region(bin.rows, bin.cols, CV_8UC1);

    double max, min;
    cv::Point empty;
    cv::minMaxLoc(bin, &min, &max, &empty, &empty );

    const float thresh = 0.4*max;
    const float thresh2 = 0.05*max;

    for (int y = 0; y < img.rows; y++){
        for (int x = 0; x < img.cols; x++){
            
            if (img.at<float>(y,x) > thresh ) {
                std::stack<cv::Point> stack;
                stack.push({x, y});
                Region region;

                while (!stack.empty()) { 
                    cv::Point p = stack.top();
                    stack.pop();

                    region.center += cv::Point2f(x, y);
                    region.count++;

                    color_pixel_region(img, result, stack, thresh2, x, y);
                }
                
                region.center /= region.count;
                regions.push_back(region);
            }
        }
    }

    return regions;
}

void intersect_img(cv::Mat const& bin, cv::Mat const& lns, cv::Mat & dst) 
{
    assert(bin.type() == lns.type());
    dst = lns.clone();
    int rows = bin.rows;
    int cols = bin.cols;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (dst.at<uchar>(r, c) != 255) continue; 
            dst.at<uchar>(r, c) = (bin.at<uchar>(r, c) == 255) ? 255 : 0; 
        }
    }
}

void draw_local_maximums(const std::vector<Region>& regions, cv::Mat& out) {
    for (auto & region : regions) {
        cv::drawMarker(out, region.center, {255,0,0}, 1, 10);
    }
}

void draw_lines(const std::vector<Region>& regions, cv::Mat& hough_lines) {
    for (auto & region : regions) {
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

            
        cv::line(hough_lines, cv::Point(x1, y1), cv::Point(x2, y2), 255, 1);
    }
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
