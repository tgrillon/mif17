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

template <typename T>
void print_mat(cv::Mat const& mat)
{
    for (int r = 0; r < mat.rows; ++r) {
        for (int c = 0; c < mat.cols; ++c) {
            std::cout << (float)mat.at<T>(r, c) << " ";
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
    mgs = cv::Mat::zeros(grds[0].size(), CV_8UC1);
    drs = cv::Mat::zeros(grds[0].size(), CV_8UC1);
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

float radians(float a) { return M_PI/180 * a; }

const float step = 1; 

void houghLines(cv::Mat bin, cv::Mat & intersections) {
    uchar th = 170;

    int max_theta = 360 / step; 
    int max_rho = std::ceil(sqrt(bin.cols*bin.cols + bin.rows*bin.rows));

    intersections = cv::Mat::zeros(max_theta, max_rho, CV_32F);
    
    for (int y = 0; y < bin.rows; y++){
        for(int x = 0; x < bin.cols; x++){
            if (bin.at<uchar>(y,x) < th) continue;

            float grad = ((float)(bin.at<uchar>(y,x) - th)) / ( 255 - th ) ;

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


    for (int y = 0; y < intersect.rows; y++){
        for (int x = 0; x < intersect.cols; x++){
            if (intersect.at<float>(y,x) < 0.3 * max) continue;

            float rho = x;
            float theta = radians(y * step);

            float a = cos(theta);
            float b = sin(theta);

            // t*dir 

            // xa + yb = p
            // y = (p - xa)/b

            // float x1 = rho/a;
            // float y1 = rho/b;
            // float x2 = (rho - b*img_height)/a;
            // float y2 = (rho - a*img_width)/b;

            float x1 = 0;
            
            float y1 = rho/b;
            float x2 = img_width;
            float y2 = (rho - img_height*a)/b;

            droites.push_back(Axis(x1,x2,y1,y2));
            cv::line(hough_lines, cv::Point(x1, y1), cv::Point(x2, y2), 255, 1);
        }
    }

    struct Segment {
        Segment(cv::Point a, cv::Point b)
        : a(a), b(b) {}
        cv::Point a, b;
    };

    std::vector<Segment> segments;

    cv::Mat dest;

    intersect_img(fnl, hough_lines, dest);

    // }

    // for (int k = 0; k < dim; ++k) {
    //     std::string title = std::to_string(k) + " gradient";
    //     cv::imshow(title, grds[k]);
    // }

    cv::imshow("img", img);
    cv::imshow("Hough lines", hough_lines);
    cv::imshow("ezaczevr", dest);
    cv::waitKey(0);
    return 0;
}




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