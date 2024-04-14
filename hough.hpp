#pragma once
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include <stack>

struct Line {
  cv::Point2i position_in_acc;
  float theta = 0.f;
  float rho = 0.f;
};

std::vector<float> thetas, rhos;

int max_rho;

bool withinMat(int x, int y, int cols, int rows) {
  return x >= 0 && x < cols && y >= 0 && y < rows;
}

void houghLines(cv::Mat bin, cv::Mat &acc, uchar thresh = 170) {
  int max_theta = 180;
  max_rho = std::ceil(sqrt(bin.cols * bin.cols + bin.rows * bin.rows));

  acc = cv::Mat::zeros(max_theta, 2 * max_rho, CV_32F);

  for (int y = 0; y < bin.rows; y++) {
    for (int x = 0; x < bin.cols; x++) {
      if (bin.at<uchar>(y, x) < thresh)
        continue;

      for (int t = 0; t < max_theta; ++t) {
        float theta = radians(t);
        int rho = int(x * cos(theta) + y * sin(theta));

        int r = rho + max_rho;

        // range of rho mapped from -max_rho : max_rho to 0 : 2max_rho
        acc.at<float>(t, r) += 1.;
      }
    }
  }
}
void houghLines(cv::Mat bin, cv::Mat &acc, cv::Mat &dirs, uchar thresh = 170) {
  int max_theta = 180;
  max_rho = std::ceil(sqrt(bin.cols * bin.cols + bin.rows * bin.rows));

  acc = cv::Mat::zeros(max_theta + 1, 2 * max_rho, CV_32F);

  for (int y = 0; y < bin.rows; y++) {
    for (int x = 0; x < bin.cols; x++) {
      if (bin.at<uchar>(y, x) < thresh)
        continue;

      float theta = dirs.at<float>(y, x);

      if (theta < 0)
        theta = radians(180) + theta;
      else if (theta > radians(180))
        theta = theta - radians(180);

      int rho = int(x * cos(theta) + y * sin(theta));

      int r = rho + max_rho;
      // range of rho mapped from -max_rho : max_rho to 0 : 2max_rho
      acc.at<float>(degrees(theta), r) += 1.;
    }
  }
}

void houghCircle(cv::Mat bin, cv::Mat &acc, uchar th) {
  // a, b, r

  int max_r = std::min(bin.cols, bin.rows);
  int max_a = bin.cols;
  int max_b = bin.rows;

  int sizes[]{max_a, max_b, max_r};

  acc = cv::Mat::zeros(3, sizes, CV_32F);

  for (int y = 0; y < bin.rows; y++) {
    for (int x = 0; x < bin.cols; x++) {
      if (bin.at<uchar>(y, x) < th)
        continue;

      for (int a = 0; a < max_a; a++) {
        for (int b = 0; b < max_b; b++) {
          float da = a - x;
          float db = b - y;
          // Calculer directement r
          float r = sqrt(da * da + db * db);
          acc.at<float>(a, b, r) += 1;
        }
      }
    }
  }
}
void incLineDir(cv::Mat &acc, float theta, int x, int y, int max_a, int max_b,
                int dir = 1) {
  int r = 1, a, b;
  do {
    a = x + dir * r * cos(theta);
    b = y + dir * r * sin(theta);

    if (withinMat(a, b, max_a, max_b)) {
      acc.at<float>(b, a, r) += 1;
    } else
      break;
    ++r;
  } while (true);
}

void houghCircleDirection(cv::Mat bin, cv::Mat &acc, cv::Mat const &dirs,
                          uchar th) {

  int max_r = sqrt(bin.rows * bin.rows + bin.cols * bin.cols);
  int max_a = bin.cols;
  int max_b = bin.rows;

  int sizes[]{max_b, max_a, max_r};

  acc = cv::Mat::zeros(3, sizes, CV_32F);

  for (int y = 0; y < bin.rows; y++) {
    for (int x = 0; x < bin.cols; x++) {
      if (bin.at<uchar>(y, x) < th)
        continue;

      float theta = dirs.at<float>(y, x);

      incLineDir(acc, theta, x, y, max_a, max_b);
      incLineDir(acc, theta, x, y, max_a, max_b, -1);
    }
  }
}

void colorPixelRegion(cv::Mat &bin, std::stack<cv::Point> &stack, int thresh,
                      unsigned int x, unsigned int y) {
  unsigned int rows = bin.rows;
  unsigned int cols = bin.cols;
  bin.at<float>(y, x) = 0;

  std::vector<cv::Point> neighbors = {
      {x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};
  for (auto neigh : neighbors) {
    if (withinMat(neigh.x, neigh.y, cols, rows)) {
      if (bin.at<float>(neigh.y, neigh.x) > thresh) {
        bin.at<float>(neigh.y, neigh.x) = 0;
        stack.push({neigh.x, neigh.y});
      }
    }
  }
}

std::vector<Line> getLines(const cv::Mat &bin, float th1 = 0.4f,
                           float th2 = 0.05f) {
  cv::Mat tmp = bin.clone();

  std::vector<Line> lines;

  double max;
  cv::Point empty;
  cv::minMaxLoc(bin, nullptr, &max);

  for (int y = 0; y < tmp.rows; y++) {
    for (int x = 0; x < tmp.cols; x++) {
      if (tmp.at<float>(y, x) < th1 * max)
        continue;
      std::stack<cv::Point> stack;
      stack.push({x, y});
      Line line;
      cv::Point2f barycenter = {0.f, 0.f};
      int count = 0;

      while (!stack.empty()) {
        cv::Point p = stack.top();
        stack.pop();

        barycenter += cv::Point2f(x, y);

        colorPixelRegion(tmp, stack, th2 * max, p.x, p.y);

        ++count;
      }

      barycenter /= count;
      line.theta = radians(barycenter.y);
      line.rho = barycenter.x - max_rho;
      line.position_in_acc = {barycenter.x, barycenter.y};
      lines.push_back(line);
    }
  }

  return lines;
}

// std::vector<Line> accLocalExtremum(cv::Mat & acc)
// {
//   std::vector<Line> extrema;
//   bool search = false;
//   for (int r = 0; r < acc.rows; ++r) {
//     for (int c = 0; c < acc.cols; ++c) {
//       float val = acc.at<float>(r, c);
//       if (val > 0) {
//         if (!search) {
//           search = true;
//           extrema.push_back({r, c});
//         } else {
//           Line line = extrema[extrema.size()-1];
//           if (val > acc.at<float>(line.theta, line.rho)) {
//             extrema[extrema.size()-1] = {r, c};
//           }
//         }
//       } else search = false;
//     }
//   }

//   return extrema;
// }

void intersectImg(cv::Mat const &bin, cv::Mat const &lns, cv::Mat &dst) {
  assert(bin.type() == lns.type());
  dst = lns.clone();
  int rows = bin.rows;
  int cols = bin.cols;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (dst.at<uchar>(r, c) != 255)
        continue;
      dst.at<uchar>(r, c) = (bin.at<uchar>(r, c) == 255) ? 255 : 0;
    }
  }
}

void drawLocalExtrema(const std::vector<Line> &lines, cv::Mat &out) {
  for (auto &line : lines) {
    cv::drawMarker(out, {line.position_in_acc.x, line.position_in_acc.y},
                   {255, 0, 0}, 1, 10);
  }
}

void drawLines(const std::vector<Line> &lines, cv::Mat &hough_lines) {
  for (auto &line : lines) {
    float theta = line.theta;
    float rho = line.rho;

    float a = cos(theta);
    float b = sin(theta);

    int x0 = rho * a;
    int y0 = rho * b;

    int x1 = int(x0 + 1000 * (-b));
    int y1 = int(y0 + 1000 * (a));
    int x2 = int(x0 - 1000 * (-b));
    int y2 = int(y0 - 1000 * (a));

    cv::line(hough_lines, cv::Point(x1, y1), cv::Point(x2, y2), 255, 1);
  }
}

void drawCircles(cv::Mat const &src, cv::Mat &dst, cv::Mat const &acc,
                 uchar th) {
  dst = cv::Mat::zeros(src.size(), CV_8UC3);
  for (int b = 0; b < acc.size[0]; ++b) {
    for (int a = 0; a < acc.size[1]; ++a) {
      for (int r = 0; r < acc.size[2]; ++r) {
        if (cv::saturate_cast<uchar>(acc.at<float>(b, a, r)) < th)
          continue;
        cv::circle(dst, cv::Point(a, b), r, {255, 0, 0}, 3);
        cv::drawMarker(dst, cv::Point(a, b), {255, 0, 0}, 1, 10);
      }
    }
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

//     } while (p.x < intersect.cols && p.x >= 0 && p.y < intersect.rows && p.y
//     >= 0);

//     for (auto s : segments){
//         cv::line(hough_lines, s.a, s.b, cv::Scalar(255, 0, 0), 1);
//     }
