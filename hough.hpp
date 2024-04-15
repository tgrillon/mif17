#pragma once
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include <stack>

struct Line {
  cv::Point2i position_in_acc;
  float theta = 0.f;
  float rho = 0.f;
};

struct Circle {
  cv::Point2i center;
  int radius;
};

std::vector<float> thetas, rhos;

int max_rho;

bool withinMat(int x, int y, int cols, int rows) 
{
  return x >= 0 && x < cols && y >= 0 && y < rows;
}

bool within3DMat(int a, int b, int r, int aSize, int bSize, int rSize) 
{
  return a >= 0 && a < aSize && b >= 0 && b < bSize && r >= 0 && r < rSize;
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

void houghCircles(cv::Mat bin, cv::Mat &acc, uchar th) {
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

void houghCircles(cv::Mat bin, cv::Mat &acc, cv::Mat const &dirs,
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

void max3DMat(cv::Mat const& mat, double& max)
{
  assert(mat.dims == 3);
  int aSize = mat.size[1];
  int bSize = mat.size[0];
  int rSize = mat.size[2];

  max = 0;
  for (int a = 0; a < aSize; ++a) {
    for (int b = 0; b < bSize; ++b) {
      for (int r = 0; r < rSize; ++r) {
        if (mat.at<float>(b,a,r) > max) {
          max = mat.at<float>(b,a,r);
        }
      }
    }
  }
}

void colorPixel3DRegion(
  cv::Mat &bin, std::stack<cv::Point3f> &stack, int thresh, int a, int b, int r
) {
  assert(bin.dims == 3);
  int aSize = bin.size[1];
  int bSize = bin.size[0];
  int rSize = bin.size[2];
  bin.at<float>(b,a,r) = 0;

  std::vector<cv::Point3f> neighbors = {
      {a - 1, b, r}, {a + 1, b, r}, 
      {a, b - 1, r}, {a, b + 1, r},
      {a, b, r - 1}, {a, b, r + 1}
  };
  for (auto neigh : neighbors) {
    if (within3DMat(neigh.x, neigh.y, neigh.z, aSize, bSize, rSize)) {
      if (bin.at<float>(neigh.y, neigh.x, neigh.z) > thresh) {
        bin.at<float>(neigh.y, neigh.x, neigh.z) = 0;
        stack.push({neigh.x, neigh.y, neigh.z});
      }
    }
  }
}


std::vector<Circle> getCircles(
  const cv::Mat &bin, float circle_thresh, float grouping_thresh
) {
  assert(bin.dims == 3);
  int aSize = bin.size[1];
  int bSize = bin.size[0];
  int rSize = bin.size[2];

  cv::Mat tmp = bin.clone();

  std::vector<Circle> circles;

  double max;
  max3DMat(bin, max);

  for (int b = 0; b < bSize; b++) {
    for (int a = 0; a < aSize; a++) {
      for (int r = 0; r < rSize; r++) {
        if (tmp.at<float>(b,a,r) < circle_thresh*max)
          continue;
        std::stack<cv::Point3f> stack;
      
        stack.push({a, b, r});
        Circle circle;
        cv::Point3f barycenter = {0.f, 0.f, 0.f};
        int count = 0;

        while (!stack.empty()) {
          cv::Point3f p = stack.top();
          stack.pop();

          barycenter += cv::Point3f(a, b, r);

          colorPixel3DRegion(tmp, stack, grouping_thresh*max, p.x, p.y, p.z);

          ++count;
        }

        barycenter /= count;
        circle.radius = barycenter.z;
        circle.center = {barycenter.x, barycenter.y};
        circles.push_back(circle);
      }
    }
  }

  return circles;
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

void drawLines(const std::vector<Line> &lines, cv::Mat &out, int thickness) {
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

    cv::line(out, cv::Point(x1, y1), cv::Point(x2, y2), {255, 0, 0}, thickness);
  }
}

void drawCircles(std::vector<Circle> circles, cv::Mat & out, int thickness) 
{
  assert(out.type() == CV_8UC3);
  for (auto & circle : circles) {
    cv::circle(out, circle.center, circle.radius, {255, 0, 0}, thickness);
    cv::drawMarker(out, circle.center, {255, 0, 0}, 1, 10);
  }
}
