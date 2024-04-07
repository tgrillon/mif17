#include "opencv2/imgcodecs.hpp"
#include "ui.hpp"
#include <cstdio>
#include <memory>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {
  const char *filepath =
      (argc > 1) ? argv[1] : "../../ressources/Droites_simples.png";

  cv::Mat img;
  img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);

  if (!img.data) {
    printf("No image data \n");
    return -1;
  }

  std::unique_ptr<Viewer> viewer(new DemoHoughLinesGrad(img));
  viewer->show();
  return 0;
}
