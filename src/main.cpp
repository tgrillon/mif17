#include "opencv2/imgcodecs.hpp"
#include "ui.hpp"
#include <cstdio>
#include <memory>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {
  Viewer* viewer;

  // if (argc == 1) {
  //   std::cout << "Usage : " << argv[0] << " [lines|circles] <filepath>";
  //   return 0; 
  // }

  const char *filepath =
      (argc > 2) ? argv[2] : "../ressources/droites_simples.png";

  cv::Mat img;
  img = cv::imread(filepath);

  if (!img.data) {
    printf("No image data \n");
    return -1;
  } 

      // viewer = new DemoHoughLinesGrad(flt);

  if (argc > 1) {
    std::string mode = argv[1];

    if (mode.compare("lines") == 0)
      viewer = new DemoHoughLinesGrad(img);
    else if (mode.compare("circles") == 0)
      viewer = new DemoHoughCirclesGrad(img);
    else{
      std::cerr << "Invalid argument for viewer mode";
      return -1;
    }
  }

  
  
  // cv::medianBlur(img, flt, md); 

  viewer->show();

  // std::unique_ptr<Viewer> viewer2(new DemoHoughCirclesGrad(img));
  // viewer2->show();
  return 0;
}
