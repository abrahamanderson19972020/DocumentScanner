#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

Mat imgOriginal, imgGray, imgCanny, imgBlur, imgThreshold, imgDilation, imgErode, imgWarp, imgCrop;
std::vector<Point> initialPoints, docPoints;

float w = 819, h = 614;

Mat PreProcessing(Mat img)
{
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
    Canny(imgBlur, imgCanny, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDilation, kernel);
    return imgDilation;
}

std::vector<Point> GetContours(Mat img) {
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<Point>> conPoly(contours.size());
    std::vector<Rect> boundRect(contours.size());
    std::vector<Point> biggest;
    int maxArea = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        int area = contourArea(contours[i]);
        if (area > 1000)
        {
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
            if (area > maxArea && conPoly[i].size() == 4)
            {
                drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 5);
                maxArea = area;
                biggest = { conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3] };
            }
        }
    }
    return biggest;
}

void DrawPoints(std::vector<Point> points, Scalar color)
{
    for (int i = 0; i < points.size(); i++)
    {
        circle(imgOriginal, points[i], 10, color, FILLED);
        putText(imgOriginal, std::to_string(i), points[i], FONT_HERSHEY_PLAIN, 4, color, 4);
    }
}

std::vector<Point> Reorder(std::vector<Point> points) {
    std::vector<Point> newPoints;
    std::vector<int>sumPoints, subPoints;
    for (int i = 0; i < 4; i++)
    {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }  
    newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // Index 0
    newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //
    newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // Index 4
    return newPoints;
}

Mat GetWarp(Mat img, std::vector<Point> points, float w, float h)
{
    Point2f src[4] = { points[0], points[1], points[2], points[3]};
    Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };
    Mat matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, Point(w, h));
    return imgWarp;
}

int main() {
    std::string path = "Resources/testdocument.jpg";
    imgOriginal = imread(path);
    if (imgOriginal.empty()) {
        std::cerr << "Could not read the image: " << path << std::endl;
        return 1;
    }
    resize(imgOriginal, imgOriginal, Size(), 0.2, 0.2);
    std::cout << "Width: " << imgOriginal.cols << " Height: " << imgOriginal.rows<<std::endl;

    // Preprocessing
    imgThreshold = PreProcessing(imgOriginal);
    // Get Contours
    initialPoints = GetContours(imgThreshold);
    //DrawPoints(initialPoints, Scalar(0, 0, 255));
    docPoints = Reorder(initialPoints);
    //DrawPoints(docPoints, Scalar(0, 255, 0));

    // Warping Original Image: Make it Straight:

    imgWarp = GetWarp(imgOriginal, docPoints, w, h);

    //Crop Image
    Rect roi(5, 5, w - (5), h - ( 5));
    imgCrop = imgWarp(roi);
    imshow("Original Image", imgOriginal);
    imshow("Preprocessed Image", imgThreshold);
    imshow("Image Warp", imgWarp);
    imshow("Image Crop", imgCrop);
    waitKey(0);
    return 0;
}
