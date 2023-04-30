#pragma once
#include<iostream>
#include<opencv.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <climits>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include<fstream>
#include <numeric>
#include"9to3.hpp"
using namespace std;
using namespace cv;

#define DEBUG12                         //������
#define DEBUG23                      //��б����
#define DEBUG36                       //��������
#define COLOR 0//0��ɫ��1��ɫ,2��ɫ
#define ISMAP1  //1��ͼƬ��2����Ƶ




double transf(double a);

//double�������ֵ
const double INF = numeric_limits<double>::infinity();

Mat precessing(Mat image);

void find_min_diff_indices(double arr[], int n, int& ind1, int& ind2, int& ind3);

void find_min_diff_angles(double arr[], int n, int& ind1, int& ind2, int& ind3);

double findMinangles(vector<double>& angles);

vector<Point2f> findApexs(vector<RotatedRect> minRect, vector<vector<Point>> contours, vector<Triangle> triangles, vector<int> index);

vector<Triangle> detectTriangles(const vector<vector<Point>>& contours, Mat& cap);

int find_four_apex(vector<vector<Point>> contours, vector<Triangle> triangle, vector<int> findMinindex, Mat cap);

vector<int> findMinareas(vector<double> areas);

vector<int> handleLight(vector<vector<Point>> contours, vector<RotatedRect> minRect, vector<Triangle> triangle, Mat cap);

vector<Point2f> handleMat(Mat src, Mat image);

void rotationMatrixToEulerAngles(Mat& R, double& roll, double& pitch, double& yaw);

void solveXYZ(std::vector<cv::Point2f> vertices, cv::Mat image);


void all(Mat image);
