#pragma once
#include"three_to_fourth.hpp"



//9选3
vector<Point2f> nin_to_3(vector<Triangle> a, Mat cap);

//三角形传参数函数
vector<Triangle> tri_init();

//可视化画三角形
void drawmap(vector<Triangle>& triangles, Mat img);
//判断是否为钝角三角形（极端情况110度）
bool is_obtuse_triangle(Triangle a);

//模块总函数
vector<Point2f> quadraDominance(vector<Triangle>& triangles, cv::Mat &img);