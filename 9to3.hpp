#pragma once
#include"three_to_fourth.hpp"



//9ѡ3
vector<Point2f> nin_to_3(vector<Triangle> a, Mat cap);

//�����δ���������
vector<Triangle> tri_init();

//���ӻ���������
void drawmap(vector<Triangle>& triangles, Mat img);
//�ж��Ƿ�Ϊ�۽������Σ��������110�ȣ�
bool is_obtuse_triangle(Triangle a);

//ģ���ܺ���
vector<Point2f> quadraDominance(vector<Triangle>& triangles, cv::Mat &img);