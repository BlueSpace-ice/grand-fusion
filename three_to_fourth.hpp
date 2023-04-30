#pragma once
#include<opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

// �����νṹ�壬��������ֱ�ߵ���ǣ������������
struct Triangle {
	double angle1 = 0;
	double angle2 = 0;
	double angle3 = 0;
	double edge_len1 = 0;
	double edge_len2 = 0;
	double edge_len3 = 0;
	Point2f pt1;
	Point2f pt2;
	Point2f pt3;
	vector<Point2f> triangle_points;
	vector<pair<Point2f, Point2f>> lines;

	// ��������������ĺ���
	double getArea() {
		// ��triangle_points�е�������������������
		double a = edge_len1;
		double b = edge_len2;
		double c = edge_len3;
		double s = (a + b + c) / 2;
		double area = sqrt(s * (s - a) * (s - b) * (s - c));
		return area;
	}
	void init() {
		if (triangle_points.size() != 3)
		{
			cout << "init error" << endl;
			return;
		}
		// �������㰴˳��ֱ�����Ϊpt1, pt2, pt3
		pt1 = triangle_points[0];
		pt2 = triangle_points[1];
		pt3 = triangle_points[2];

		// ���������ߵĳ���
		edge_len1 = sqrt(pow(pt2.x - pt3.x, 2) + pow(pt2.y - pt3.y, 2));
		edge_len2 = sqrt(pow(pt1.x - pt3.x, 2) + pow(pt1.y - pt3.y, 2));
		edge_len3 = sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));

		// ���������Ƕȵ�ֵ
		angle1 = acos((edge_len2 * edge_len2 + edge_len3 * edge_len3 - edge_len1 * edge_len1) / (2 * edge_len2 * edge_len3)) * 180 / CV_PI;
		angle2 = acos((edge_len1 * edge_len1 + edge_len3 * edge_len3 - edge_len2 * edge_len2) / (2 * edge_len1 * edge_len3)) * 180 / CV_PI;
		angle3 = acos((edge_len1 * edge_len1 + edge_len2 * edge_len2 - edge_len3 * edge_len3) / (2 * edge_len1 * edge_len2)) * 180 / CV_PI;

		// ��ʼ�������߶�Ӧ�ĵ��
		lines.resize(3);
		std::rotate(triangle_points.begin(), triangle_points.begin() + 1, triangle_points.end()); // ѭ����λ
		lines[0] = std::make_pair(triangle_points[1], triangle_points[2]);
		lines[1] = std::make_pair(triangle_points[0], triangle_points[2]);
		lines[2] = std::make_pair(triangle_points[0], triangle_points[1]);

	}

};

//three_to_fourth��������
Point2f getThirdPoint(vector<Triangle>& triangle, vector<Point2f>& point, Mat& img);
