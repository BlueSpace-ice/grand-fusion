#include<iostream>
#include<opencv.hpp>
#include"9to3.hpp"

using namespace std;
using namespace cv;
int quereng;
#define NONE_DUN

// 计算三个点组成的向量的叉积
double cross(const Point2f& p1, const Point2f& p2, const Point2f& p3) {
	double ans = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
	return ans;
}

void drawmap(vector<Triangle>& triangles, Mat img)
{
	// 在图像中绘制三角形
	for (const auto& triangle : triangles) {
		// 将三个点的坐标存储在一个vector中
		vector<Point> points;
		for (const auto& point : triangle.triangle_points) {
			Point2f p(point.x, point.y);
			points.push_back(p);
		}
		// 绘制三角形的边
		line(img, points[0], points[1], Scalar(0, 0, 255), 3);
		line(img, points[1], points[2], Scalar(0, 0, 255), 3);
		line(img, points[2], points[0], Scalar(0, 0, 255), 3);
		points.clear();
	}
}

//判断是否为钝角三角形
bool is_obtuse_triangle(Triangle a) {

	if (a.angle1 > 92 || a.angle2 > 92 || a.angle3 > 92)
		return true;
	else
		return false;
}

// 判断三个点组成的折线是否是逆时针方向
bool ccw(const Point2f& p1, const Point2f& p2, const Point2f& p3) {
	return cross(p1, p2, p3) > 0;
}

vector<Point2f> get_min_angles_points(Mat& img, const vector<Point2f>& points, const Point2f ob_p) {
	vector<Point2f> hull_points;
	cv::convexHull(points, hull_points, true);
	vector<Point> tmp;
	for (int i = 0; i < hull_points.size(); i++)//类型转换
		tmp.push_back(hull_points[i]);
	vector<vector<Point>> hull1;
	hull1.push_back(tmp);
	const cv::Point* pts[] = { hull1[0].data() };
	int npts[] = { static_cast<int>(hull1[0].size()) };
	cv::polylines(img, pts, npts, 1, true, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);//验证通过，凸包没问题

	double min_angle_sum = DBL_MAX;
	std::vector<cv::Point2f> min_angle_points;


	// 计算凸包
	std::vector<int> hull;
	cv::convexHull(points, hull);

	double tmp1 = INFINITY;
	Point2f pt1, pt2;
	double tmp2 = INFINITY;
	cout << endl << endl;
	// 计算每个点对应的内角和
	std::vector<double> angles(points.size(), 0.0);
	for (int i = 0; i < hull.size(); i++) {
		cv::Point2f prev = points[hull[(i - 1 + hull.size()) % hull.size()]];
		cv::Point2f p = points[hull[i]];
		cv::Point2f next = points[hull[(i + 1) % hull.size()]];
		cout << hull[(i - 1 + hull.size()) % hull.size()] << "      " << hull[i] << "       " << hull[(i + 1) % hull.size()] << endl;
		cv::Point2f v1 = prev - p;
		cv::Point2f v2 = next - p;
		v1 = v1 / cv::norm(v1);
		v2 = v2 / cv::norm(v2);

		double cos_angle = v1.dot(v2);
		double angle = acos(cos_angle) * 180.0 / CV_PI;
		angles[hull[i]] = angle;
		Mat cap = img.clone();
		//调试区
		circle(cap, p, 10, Scalar(187, 219, 136), -1);
		cout << "第" << i + 1 << "次夹角：" << angle << endl;
		//ob_p是已经加入结果的点
		//取夹角最小的两个点
		if (tmp1 > angle && p != ob_p) {
			tmp2 = tmp1;
			pt2 = pt1;
			tmp1 = angle;
			pt1 = p;
		}
		if (tmp2 > angle && tmp1 != angle && p != ob_p) {
			tmp2 = angle;
			pt2 = p;
		}
		namedWindow("get_min_angles", WINDOW_NORMAL);
		imshow("get_min_angles", cap);

		cap.release();
	}



	// 返回结果
	std::vector<cv::Point2f> result;
	result.push_back(pt1);
	result.push_back(pt2);

	return result;
}


// 计算两点之间的距离
double distance(Point2f p1, Point2f p2) {
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 计算两条线段之间的夹角
double angleBetween(Point2f p1, Point2f p2, Point2f p3, Point2f p4) {
	double dot = (p2.x - p1.x) * (p4.x - p3.x) + (p2.y - p1.y) * (p4.y - p3.y);
	double len1 = distance(p1, p2);
	double len2 = distance(p3, p4);
	double cos_value = dot / (len1 * len2);
	return acos(cos_value) * 180 / CV_PI;
}
//返回三个点的，错误集中区
vector<Point2f> nin_to_3(vector<Triangle> a, Mat cap)
{
	vector<bool> abool(a.size());//标记三角形是钝角三角形
	vector<Point2f> ans;
	int obtuse_count = 0;
	for (int i = 0; i < a.size(); i++)
		if (is_obtuse_triangle(a[i]))
		{
			abool[i] = 1;
			obtuse_count++;
			double max_angle = max(a[i].angle1, max(a[i].angle2, a[i].angle3));
			if (max_angle == a[i].angle1) {//第一个点，是钝角三角形的顶点
				ans.push_back(a[i].pt1);
			}
			else if (max_angle == a[i].angle2) {
				ans.push_back(a[i].pt2);
			}
			else {
				ans.push_back(a[i].pt3);
			}
			cout <<max_angle<<"    " << ans << endl;
		}
	//只有一个钝角三角形
	if (obtuse_count == 1) {
		cout << "只有一个钝角三角形" << endl;
		vector<Point2f> all_points;

		all_points.insert(all_points.end(), a[0].triangle_points.begin(), a[0].triangle_points.end());
		all_points.insert(all_points.end(), a[1].triangle_points.begin(), a[1].triangle_points.end());
		all_points.insert(all_points.end(), a[2].triangle_points.begin(), a[2].triangle_points.end());//9个点进去了

		vector<Point2f> true_all_points = get_min_angles_points(cap, all_points, ans[0]);
		ans.insert(ans.end(), true_all_points.begin(), true_all_points.end());
		return ans;
	}
	//只有两个直角三角形
	else if (obtuse_count == 2) {
		cout << "只有两个直角三角形" << endl;
		double dist = -1;//距离值
		int pindex = -1;//点索引
		int tindex = -1;//三角形索引
		Point2f mean = (ans[0] + ans[1]) / 2;
		for (int i = 0; i < a.size(); i++)
			if (abool[i] == 0)//未访问
			{
				tindex = i;
				for (int j = 0; j < a[i].triangle_points.size(); j++)
				{
					//调试可视化
					Mat ob_2 = cap.clone();
					circle(ob_2, mean, 20, Scalar(255, 0, 255), -1);
					circle(ob_2,a[i].triangle_points[j],20,Scalar(255, 0, 255), -1);
					namedWindow("ob_2", WINDOW_NORMAL);
					imshow("ob_2", ob_2);
					if (dist < distance(a[i].triangle_points[j], mean))
					{
						dist = distance(a[i].triangle_points[j], mean);
						pindex = j;
					}
				}
			}
		if (pindex != -1 && dist != -1)
			ans.push_back(a[tindex].triangle_points[pindex]);
		else {
			cout << "error" << endl;
			return *new vector<Point2f>;
		}
		cout << "ans.size()" << ans.size() << endl;
		return ans;
	}
	else
		cout << "使用斜矩形拟合" << endl;
	return *new vector<Point2f>;
}


//初始化
vector<Triangle> tri_init() {
	// 初始化三个三角形结构体

#ifdef NONE_DUN
	Triangle triangle1;
	triangle1.angle1 = 117.418;
	triangle1.angle2 = 160.763;
	triangle1.angle3 = 104.314;
	triangle1.edge_len1 = 496.212;
	triangle1.edge_len2 = 612.825;
	triangle1.edge_len3 = 693.375;
	triangle1.triangle_points.push_back(Point2f(496.212, 796.269));
	triangle1.triangle_points.push_back(Point2f(612.825, 782.55));
	triangle1.triangle_points.push_back(Point2f(479.062, 693.375));

	Triangle triangle2;
	triangle2.angle1 = 117.496;
	triangle2.angle2 = 109.129;
	triangle2.angle3 = 160.802;
	triangle2.edge_len1 = 522.591;
	triangle2.edge_len2 = 406.126;
	triangle2.edge_len3 = 359.567;
	triangle2.triangle_points.push_back(Point2f(522.591, 235.788));
	triangle2.triangle_points.push_back(Point2f(406.126, 251.317));
	triangle2.triangle_points.push_back(Point2f(419.945, 359.567));

	Triangle triangle3;
	triangle3.angle1 = 108.42;
	triangle3.angle2 = 110.051;
	triangle3.angle3 = 153.311;
	triangle3.edge_len1 = 925.38;
	triangle3.edge_len2 = 1032.75;
	triangle3.edge_len3 = 600.473;
	triangle3.triangle_points.push_back(Point2f(925.38, 724.28));
	triangle3.triangle_points.push_back(Point2f(1032.75, 709.211));
	triangle3.triangle_points.push_back(Point2f(1015.8, 600.473));

#endif

#ifdef ONE_DUN
	Triangle triangle1;
	triangle1.angle1 = 95.9913;
	triangle1.angle2 = 31.6782;
	triangle1.angle3 = 52.3306;
	triangle1.edge_len1 = 142.739;
	triangle1.edge_len2 = 75.3707;
	triangle1.edge_len3 = 113.605;
	triangle1.triangle_points.push_back(Point2f(476.618, 539.753));
	triangle1.triangle_points.push_back(Point2f(584.017, 502.718));
	triangle1.triangle_points.push_back(Point2f(444.745, 471.453));

	Triangle triangle2;
	triangle2.angle1 = 47.5428;
	triangle2.angle2 = 67.5039;
	triangle2.angle3 = 64.9533;
	triangle2.edge_len1 = 80.1009;
	triangle2.edge_len2 = 100.308;
	triangle2.edge_len3 = 98.3604;
	triangle2.triangle_points.push_back(Point2f(863.949, 391.36));
	triangle2.triangle_points.push_back(Point2f(955.804, 356.182));
	triangle2.triangle_points.push_back(Point2f(900.715, 298.032));

	Triangle triangle3;
	triangle3.angle1 = 30.7792;
	triangle3.angle2 = 87.5805;
	triangle3.angle3 = 61.6403;
	triangle3.edge_len1 = 51.8314;
	triangle3.edge_len2 = 101.196;
	triangle3.edge_len3 = 89.1301;
	triangle3.triangle_points.push_back(Point2f(424.303, 215.899));
	triangle3.triangle_points.push_back(Point2f(339.746, 244.085));
	triangle3.triangle_points.push_back(Point2f(358.198, 292.52));
#endif

#ifdef TWO_DUN
	Triangle triangle1;
	triangle1.edge_len1 = 146.49;
	triangle1.edge_len2 = 192.34;
	triangle1.edge_len3 = 118.858;
	triangle1.angle1 = 49.552;
	triangle1.angle2 = 92.3176;
	triangle1.angle3 = 38.1304;
	triangle1.triangle_points.push_back(Point2f(768.643, 240.353));
	triangle1.triangle_points.push_back(Point2f(650.438, 252.796));
	triangle1.triangle_points.push_back(Point2f(659.87, 398.982));

	// 初始化三角形2
	Triangle triangle2;
	triangle2.edge_len1 = 135.231;
	triangle2.edge_len2 = 121.156;
	triangle2.edge_len3 = 112.547;
	triangle2.angle1 = 70.6007;
	triangle2.angle2 = 57.6778;
	triangle2.angle3 = 51.7215;
	triangle2.triangle_points.push_back(Point2f(714.646, 946.899));
	triangle2.triangle_points.push_back(Point2f(816.634, 899.304));
	triangle2.triangle_points.push_back(Point2f(702.786, 826.324));

	// 初始化三角形3
	Triangle triangle3;
	triangle3.edge_len1 = 98.2678;
	triangle3.edge_len2 = 143.122;
	triangle3.edge_len3 = 72.4197;
	triangle3.angle1 = 39.1665;
	triangle3.angle2 = 113.094;
	triangle3.angle3 = 27.7392;
	triangle3.triangle_points.push_back(Point2f(1049.21, 772.582));
	triangle3.triangle_points.push_back(Point2f(1114.57, 741.389));
	triangle3.triangle_points.push_back(Point2f(1110.42, 643.209));

#endif

	// 将三角形结构体存储在vector中
	vector<Triangle> triangles;
	triangles.push_back(triangle1);
	triangles.push_back(triangle2);
	triangles.push_back(triangle3);
	return triangles;
}



vector<Point2f> quadraDominance(vector<Triangle>& triangles, cv::Mat &img) {
	for (int j = 0; j < triangles.size(); j++)
		triangles[j].init();
	vector<Point2f> true_points = nin_to_3(triangles, img);

	Mat m9to3 = img.clone();
	for(int i=0;i<true_points.size();i++)
		circle(m9to3, true_points[i], 20, Scalar(187, 219, 136), -1);
	namedWindow("m9to3", WINDOW_NORMAL);
	imshow("m9to3", m9to3);
	waitKey(1);
	Point2f fourth_p = getThirdPoint(triangles, true_points, img);

	true_points.push_back(fourth_p);
	cout << "true_points.size" << true_points.size() << endl;
	// 在图像中绘制三角形
	for (const auto& triangle : triangles) {
		// 将三个点的坐标存储在一个vector中
		vector<Point> points;
		for (const auto& point : triangle.triangle_points) {
			Point2f p(point.x, point.y);
			points.push_back(p);
		}
		// 绘制三角形的边
		line(img, points[0], points[1], Scalar(0, 0, 255), 3);
		line(img, points[1], points[2], Scalar(0, 0, 255), 3);
		line(img, points[2], points[0], Scalar(0, 0, 255), 3);
		points.clear();
	}


	for(int i=0;i<true_points.size();i++)
		circle(img, fourth_p, 20, Scalar(187, 219, 136), -1);

	imshow("qua_img", img);
	waitKey(1);
	return true_points;



}
