#include"three_to_fourth.hpp"


//求点到的直线的距离
double pointToLineDistance(Point2f p, Point2f l1, Point2f l2) {
	double numerator = abs((l2.y - l1.y) * p.x - (l2.x - l1.x) * p.y + l2.x * l1.y - l2.y * l1.x);
	double denominator = sqrt(pow(l2.y - l1.y, 2) + pow(l2.x - l1.x, 2));
	return numerator / denominator;
}


//点到点的距离
double pointToPointDistance(Point2f pt1, Point2f pt2) {
	return sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
}


Point2f getIntersection(pair<Point2f, Point2f> line1, pair<Point2f, Point2f> line2) {
	double A1 = line1.second.y - line1.first.y;
	double B1 = line1.first.x - line1.second.x;
	double C1 = A1 * line1.first.x + B1 * line1.first.y;

	double A2 = line2.second.y - line2.first.y;
	double B2 = line2.first.x - line2.second.x;
	double C2 = A2 * line2.first.x + B2 * line2.first.y;

	double determinant = A1 * B2 - A2 * B1;

	if (determinant == 0) {
		// 两条直线平行
		return Point2f(-1, -1);
	}
	else {
		double x = (B2 * C1 - B1 * C2) / determinant;
		double y = (A1 * C2 - A2 * C1) / determinant;
		return Point2f(x, y);
	}
}


// 主函数，找第四个点可能的坐标
Point2f getThirdPoint(vector<Triangle>& triangle, vector<Point2f>& point, Mat& img) {
	if (triangle.size() != 3 || point.size() != 3)
		return *new Point2f;

	// 分别求出剩余6个点到3条线的距离
	Triangle big_tri;
	big_tri.triangle_points = point;
	big_tri.init();
	Mat big_cmp;

	double dist = 1;
	vector<double> dists;
	bool is_deng = 0;
	//这一层遍历三个顶点
	for (int i = 0; i < big_tri.triangle_points.size(); i++) {

		dist = 1;
		//下面两重遍历所有三角形的所有点 
		for (int j = 0; j < triangle.size(); j++) {
			for (int w = 0; w < triangle[j].triangle_points.size(); w++) {
				is_deng = 0;
				big_cmp = img.clone();
				//检测是否包含
				for (int p = 0; p < big_tri.triangle_points.size(); p++)
					if (big_tri.triangle_points[p] == triangle[j].triangle_points[w])
						is_deng = 1;
				if (!is_deng)
				{
					dist *= pointToLineDistance(triangle[j].triangle_points[w], big_tri.lines[i].first, big_tri.lines[i].second);
					line(big_cmp, big_tri.lines[i].first, big_tri.lines[i].second, Scalar(255, 255, 0), 5);//青色
					circle(big_cmp, triangle[j].triangle_points[w], 10, Scalar(0, 255, 255), -1);//黄色
					namedWindow("big_cmp", WINDOW_NORMAL);
					imshow("big_cmp", big_cmp);
					//waitKey(0);
				}
				big_cmp.release();
			}
		}
		dists.push_back(dist);
		cout << endl;
	}

	//找出特别的那个点,距离的最大值
	Point2f spe;
	double max_tmp = -1;
	for (int i = 0; i < dists.size(); i++)
	{
		if (max_tmp < dists[i])
		{
			spe = big_tri.triangle_points[i];
			max_tmp = dists[i];
		}
	}
	vector<Point2f> dis_spe;
	//把不是spe的两个点push进一个数组
	for (int i = 0; i < point.size(); i++)
		if (point[i] != spe)
			dis_spe.push_back(point[i]);

	//开始找延长线的一点
	//遍历所有点
	vector<Point2f> tmp_lines;
	Point2f tar_p;
	for (int j = 0; j < triangle.size(); j++) {
		//不包含spe的,一共两个三角形
		double p_dist = -1;
		bool jg = 0;
		if (std::find(triangle[j].triangle_points.begin(), triangle[j].triangle_points.end(), spe) == triangle[j].triangle_points.end())
			for (int w = 0; w < triangle[j].triangle_points.size(); w++) {
				//不包含point里面的点,一共两个点
				//算距离，找到了一个三角形中距离spe最远的点tar_p
				if (std::find(point.begin(), point.end(), triangle[j].triangle_points[w]) == point.end())
				{
					if (pointToPointDistance(triangle[j].triangle_points[w], spe) > p_dist)
					{
						jg = 1;
						tar_p = triangle[j].triangle_points[w];
						p_dist = pointToPointDistance(triangle[j].triangle_points[w], spe);
					}
				}
			}
		if (jg)
			tmp_lines.push_back(tar_p);
	}
	//画画看
	for (int i = 0; i < tmp_lines.size(); i++)
		circle(img, tmp_lines[i], 19, Scalar(255, 255, 255), -1);
	for (int i = 0; i < dis_spe.size(); i++)
		circle(img, dis_spe[i], 19, Scalar(255, 255, 255), -1);

	vector<pair<Point2f, Point2f>> sum_lines;
	//画延长线
	for (int i = 0; i < triangle.size(); i++)
		for (int j = 0; j < dis_spe.size(); j++)
			for (int k = 0; k < tmp_lines.size(); k++)
			{
				if (find(triangle[i].triangle_points.begin(), triangle[i].triangle_points.end(), dis_spe[j]) != triangle[i].triangle_points.end()
					&& find(triangle[i].triangle_points.begin(), triangle[i].triangle_points.end(), tmp_lines[k]) != triangle[i].triangle_points.end())
				{
					sum_lines.push_back({ dis_spe[j],tmp_lines[k] });
				}
			}
	for (int i = 0; i < sum_lines.size(); i++)
		cout << sum_lines[i].first << "   " << sum_lines[i].second << endl;
	Point2f ans_p;
	if (sum_lines.size() == 2)
		ans_p = getIntersection(sum_lines[0], sum_lines[1]);
	circle(img, ans_p, 30, Scalar(0, 0, 255), -1);
	namedWindow("img", WINDOW_NORMAL);
	imshow("img", img);

	return spe;
}
