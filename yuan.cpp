#include"yuan.hpp"
std::ofstream myfile("D:\\����\\output1.txt");
std::ofstream myfile2("D:\\����\\output2.txt");
std::ofstream myfile3("D:\\����\\output3.txt");

int d1, d2, d3, d4;
int d = 0;
int capcols, caprows;
int mean_lenth = 0;
std::vector<cv::Point2f> vertices3(4);

//�ѽǶ�ת����ͳһ��          (����Ķ�������)
double transf(double a)
{
	if (a < 0)
		return 180.0 + a;
	else if (a >= 180)
		return 0;
	else
		return a;
}


//ͼ��Ԥ����                  (����Ķ�������)
Mat precessing(Mat image)
{
	// ת��Ϊ�Ҷ�ͼ��
	Mat gray;
#if COLOR==1
	inRange(image, Scalar(230, 230, 230), Scalar(255, 255, 255), gray);
#elif COLOR==0        
	inRange(image, Scalar(27, 31, 191), Scalar(117, 180, 255), gray);             //���Ե�ʱ��ǵÿ��زĸ�����(�����ͼƬ�ģ�
#elif
	inRange(image, Scalar())
#endif
		cv::namedWindow("gray", WINDOW_NORMAL);                                      //���������ߵ�Ч��
	cv::imshow("gray", gray);
	cv::waitKey(1);
	return gray;
}

//���������С��������        (����Ķ�������)
void find_min_diff_indices(double arr[], int n, int& ind1, int& ind2, int& ind3)
{
	if (n <= 3) {
		cout << "find_min_diff_indices error" << endl;
		return;
	}

	double min_var = numeric_limits<double>::infinity();
	int idx1 = -1, idx2 = -1, idx3 = -1;     // ������С�����������±�
	for (int i = 1; i < n - 1; i++)
	{
		double mean = (arr[i - 1] + arr[i] + arr[i + 1]) / 3.0;
		double variance = ((arr[i - 1] - mean) * (arr[i - 1] - mean) + (arr[i] - mean) * (arr[i] - mean) + (arr[i + 1] - mean) * (arr[i + 1] - mean)) / (mean * mean * mean);
		if (variance < min_var)
		{
			min_var = variance;
			idx1 = i - 1;
			idx2 = i;
			idx3 = i + 1;
		}
	}
	ind1 = idx1;
	ind2 = idx2;
	ind3 = idx3;
}


//�ҵ����������ε����ƶȷ���
double drawMostSimilarContours(const std::vector<std::vector<cv::Point>>& contours)
{
	if (contours.size() != 3)
	{
		std::cout << "drawMostSimilarContours error" << std::endl;
		return 0;
	}
	double ans = 0;
	ans += matchShapes(contours[0], contours[1], CONTOURS_MATCH_I3, 0);
	ans += matchShapes(contours[1], contours[2], CONTOURS_MATCH_I3, 0);
	ans += matchShapes(contours[2], contours[0], CONTOURS_MATCH_I3, 0);
	return ans;
}

//���������С�������Ƕ�
void find_min_diff_angles(double arr[], int n, int& ind1, int& ind2, int& ind3)
{
	if (n <= 3) {
		cout << "find_min_diff_indices error" << endl;
		return;
	}
	double min_var = numeric_limits<double>::infinity();
	int idx1 = -1, idx2 = -1, idx3 = -1; // ������С�����������±�
	for (int i = 1; i < n - 1; i++)
	{
		double mean = (arr[i - 1] + arr[i] + arr[i + 1]) / 3.0;
		double variance = abs(arr[i - 1] - mean) + abs(arr[i] - mean) + abs(arr[i + 1] - mean);
		if (variance < min_var)
		{
			min_var = variance;
			idx1 = i - 1;
			idx2 = i;
			idx3 = i + 1;
		}
	}
	ind1 = idx1;
	ind2 = idx2;
	ind3 = idx3;
}

//������С����ǵ�ƽ��ֵ      (����Ķ�������)
double findMinangles(vector<double>& angles)
{
	if (angles.size() < 3)
	{
		cout << "findMinangles error" << endl;
		return 0;
	}
	double nums[10] = { 0 };
	int min1 = min((int)angles.size(), 10);
	for (int i = 0; i < min1; i++)
		nums[i] = angles[i];
	int n = min1;
	int idx1, idx2, idx3;
	find_min_diff_angles(nums, n, idx1, idx2, idx3);
	if (idx1 >= angles.size() || idx2 >= angles.size() || idx3 >= angles.size())
	{
		return 0;
	}
	angles.push_back(INF);
	angles.erase(angles.begin() + idx1);
	angles.erase(angles.begin() + idx2 - 1);
	angles.erase(angles.begin() + idx3 - 2);
	double ans = (nums[idx1] + nums[idx2] + nums[idx3]) / 3.0;
	return ans;
}

//���ĸ����㣬�����ĸ������vector
vector<Point2f> findApexs(vector<RotatedRect> minRect, vector<vector<cv::Point>> contours, vector<Triangle> triangles, vector<int> index)
{
	if (index.size() != 4) {
		cout << "findApexs error" << endl;
		return *new vector<Point2f>;
	}

	//�����������
	float x = 0, y = 0;

	for (int i = 0; i < 4; i++)
	{
		x += minRect[index[i]].center.x;
		y += minRect[index[i]].center.y;
	}
	x /= 4;
	y /= 4;//Point(x,y)�����ĵ�


	//ȷ���ĸ���
	vector<Point2f> bigRect_points;
	for (int i = 0; i < index.size(); i++)
	{
		int max = 0, max_i = 0;
		float true_x = 0, true_y = 0;
		for (int j = 0; j < contours[index[i]].size(); j++)
		{
			float anotherx = contours[index[i]][j].x;
			float anothery = contours[index[i]][j].y;
			float differ = pow(x - anotherx, 2) + pow(y - anothery, 2);
			if (differ > max)
			{
				max = differ;
				true_x = anotherx;
				true_y = anothery;
			}
		}
		bigRect_points.push_back(Point2f(true_x, true_y));
	}

	return bigRect_points;
}

// ���ͼ���е������β�����ÿ�������ε������ߵĽǶȺͳ���
vector<Triangle> detectTriangles(const vector<vector<cv::Point>>& contours, Mat& cap)
{
	if (contours.size() <= 3) {
		cout << "detectTriangles error" << endl;
		return *new vector<Triangle>;
	}
	vector<Triangle> triangles;

	// ��ÿ���������д���
	for (int i = 0; i < contours.size(); i++)
	{
		// ʹ��minEnclosingTriangle������ȡ��С���������
		vector<Point2f> triangle_points;
		minEnclosingTriangle(contours[i], triangle_points);

		Triangle t;
		t.triangle_points = triangle_points;
		t.init();
		triangles.push_back(t);
	}
	return triangles;
}

//��������
float getDistance(Point2f a, Point2f b)
{
	float distance;
	distance = powf((a.x - b.x), 2) + powf((a.y - b.y), 2);
	distance = sqrtf(distance);
	return distance;
}

//�ҵ��ĸ�����Ľǣ������±�
int find_four_apex(vector<vector<cv::Point>> contours, vector<Triangle> triangle, vector<int> findMinindex, Mat cap, vector<Point2f> p3_to_circles)
{
	if (findMinindex.size() != 3)
	{
		std::cout << "find_four_apex error" << endl;
		return 0;
	}
	else
	{





		//�������ε������ǻ���
		vector<double> handle_angles;
		for (int i = 0; i < findMinindex.size(); i++)
		{
			handle_angles.push_back(transf(triangle[findMinindex[i]].angle1));
			handle_angles.push_back(transf(triangle[findMinindex[i]].angle2));
			handle_angles.push_back(transf(triangle[findMinindex[i]].angle3));
		}

		std::sort(handle_angles.rbegin(), handle_angles.rend());


		double angle1, angle2, angle3;
		angle1 = findMinangles(handle_angles);
		angle2 = findMinangles(handle_angles);

		double min1 = INF;
		int min1index = 0;
		for (int i = 0; i < triangle.size(); i++)
		{
			bool isture = 0;
			for (int j = 0; j < p3_to_circles.size(); j++)
			{
				for (int k = 0; k < triangle[i].triangle_points.size(); k++)
				{
					if (getDistance(p3_to_circles[j], triangle[i].triangle_points[k]) <= mean_lenth)
						isture = 1;
				}
			}
			if (isture) {
				double ans11 = min(triangle[i].angle1 - angle1, triangle[i].angle2 - angle1);
				double ans1 = min(ans11, triangle[i].angle3 - angle1);
				double ans22 = min(triangle[i].angle1 - angle2, triangle[i].angle2 - angle2);
				double ans2 = min(ans22, triangle[i].angle3 - angle2);
				double diff = abs(ans1) + abs(ans2);

				//�����������
				double s = (triangle[i].edge_len1 + triangle[i].edge_len2 + triangle[i].edge_len3) / 2.0;
				double area = s * (s - triangle[i].edge_len1) * (s - triangle[i].edge_len2) * (s - triangle[i].edge_len3);

				//�ۺ��о�
				diff = diff / area;
				if (min1 > diff && (triangle[i].edge_len1 > 8 && triangle[i].edge_len2 > 8 && triangle[i].edge_len3 > 8) && find(findMinindex.begin(), findMinindex.end(), i) == findMinindex.end())  //�Ҳ�������˼
				{
					min1 = diff;
					min1index = i;
				}
			}
		}

		return min1index;
	}
}

//һ��ת�����ã�����������У�����������С����������������±�
vector<int> findMinareas(vector<double> areas)    //areas�Ѿ�������(����)
{
	if (areas.size() <= 3) {
		cout << "findMinareas error" << endl;
		return *new vector<int>;
	}

	double nums[7] = { 0 };
	int min1 = min((int)areas.size(), 7);
	for (int i = 0; i < min1; i++)
		nums[i] = areas[i];
	int n = min1;
	int idx1, idx2, idx3;
	find_min_diff_indices(nums, n, idx1, idx2, idx3);

	vector<int> ans;
	ans.push_back(idx1);
	ans.push_back(idx2);
	ans.push_back(idx3);
	return ans;
}
//��roi�ĵ㻭����
vector<Point2f> detectmask(vector<RotatedRect> minRect, vector<int> Minindex, Mat cap)
{
	vector<Point2f> ans, res;
	double maxDiagonal = 0;
	for (int i = 0; i < Minindex.size(); i++)
	{
		RotatedRect rect = minRect[Minindex[i]];
		double diagonal = std::sqrt(rect.size.width * rect.size.width + rect.size.height * rect.size.height);
		if (diagonal > maxDiagonal) {
			maxDiagonal = diagonal;
		}
	}
	mean_lenth = maxDiagonal;

	ans.push_back(minRect[Minindex[0]].center + minRect[Minindex[1]].center - minRect[Minindex[2]].center);
	ans.push_back(minRect[Minindex[1]].center + minRect[Minindex[2]].center - minRect[Minindex[0]].center);
	ans.push_back(minRect[Minindex[2]].center + minRect[Minindex[0]].center - minRect[Minindex[1]].center);
	for (int i = 0; i < ans.size(); i++)
		if (ans[i].x > 0 && ans[i].x < capcols && ans[i].y>0 && ans[i].y < caprows)
			res.push_back(ans[i]);
	//�������
	Mat capshow = cap.clone();
	if (!res.empty())
		for (int i = 0; i < res.size(); i++)
			circle(capshow, res[i], mean_lenth, Scalar(255, 255, 255), -1);
	imshow("capshow", capshow);
	waitKey(1);
	return res;

}
//ɸѡ���ĸ�ֱ�ǵĵ����������±�
//�������9to3�ȽϺ�
vector<int> handleLight(vector<vector<cv::Point>> contours, vector<RotatedRect> minRect, vector<Triangle> triangle,cv:: Mat cap)
{
	//�������ų�    ���ڵķ�����1.ȥ��̫С�ľ���2.����������������Ƶ��������
	vector<double> areas;
	for (int i = 0; i < triangle.size(); i++) {
		if (triangle[i].getArea() > 20)
			areas.push_back(triangle[i].getArea());
	}
	//findMinindex���±�

	if (areas.size() <= 3) {
		cout << "handleLight size error" << endl;
		return *new vector<int>;
	}

	vector<int> Minindex = findMinareas(areas);

	//������������������������ʶ���Ƿ�׼ȷ������matchShapes
	vector<vector<Point>> tmp_contours;
	for (int i = 0; i < Minindex.size(); i++)
		tmp_contours.push_back(contours[Minindex[i]]);

	myfile3 << drawMostSimilarContours(tmp_contours) << endl;

	//�����������

	if (Minindex.size() == 3) {
		//�ҵ��ĸ�����+ת������+�����������
		bool jg_use = 0;//Ĭ���Ϸ���
		for (int i = 0; i < Minindex.size(); i++)
			if (is_obtuse_triangle(triangle[Minindex[i]]))
				jg_use = 1;
		//����QuadraDominance,����û�в���ԭ�����������棬�ȿ�һ��Ч��
			vector<Triangle> tmp_triangles;
			for(int i=0;i<Minindex.size();i++)
				tmp_triangles.push_back(triangle[Minindex[i]]);
			Mat tmp_mat = Mat::zeros(cap.size(), CV_8UC3);
			drawmap(tmp_triangles, cap);
			vector<Point2f> tri_points = quadraDominance(tmp_triangles, tmp_mat);
			for (int i = 0; i < tri_points.size(); i++)
				circle(tmp_mat, tri_points[i], 20,Scalar(255,255,0), -1);
			namedWindow("tmp_mat", WINDOW_NORMAL);
			imshow("tmp_mat", tmp_mat);
			waitKey(1);
		
	//���ô�ͳ�ı��η���
		
			vector<Point2f> p3_to_circles = detectmask(minRect, Minindex, cap);//�����������ѡ�������(��һ����������,���ص��Ǹ���ƽ���ı��η���ȷ����3����
			vector<vector<cv::Point>> tmp_contours;
			Minindex.push_back(find_four_apex(tmp_contours, triangle, Minindex, cap, p3_to_circles));//����ѵ��ĸ���������
		
		//��һ��
		imshow("cap", cap);
		waitKey(1);
		for (int i = 0; i < Minindex.size(); i++)
			if (Minindex[i] < 0)
				return *new vector<int>;
		return Minindex;
	}
	else
		return *new vector<int>;

}

//��������������ö�ֵͼ���ȫ����
vector<Point2f> handleMat(Mat src, Mat image)
{
	vector<vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;

	// �ҵ���������
	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
#ifdef GRAY
	std::cout << "����������" << contours.size() << endl;
	if (contours.size() < 10000000000) {
		vector<Point2f> a;
		return a;
	}
#endif // GRAY
	// Ϊÿ�������ҵ�һ��б���κ���С���������
	vector<RotatedRect> minRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(contours[i]);
	}
	vector<Triangle> triangles = detectTriangles(contours, image);//���ԣ��������

	//������б���ε����������б���ζ����򣬲�����������б���α䳤���������һһ��Ӧ
	vector<RotatedRect> handle_minRect(minRect.size());
	vector<vector<cv::Point>> handle_contours(contours.size());
	vector<Triangle> handle_triangles(triangles.size());
	//vector<Triangle> handle_triangles(triangles.size());
	//�Լ�д��ѡ������(�����Ż�),����minRect�������С����
	for (int i = 0; i < contours.size(); i++)
	{
		int max = 0, max_i = 0;
		for (int j = 0; j < contours.size(); j++)
		{
			if (minRect[j].size.area() > max)
			{
				max = minRect[j].size.area();
				max_i = j;
			}
		}
		handle_minRect[i] = minRect[max_i];
		handle_contours[i] = contours[max_i];
		handle_triangles[i] = triangles[max_i];
		minRect[max_i] = *new RotatedRect;
	}


	//�ж��������ĸ���
	vector<int> true_index;
	if (minRect.size() >= 4)
		true_index = handleLight(handle_contours, handle_minRect, handle_triangles, image);
	else
	{
		std::cout << "û��ʶ���ĸ���" << endl;
		return *new vector<Point2f>;
	}
	if (true_index.empty())
	{
		std::cout << "�±����" << endl;
		return *new vector<Point2f>;
	}
	//����,���ĸ�����
	cout << true_index.size() << endl;
	if (true_index.empty())
		return *new vector<Point2f>;
	vector<Point2f> true_vertexs = findApexs(handle_minRect, handle_contours, handle_triangles, true_index);
	Scalar colors[] = { Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 128, 255) };
	Mat cap1 = image.clone();
	Mat cap2 = image.clone();
	Mat cap3 = image.clone();
#ifdef DEBUG1//����
	// ������ɫ����,RGB˳��ֱ��������̣��죬����

	// �������ͼ��

	for (int i = 0; i < true_vertexs.size(); i++)
		circle(cap1, true_vertexs[i], 5, colors[i], -1);
	cv::namedWindow("cap1", WINDOW_NORMAL);
	cv::imshow("cap1", cap1);
#endif

#ifdef DEBUG2//б����
	// ������ת����
	for (int i = 0; i < true_index.size(); i++)
	{
		// �������ȷ����ɫ
		Scalar color;
		color = colors[i];

		Point2f rect_points[4];
		handle_minRect[true_index[i]].points(rect_points);
		for (int j = 0; j < 4; j++)
		{
			line(cap2, rect_points[j], rect_points[(j + 1) % 4], color, 7, LINE_AA);
		}
	}
	cv::namedWindow("cap2", WINDOW_NORMAL);
	cv::imshow("cap2", cap2);
#endif // DEBUG2

#ifdef DEBUG3
	for (int i = 0; i < true_index.size(); i++)
	{
		Scalar color;
		color = colors[i];
		for (int j = 0; j < 3; j++)
		{
			line(cap3, handle_triangles[i].triangle_points[j], handle_triangles[i].triangle_points[(j + 1) % 3], color, 7);
		}
	}
	cv::namedWindow("cap3", WINDOW_NORMAL);
	cv::imshow("cap3", cap3);
#endif
	cv::waitKey(1);

	//����ֵ����
	return true_vertexs;
}

//��ת����ת����ŷ����
void rotationMatrixToEulerAngles(Mat& R, double& roll, double& pitch, double& yaw)
{
	double r11 = R.at<double>(0, 0);
	double r12 = R.at<double>(0, 1);
	double r13 = R.at<double>(0, 2);
	double r21 = R.at<double>(1, 0);
	double r22 = R.at<double>(1, 1);
	double r23 = R.at<double>(1, 2);
	double r31 = R.at<double>(2, 0);
	double r32 = R.at<double>(2, 1);
	double r33 = R.at<double>(2, 2);

	pitch = asin(-r31);
	if (cos(pitch) != 0) {
		roll = atan2(r32 / cos(pitch), r33 / cos(pitch));
		yaw = atan2(r21 / cos(pitch), r11 / cos(pitch));
	}
	else {
		roll = 0;
		yaw = atan2(-r12, r22);
	}
}

//�����㷨2 ���ĸ�����
bool cmp(Point2f a, Point2f b) {               //��Ϊ�Ƚ�������yΪ��һ���Ƚ�������,��convex_hull������sort����ʱʹ�ã�����
	if (a.y == b.y) return a.x < b.x;
	return a.y < b.y;
}

int cross(Point2f a, Point2f b, Point2f c) {                    //ͨ���������ȷ����c��ֱ��ab����һ�ࡣֵΪ��������ࡣ
	return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
}

vector<Point2f> convex_hull(vector<Point2f>& p) {
	int n = p.size();
	sort(p.begin(), p.end(), cmp);   //��������
	Point2f res;
	if (p[0].x > p[1].x) {
		res = p[0];
		p[0] = p[1];
		p[1] = res;
	}
	int k = 0;					     //��ǰ͹���е������	
	vector<Point2f> q(n * 2);
	for (int i = 0; i < n; i++) {
		while (k > 1 && cross(q[k - 2], q[k - 1], p[i]) <= 0) k--;  //ͨ������ȷ��p[i]�������ֱ��q[k-2]q[k-1]��λ�ã����������q[k-1]��Ҫɾȥ
		q[k++] = p[i];
	}
	for (int i = n - 2, t = k; i >= 0; i--) {
		while (k > t && cross(q[k - 2], q[k - 1], p[i]) <= 0) k--;  //ͬ��
		q[k++] = p[i];
	}
	q.resize(k - 1);                                                //ɾȥ���õ�
	return q;
}

void sort_points(vector<Point2f>& p) {
	vector<Point2f> q = convex_hull(p);
	int n = q.size();
	int pos = 0;
	for (int i = 1; i < n; i++) {
		if (q[i].x < q[pos].x) pos = i;
	}
	vector<Point2f> ans(n);
	int cnt = 0;
	for (int i = pos; i < n; i++) ans[cnt++] = q[i];
	for (int i = 0; i < pos; i++) ans[cnt++] = q[i];
	p = ans;
	return;
}

//pnp�㷨
void solveXYZ(std::vector<cv::Point2f> vertices, cv::Mat image)
{

	//cv::Mat image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\1.png");
	double half_x;
	double half_y;
	double width_target;
	double height_target;

	double cam1[3][3] = {                                      //�ڲξ��� (�Ķ��˲���֮��֪��Ϊʲô��׼�ˣ������¼�)
		1459.2, 0, 625.2,
		0, 1458.1, 514.2,
		0, 0, 1 };

	double distCoeff1[5] = { 0.01953, -0.0917, 0, 0, 0 };     //������� (ǰ��Ե���һ��Ҳ��׼�ˣ�ԭ����)

	cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, cam1);
	cv::Mat distortion_coeff = cv::Mat(5, 1, CV_64FC1, distCoeff1);

	width_target = 24;             //����
	height_target = 24;

	std::vector<cv::Point2f> Points2D;    //ͼ������ϵ�����
	Points2D.push_back(vertices[0]);
	Points2D.push_back(vertices[1]);
	Points2D.push_back(vertices[2]);
	Points2D.push_back(vertices[3]);

	std::vector<cv::Point3f> Point3d;     //��������ϵ�����

	half_x = (width_target) / 2.0;
	half_y = (height_target) / 2.0;

	Point3d.push_back(cv::Point3f(-half_x, half_y, 0));
	Point3d.push_back(cv::Point3f(-half_x, -half_y, 0));
	Point3d.push_back(cv::Point3f(half_x, -half_y, 0));
	Point3d.push_back(cv::Point3f(half_x, half_y, 0));

	cv::Mat rot1 = cv::Mat::eye(3, 3, CV_64FC1);           //��ת����
	cv::Mat trans1 = cv::Mat::zeros(3, 1, CV_64FC1);       //ƽ�ƾ���

	cv::solvePnP(Point3d, Points2D, cam_matrix, distortion_coeff, rot1, trans1, false);
	cv::Mat_<double> rot_mat;
	cv::Mat_<double> trans_mat;
	cv::Rodrigues(rot1, rot_mat);
	cv::Rodrigues(trans1, trans_mat);

	cv::Mat xyz = image.clone();
	std::vector<cv::Point3f> Point3ds;
	std::vector<cv::Point2f> outputPoints;

	Point3ds.push_back(cv::Point3d(0, 0, 0));
	Point3ds.push_back(cv::Point3d(12, 0, 0));
	Point3ds.push_back(cv::Point3d(0, 12, 0));
	Point3ds.push_back(cv::Point3d(0, 0, 12));

	cv::projectPoints(Point3ds, rot1, trans1, cam_matrix, distortion_coeff, outputPoints);
	std::cout << outputPoints << endl;

	line(xyz, outputPoints[3], outputPoints[0], cv::Scalar(0, 0, 255), 3, 8);
	line(xyz, outputPoints[2], outputPoints[0], cv::Scalar(0, 255, 0), 3, 8);
	line(xyz, outputPoints[1], outputPoints[0], cv::Scalar(255, 0, 0), 3, 8);

	cv::putText(xyz, "Z", outputPoints[3], cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 3, 8);
	cv::putText(xyz, "Y", outputPoints[2], cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 255, 0), 3, 8);
	cv::putText(xyz, "X", outputPoints[1], cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(255, 0, 0), 3, 8);

	std::string rot, trans;
	rot << rot1;
	trans << trans1;

	cv::putText(xyz, "rot" + rot, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1);
	cv::putText(xyz, "trans" + trans, cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1);

	double dist = sqrt(trans1.at<double>(0, 0) * trans1.at<double>(0, 0) + trans1.at<double>(1, 0) * trans1.at<double>(1, 0) + trans1.at<double>(2, 0) * trans1.at<double>(2, 0));

	/*double x, y, z;                                //�����Ǹ�����Ҫ���㲻׼�����������Բ��Ծ������
	z = std::atan2(rot_mat.at<double>(1, 0), rot_mat.at<double>(0, 0));
	y = std::atan2(rot_mat.at<double>(-rot_mat.at<double>(2, 0)), std::sqrt(rot_mat.at<double>(2, 0) * rot_mat.at<double>(2, 0) + rot_mat.at<double>(2, 2) * rot_mat.at<double>(2, 2)));
	x = std::atan2(rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2));
	x = x * 180.0 / CV_PI;
	y = y * 180.0 / CV_PI;
	z = z * 180.0 / CV_PI;

	cout << "x = " << x << endl << "y = " << y << endl << "z = " << z << endl;*/

	double sy = std::sqrt(rot_mat.at<double>(0, 0) * rot_mat.at<double>(0, 0) + rot_mat.at<double>(1, 0) * rot_mat.at<double>(1, 0));

	bool singularot_mat = sy < 1e-6;

	double x, y, z;
	if (!singularot_mat)
	{
		x = std::atan2(rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2));
		y = std::atan2(-rot_mat.at<double>(2, 0), sy);
		z = std::atan2(rot_mat.at<double>(1, 0), rot_mat.at<double>(0, 0));
	}
	else
	{
		x = std::atan2(-rot_mat.at<double>(1, 2), rot_mat.at<double>(1, 1));
		y = std::atan2(-rot_mat.at<double>(2, 0), sy);
		z = 0;
	}

	// Convert angles to degrees
	x = x * 180.0 / CV_PI;
	y = y * 180.0 / CV_PI;
	z = z * 180.0 / CV_PI;
	//cout << trans1 << endl;

	/*bool isput = 1;
	for (int i = 0; i < 3; i++) {
		if (trans1.at<double>(i, 0) < 3000 && trans1.at<double>(i, 0) > -3000)
			myfile << trans1.at<double>(i, 0) << ",";
		isput = 0;
	}
	for (int i = 0; i < 3; i++) {
		if (isput = 1 && -180 < x < 180 && -180 < y < 180 && -180 < z < 180)
			myfile << rot1.at<double>(i, 0) << "," << std::endl;
	}*/
	std::string s1, s2, s3, s4;
	s1 = "z: " + std::to_string(z) + " ";
	s2 = "y: " + std::to_string(y) + " ";
	s3 = "x: " + std::to_string(x) + " ";
	s4 = "dist: " + std::to_string(dist) + " ";
	cv::putText(xyz, (s1 + s2 + s3 + s4), cv::Point(30, 150), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
	cv::imshow("xyz", xyz);
	cv::waitKey(1);
}

//�ṩ��main��ȫ���ú���
void all(Mat image)
{
	// ����ͼ��
	if (image.empty())
		return;
	Mat processmat = precessing(image);//Ԥ������

	vector<Point2f> vertexs;//�����ĸ�����
	vertexs = handleMat(processmat, image);//�����ĸ�����
	if (vertexs.size() != 4) {
		cout << "handleMat vertexs error" << endl;
		return;
	}
	sort_points(vertexs);

	//vertexs = uuusortxy(vertexs);
	solveXYZ(vertexs, image);
	return;
}
#ifdef ISMAP
int main()
{
	VideoCapture capture("D:\\����\\�һ�վ��Ƶ\\�һ�վ\\�һ�վ\\890.mp4");
	Mat image;
	while (1) {
		capture.read(image);
		if (!image.empty())
		{
			capcols = image.cols;
			caprows = image.rows;
			all(image);
			cv::namedWindow("image", WINDOW_NORMAL);
			imshow("image", image);
			std::cout << endl << endl << endl;
			int c = cv::waitKey(1);
			if (c == 27)
				break;
		}
		else
			return 0;
	}
}

#else
int main()
{
	// ����ͼ��
	char i = '3';
	for (int i = 1; i <= 1372; i++)
	{
		string filename = "D:\\����\\�һ�վ��Ƶ\\�һ�վ\\�һ�վ\\output_folder\\1 (" + to_string(i) + ").png";
		Mat image = imread(filename);
		if (!image.empty())
		{
			capcols = image.cols;
			caprows = image.rows;
			all(image);
			cv::namedWindow("image", WINDOW_NORMAL);
			cv::imshow("image", image);
			int c = cv::waitKey(1);
		}
	}
}
#endif 