#include <algorithm> 
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <fstream>						// output to file
#include <opencv\cv.h>					// opencv library
#include <opencv2\opencv.hpp>			// opencv library
#include <opencv2/imgproc/imgproc.hpp>	// opencv library
#include <cmath>
using namespace std;
using namespace cv;

RNG rng(12345);

//Common Method
void readImages();
void readFiles();

//Normalisasi
float valnorm;

//Image Enhancement 
Mat medianFilter(Mat source, int mask);

Mat higboostFilter(Mat source, Mat medImg, int kons);
Mat morph(Mat source);
Mat thresholding(Mat source);
Mat canny(Mat source);

//Image Segmentation
Mat regionFilter(Mat source);
Mat coLinear(Mat source);

//Pencarian Nilai Fitur Berdasarkan Pergerakan Sistole Menuju Diastole
void GetGoodFeature();
void opticalFlowCalc();
void FeatureExtraction();

//Menggunakan Berbagai Metode Ekstraksi Fitur 
void ExtractionMethodI();
void ExtractionMethodII();
void ExtractionMethodIII();
void ExtractionMethodIV();


int intersectionLine(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4);	// determining whether some points in one line or not
void straightLine(int x1, int y1, int x2, int y2, float *slope, float *intercept);		// determining whether some points in one line or not

//Variabel
vector<String> files;
vector<Mat> images;
vector<Mat> sources;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
vector<float> lengthDiffirence[9];
int mouseActId = 0, X1 = 227, Y1 = 145, X2 = 249, Y2 = 168, CCX[100], CCY[100];;
Point centerPt = Point(X1, Y1);
vector<Point2f> goodFeatures[10];

Mat rawImage;
Mat source, res;
int out[2];

float direction[24][9], directionI[24][9], length[24][9];
float R = 71;
#define PHI 3.1415926

int temp1, temp2, temp3, jumlah = 12;

Point2f coordinate1[500][10];
Point2f coordinate2[500][10];

bool sortbysec(const pair<float, float> &a,
	const pair<float, float> &b)
{
	return (a.second < b.second);
}

int slope(float x1, float y1, float x2, float y2)
{
	float tanx, s;
	tanx = (y2 - y1) / (x2 - x1);
	s = atan(tanx);
	s = (180 / PHI)*s;
	return s;
}


int main()
{
	clock_t tStart = clock();
	for (int i = 3; i <= 10;i++) {

		readImages();
		rawImage = images[0];
		//namedWindow("Image", WINDOW_AUTOSIZE);
		//imshow("Image", rawImage);

		res = medianFilter(rawImage, 27);
		//namedWindow("Median Filter", WINDOW_AUTOSIZE);
		//imshow("Median Filter", res);

		res = higboostFilter(rawImage, res, i);
		cvtColor(res, res, CV_BGR2GRAY);
		//namedWindow("HighBost Filter", WINDOW_AUTOSIZE);
		//imshow("HighBost Filter", res);

		res = morph(res);
		//namedWindow("Morfologi", WINDOW_AUTOSIZE);
		//imshow("Morfologi", res);

		res = thresholding(res);
		//namedWindow("Thresholding", WINDOW_AUTOSIZE);
		//imshow("Thresholding", res);

		res = canny(res);
		//namedWindow("Canny", WINDOW_AUTOSIZE);
		//imshow("Canny", res);

		//readFiles();

		res = regionFilter(res);
		//namedWindow("regionFilter", WINDOW_AUTOSIZE);
		//imshow("regionFilter", res);

		res = coLinear(res);
		//namedWindow("coLinear", WINDOW_AUTOSIZE);
		//imshow("coLinear", res);

		findContours(res, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		for (int i = 0; i < contours.size(); i++) {
			if (contours[i].size() > R) {
				centerPt.x = X1; centerPt.y = Y1;
				out[i] = (int)pointPolygonTest(contours[i], centerPt, 0);
			}
		}
		if (out[1] <= 0) {
			//cout << "TERBUKA" << endl;
		}
		else {
			//cout << "TERTUTUP" << endl;
			break;
		}
	}

	//Search GoodFeature Cardiac
	GetGoodFeature();
	
	//Tracking Optical Flow 
	opticalFlowCalc();
	
	//FeatureExtraction /Masuk dan /Keluar
	FeatureExtraction();
	
	//Metode ekstraksi yang menggunakan 24 goodfeature (Direction & Length) sebagai fitur /Masuk dan /Keluar
	//ExtractionMethodI();

	//Metode ekstraksi yang menggunakan 24 goodfeature (Direction & Length) sebagai fitur /Masuk(+-) dan /Keluar(+-)
	//ExtractionMethodII();

	//Metode ekstraksi yang menggunakan 24 goodfeature (Direction & Length) sebagai fitur Masuk dengan Melakukan Pembagian Keluar
	//ExtractionMethodIII();

	//Metode ekstraksi yang membentuk kontour dan menghitung jarak antar dinding sistole menuju diastole (Metode nya mbak Arvina)
	//ExtractionMethodIV();

	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	imshow("HASIL ", res);
	waitKey(0);
	return 0;
}

void readImages() {

	String path("PIC*.jpg");
	glob(path, files, true);
	if (files.size() > 0) {
		for (size_t i = 0; i < files.size(); i++) {
			if (!imread(files[i]).empty()) {
				images.push_back(imread(files[i]));
			}
		}
	}
}

Mat medianFilter(Mat source, int mask) {
	Mat res;
	source.copyTo(res);
	medianBlur(source, res, mask);
	return res;
}


Mat higboostFilter(Mat source, Mat lpf_source, int kons) {
	// Fundamental : HBF image = k(original image) – LPF image
	Mat res;
	source.copyTo(res);

	// Manual process
	for (int i = 0; i < source.rows; i++) {
		for (int j = 0; j < source.cols; j++) {
			Vec3b lpf_rgb = lpf_source.at<Vec3b>(i, j);
			Vec3b src_rgb = source.at<Vec3b>(i, j);

			for (int k = 0; k < 3; k++) {							// Sum of RGB
				//float val = (kons * src_rgb.val[k]) - lpf_rgb.val[k];						
				float val = (kons * lpf_rgb.val[k]);
				if (val > 255) val = 255;
				if (val < 0) val = 0;
				res.at<Vec3b>(i, j)[k] = val;
			}
		}
	}

	return res;
}


Mat morph(Mat source) {
	Mat res;
	source.copyTo(res);
	Mat ellipse = getStructuringElement(MORPH_ELLIPSE, Size(12, 12), Point(3, 3));
	morphologyEx(source, res, MORPH_OPEN, ellipse);
	morphologyEx(res, res, MORPH_CLOSE, ellipse);

	return res;
}

Mat thresholding(Mat source) {
	Mat res;
	source.copyTo(res);
	threshold(source, res, 90, 255, CV_THRESH_BINARY);

	return res;
}

Mat canny(Mat source) {
	Mat res;
	source.copyTo(res);
	Canny(source, res, 0, 255, 3);
	Mat *can = &res;
	uchar *ptr = can->data;

	return res;
}

void readFiles() {
	String filename = ".\/" + files[0].substr(2, files[0].size() - 6) + ".txt";
	string dataFile;
	ifstream myReadFile;
	vector<string> tokens;

	myReadFile.open(filename);
	while (!myReadFile.eof()) // To get you all the lines.
	{
		while (getline(myReadFile, dataFile, ' ')) {
			tokens.push_back(dataFile);
		}
	}

	if (tokens.size() > 0) {
		X1 = stoi(tokens[0]);
		Y1 = stoi(tokens[1]);
		R = stof(tokens[2]);
	}

	myReadFile.close();
}

Mat regionFilter(Mat source) {
	Mat res;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	uchar *ptr;

	findContours(source, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	res = Mat::zeros(source.size(), source.type());
	for (int i = 0; i < contours.size(); i++) {
		if (contours[i].size() > R) {					// if contour will be deleted if the size of contour more than the radius from circle 			
			drawContours(res, contours, i, Scalar(255, 0, 0), 1, 8, hierarchy, 0, Point());
		}
	}

	return res;
}

Mat coLinear(Mat source) {
	Mat res;
	static int data[100];
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(source, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	res = Mat::zeros(source.size(), source.type());
	//source.copyTo(res);

	int idk = 0;
	for (int i = 0; i < contours.size(); i++) {
		if (contours[i].size() > R * 2) {
			Point pt = contours[i][contours[i].size() / 4];
			CCX[idk] = pt.x;
			CCY[idk] = pt.y;
			data[idk] = 0;
		}
		else {
			CCX[idk] = 0;
			CCY[idk] = 0;
			data[idk] = 1;
		}

		idk++;
	}

	// intersectionLine Evaluation
	for (int i = 0; i < contours.size(); i++) {
		for (int j = 0; j < contours.size(); j++) {
			if (i == j) continue;
			int out = 0;
			for (int k = 0; k < contours[i].size() / 2; k++) {
				Point pt1 = contours[i][k];
				Point pt2 = contours[i][k + 2];
				out = intersectionLine(X1, Y1, CCX[j], CCY[j], pt1.x, pt1.y, pt2.x, pt2.y);
				if (out == 1) {
					if ((abs(CCX[j] - pt1.x) < 2) && (abs(CCY[j] - pt1.y) < 2)) {
						data[j] = 0;
					}
					else {
						data[j] = 1;
					}
				}
			}
		}
	}

	for (int i = 0; i < contours.size(); i++) {
		if (data[i] == 0) {
			drawContours(res, contours, i, Scalar(255, 255, 255), 1, 8, hierarchy, 0, Point());
		}
	}

	return res;
}

int intersectionLine(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4) {
	float m1, m2, c1, c2, xp, yp;
	straightLine(x1, y1, x2, y2, &m1, &c1);
	straightLine(x3, y3, x4, y4, &m2, &c2);

	if (m1 == m2)return 0;
	xp = (c2 - c1) / (m1 - m2);
	yp = m1 * xp + c1;

	if ((x1 == x2) && (((xp - x3)*(xp - x4)) < 0) && (((yp - y1)*(yp - y2)) < 0)) return 1;
	if ((x3 == x4) && (((xp - x1)*(xp - x2)) < 0) && (((yp - y3)*(yp - y4)) < 0)) return 1;

	if ((((xp - x1)*(xp - x2)) < 0) && (((xp - x3)*(xp - x4)) < 0))
		return 1;
	else
		return 0;
}

void straightLine(int x1, int y1, int x2, int y2, float *slope, float *intercept) {
	float m, b;
	int x;

	x = x1 - x2;
	if (x == 0)
		m = 1e6;
	else
		m = (float)(y1 - y2) / x;

	b = y1 - m * x1;
	*slope = m;
	*intercept = b;
}


void GetGoodFeature() {
	Point2f rect_points[4];
	float degree;
	Mat garis = Mat::zeros(res.size(), res.type());
	Mat hasil = Mat::zeros(res.size(), res.type());
	Scalar color = Scalar(rng.uniform(256, 256), rng.uniform(256, 256), rng.uniform(256, 256));

	findContours(res, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<RotatedRect> minRect(contours.size());

	for (size_t i = 0; i < contours.size();i++) {

		minRect[i] = minAreaRect(contours[i]);
	}

	for (size_t i = 0; i < contours.size();i++) {

		drawContours(garis, contours, (int)i, color);
		minRect[i].points(rect_points);
	}
	//JIKA DEGREE POSITIF MAKA SISI KANAN, JIKA NEGATIF MAKA SISI KIRI
	int kondisi1 = rect_points[3].x - rect_points[0].x;
	int kondisi2 = rect_points[0].y - rect_points[3].y;

	if (kondisi1 < kondisi2) {
		cout << "KANAN" << endl;

		//Mendapatkan nilai untuk normalisasi
		valnorm = sqrt(pow(rect_points[2].x - rect_points[3].x, 2) + pow(rect_points[2].y - rect_points[3].y, 2));

		//GARIS KANAN
		garis = Scalar::all(0);
		line(garis, rect_points[3], rect_points[0], color);
		for (int y = 0; y < garis.rows; y++) {
			for (int x = 0; x < garis.cols; x++) {
				if (garis.at<uchar>(y, x) > 0) {
					temp1++;
					coordinate1[temp1][0] = Point2f((float)x, (float)y);
				}
			}
		}

		//Melakukan Pembagian range dengan melakukan pembagian sebanyak jumlah garis, hasil pembagian dijadikan range
		float batasan = temp1;
		//Nilai Akan Menghasilkan sebanyak jumlah + 1, namun bagian 1 akan diabaikan, yang merupakan bagian bawah
		float data = (float)temp1 / (jumlah + 1);
		temp1 = 0;
		for (float i = data / 2; i <= batasan; i += data) {
			temp1++;
			temp2 = (int)round(i);
			coordinate2[temp1][0] = coordinate1[temp2][0];
			//Mengabaikan Garis Ke 13
			if (temp1 == jumlah) {
				break;
			}
		}

		//GARIS KIRI
		garis = Scalar::all(0);
		temp1 = 0;
		temp2 = 0;
		line(garis, rect_points[1], rect_points[2], color);
		for (int y = 0; y < garis.rows; y++) {
			for (int x = 0; x < garis.cols; x++) {
				if (garis.at<uchar>(y, x) > 0) {
					temp1++;
					coordinate1[temp1][1] = Point2f((float)x, (float)y);
				}
			}
		}

		//Melakukan Pembagian range dengan melakukan pembagian sebanyak jumlah garis, hasil pembagian dijadikan range
		batasan = temp1;
		//Nilai Akan Menghasilkan sebanyak jumlah + 1, namun bagian 1 akan diabaikan, yang merupakan bagian bawah
		data = (float)temp1 / (jumlah + 1);
		temp1 = 0;
		for (float i = data / 2; i <= batasan; i += data) {
			temp1++;
			temp2 = (int)round(i);
			coordinate2[temp1][1] = coordinate1[temp2][1];
			//Mengabaikan Garis Ke 13
			if (temp1 == jumlah) {
				break;
			}
		}

		temp1 = 0;
		temp2 = jumlah;
		garis = Scalar::all(0);

		for (int i = 1; i <= jumlah; i++) {
			line(garis, coordinate2[i][0], coordinate2[i][1], Scalar(255), 1, 1, 0);
			hasil = garis & res;
			temp3 = 0;
			//MENDAPATKAN COORDINATE DARI DINDING JANTUNG KANAN KE KIRI
			for (int x = hasil.cols - 1; x > 0; x--) {
				for (int y = hasil.rows - 1; y > 0; y--) {
					if (hasil.at<uchar>(y, x) > 0) {
						temp1++;
						temp3++;
						coordinate2[temp1][2] = Point2f((float)x, (float)y);
						break;
					}
				}
				if (temp3 > 0) {
					break;
				}
			}
			//MENDAPATKAN COORDINATE DARI DINDING JANTUNG KIRI KE KANAN
			for (int x = 0; x < hasil.cols; x++) {
				for (int y = 0; y < hasil.rows; y++) {
					if (hasil.at<uchar>(y, x) > 0) {
						temp2++;
						coordinate2[temp2][2] = Point2f((float)x, (float)y);
						hasil = Scalar::all(0);
						garis = Scalar::all(0);
						break;
					}
				}
			}
		}
	}

	else {
		cout << "KIRI" << endl;
		
		//Mendapatkan nilai untuk normalisasi
		valnorm = sqrt(pow(rect_points[1].x - rect_points[2].x, 2) + pow(rect_points[1].y - rect_points[2].y, 2));

		//Melakukan Pencarian Coordinate XY pada sebuah
		garis = Scalar::all(0);
		line(garis, rect_points[2], rect_points[3], color);
		for (int x = 0; x < garis.cols; x++) {
			for (int y = 0; y < garis.rows; y++) {
				if (garis.at<uchar>(y, x) > 0) {
					temp1++;
					coordinate1[temp1][0] = Point2f((float)x, (float)y);
				}
			}
		}

		//Melakukan Pembagian range dengan melakukan pembagian sebanyak jumlah garis, hasil pembagian dijadikan range
		float batasan = temp1;
		//Nilai Akan Menghasilkan sebanyak jumlah + 1, namun bagian 1 akan diabaikan, yang merupakan bagian bawah
		float data = (float)temp1 / (jumlah + 1);
		temp1 = 0;
		for (float i = data / 2; i <= batasan; i += data) {
			temp1++;
			temp2 = (int)round(i);
			coordinate2[temp1][0] = coordinate1[temp2][0];
			//Mengabaikan Garis Ke 13
			if (temp1 == jumlah) {
				break;
			}
		}

		//SISI KIRI
		temp1 = 0;
		garis = Scalar::all(0);
		line(garis, rect_points[0], rect_points[1], color);
		for (int x = 0; x < garis.cols; x++) {
			for (int y = 0; y < garis.rows; y++) {
				if (garis.at<uchar>(y, x) > 0) {
					temp1++;
					coordinate1[temp1][1] = Point2f((float)x, (float)y);
				}
			}
		}

		//Melakukan Pembagian range dengan melakukan pembagian sebanyak jumlah garis, hasil pembagian dijadikan range
		batasan = temp1;
		//Nilai Akan Menghasilkan sebanyak jumlah + 1, namun bagian 1 akan diabaikan, yang merupakan bagian bawah
		data = (float)temp1 / (jumlah + 1);
		temp1 = 0;
		temp2 = 0;
		for (float i = data / 2; i <= batasan; i += data) {
			temp1++;
			temp2 = (int)round(i);
			coordinate2[temp1][1] = coordinate1[temp2][1];
			//Mengabaikan Garis Ke 13
			if (temp1 == jumlah) {
				break;
			}
		}

		temp1 = 0;
		temp2 = jumlah;
		garis = Scalar::all(0);

		for (int i = 1; i <= jumlah; i++) {
			line(garis, coordinate2[i][0], coordinate2[i][1], Scalar(255), 1, 1, 0);
			hasil = garis & res;
			temp3 = 0;
			//MENDAPATKAN COORDINATE DARI DINDING JANTUNG KANAN KE KIRI
			for (int x = hasil.cols - 1; x > 0; x--) {
				for (int y = hasil.rows - 1; y > 0; y--) {
					if (hasil.at<uchar>(y, x) > 0) {
						temp1++;
						temp3++;
						coordinate2[temp1][2] = Point2f((float)x, (float)y);
						break;
					}
				}
				if (temp3 > 0) {
					break;
				}
			}
			//MENDAPATKAN COORDINATE DARI DINDING JANTUNG KIRI KE KANAN
			for (int x = 0; x < hasil.cols; x++) {
				for (int y = 0; y < hasil.rows; y++) {
					if (hasil.at<uchar>(y, x) > 0) {
						temp2++;
						coordinate2[temp2][2] = Point2f((float)x, (float)y);
						hasil = Scalar::all(0);
						garis = Scalar::all(0);
						break;
					}
				}
			}
		}
	}
}

void opticalFlowCalc() {
	vector<uchar> status;
	vector<float> errs[9], lengthFlags;
	TermCriteria termCrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size winSize(21, 21);
	int maxLevel = 10, in = 0, out = 0;
	//Memanggil Array Citra dan GoodFeatures
	for (int i = 1; i <= jumlah * 2;i++) {
		goodFeatures[0].push_back(Point2f(coordinate2[i][2].x, coordinate2[i][2].y));
		if (i == (jumlah * 2)) {
			for (int i = 0; i < images.size(); i++) {
				sources.push_back(images[i]);
			}
		}
	}

	for (int i = 0; i < 9; i++) {
		maxLevel = 3;
		sources[i] = medianFilter(sources[i], 9);
		calcOpticalFlowPyrLK(sources[i], sources[i + 1], goodFeatures[i], goodFeatures[i + 1], status, errs[i], winSize, maxLevel, termCrit);
		for (int j = 0;j < goodFeatures[i].size();j++) {
			//Jarak dilakukan normalisasi berdasarkan ukuran lebar dari ventrikel kiri two & four chamber
			float length = (sqrt(pow(goodFeatures[i][j].x - goodFeatures[i + 1][j].x, 2) + pow(goodFeatures[i][j].y - goodFeatures[i + 1][j].y, 2)))/(valnorm)*100;
			lengthDiffirence[i].push_back(length);
		}
	}
}

// BAGIAN FEATURE EXTRACTION MASUK/JARAK ARAH (+-) DAN KELUAR/JARAK (+-) 
void FeatureExtraction() {

	double angle1, angle2, quadrant1, quadrant2;
	double a1, b1, c1, a2, b2, c2;

	for (int j = 0; j < jumlah; j++) {
		for (int i = 0; i < 9;i++) {
			//PENCARIAN SISI KIRI(GOODFEATURE) DERAJAT KEMIRINGAN
			a1 = sqrt(pow(goodFeatures[i][j].x - goodFeatures[i + 1][j + jumlah].x, 2) + pow(goodFeatures[i][j].y - goodFeatures[i + 1][j + jumlah].y, 2));
			b1 = sqrt(pow(goodFeatures[i + 1][j + jumlah].x - goodFeatures[i][j + jumlah].x, 2) + pow(goodFeatures[i + 1][j + jumlah].y - goodFeatures[i][j + jumlah].y, 2));
			c1 = sqrt(pow(goodFeatures[i][j + jumlah].x - goodFeatures[i][j].x, 2) + pow(goodFeatures[i][j + jumlah].y - goodFeatures[i][j].y, 2));
			angle1 = acos((b1*b1 + c1 * c1 - a1 * a1) / (2 * b1*c1)) * 180 / PHI;
			quadrant1 = (b1*b1 + c1 * c1 - a1 * a1) / (2 * b1*c1) * 180 / PHI;
			if (quadrant1 >= -1.27222e-14) {
				//MASUK
				direction[j + jumlah][i] = int(1);
				double slope1 = (goodFeatures[i + 1][j + jumlah].y - goodFeatures[i][j + jumlah].y) / (goodFeatures[i + 1][j + jumlah].x - goodFeatures[i][j + jumlah].x);
				double slope2 = (goodFeatures[i][j].y - goodFeatures[i][j + jumlah].y) / (goodFeatures[i][j].x - goodFeatures[i][j + jumlah].x);
				if (slope1 > slope2) {
					//cout << "MASUK ++ " << angle1 << endl;
					directionI[j + jumlah][i] = int(1);
				}
				else {
					//cout << "MASUK -- " << angle1 << endl;
					directionI[j + jumlah][i] = int(2);
				}
				//PEMBAGIAN EKSTRAKSI DILAKUKAN DISINI YA UNTUK BAGIAN YANG MASUK
			}
			else {
				//KELUAR
				direction[j + jumlah][i] = int(0);
				double slope1 = (goodFeatures[i + 1][j + jumlah].y - goodFeatures[i][j + jumlah].y) / (goodFeatures[i + 1][j + jumlah].x - goodFeatures[i][j + jumlah].x);
				double slope2 = (goodFeatures[i][j].y - goodFeatures[i][j + jumlah].y) / (goodFeatures[i][j].x - goodFeatures[i][j + jumlah].x);
				if (slope1 < slope2) {
					//cout << "KELUAR -- " << angle1 << endl;
					directionI[j + jumlah][i] = int(3);
				}
				else {
					//cout << "KELUAR ++ " << angle1 << endl;
					directionI[j + jumlah][i] = int(4);
				}
				//PEMBAGIAN EKSTRAKSI DILAKUKAN DISINI YA UNTUK BAGIAN YANG MASUK
			}
			//PENCARIAN SISI KANAN (GOODFEATURE) DERAJAT KEMIRINGAN
			a2 = sqrt(pow(goodFeatures[i + 1][j].x - goodFeatures[i][j].x, 2) + pow(goodFeatures[i + 1][j].y - goodFeatures[i][j].y, 2));
			b2 = sqrt(pow(goodFeatures[i][j + jumlah].x - goodFeatures[i + 1][j].x, 2) + pow(goodFeatures[i][j + jumlah].y - goodFeatures[i + 1][j].y, 2));
			c2 = sqrt(pow(goodFeatures[i][j].x - goodFeatures[i][j + jumlah].x, 2) + pow(goodFeatures[i][j].y - goodFeatures[i][j + jumlah].y, 2));
			angle2 = acos((c2*c2 + a2 * a2 - b2 * b2) / (2 * a2*c2)) * 180 / PHI;
			quadrant2 = (c2*c2 + a2 * a2 - b2 * b2) / (2 * a2*c2) * 180 / PHI;
			if (quadrant2 >= -1.27222e-14) {
				//MASUK
				direction[j][i] = int(1);
				double slope1 = (goodFeatures[i + 1][j].y - goodFeatures[i][j].y) / (goodFeatures[i + 1][j].x - goodFeatures[i][j].x);
				double slope2 = (goodFeatures[i][j + jumlah].y - goodFeatures[i][j].y) / (goodFeatures[i][j + jumlah].x - goodFeatures[i][j].x);
				if (slope1 < slope2) {
					//cout << "MASUK -- " << angle2 << endl;
					directionI[j][i] = int(1);
				}
				else {
					//cout << "MASUK ++ " << angle2 << endl;
					directionI[j][i] = int(2);
				}
			}

			else {
				//KELUAR
				direction[j][i] = int(0);
				double slope1 = (goodFeatures[i + 1][j].y - goodFeatures[i][j].y) / (goodFeatures[i + 1][j].x - goodFeatures[i][j].x);
				double slope2 = (goodFeatures[i][j + jumlah].y - goodFeatures[i][j].y) / (goodFeatures[i][j + jumlah].x - goodFeatures[i][j].x);
				if (slope1 > slope2) {
					//cout << "KELUAR ++ " << angle2 << endl;
					directionI[j][i] = int(3);
				}
				else {
					//cout << "KELUAR -- " << angle2 << endl;
					directionI[j][i] = int(4);
				}
			}
		}
	}
}

void ExtractionMethodI() {
	//##################### METHOD I #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK KELUAR MASUK (24 FITUR YANG DIKALKULASI)
	float pf[24], nf[24], pm[24], nm[24];
	for (int j = 0; j < jumlah * 2;j++) {
		float num1 = 0, num2 = 0, num3 = 0, num4 = 0;
		for (int i = 0; i < 9;i++) {
			if (direction[j][i] == 1) {
				num1++;
				num3 += lengthDiffirence[i][j];
			}
			else {
				num2++;
				num4 += lengthDiffirence[i][j];
			}
		}
		pf[j] = num1 / 9, nf[j] = num2 / 9;
		pm[j] = num3, nm[j] = num4;
	}

	//##################### METHOD II #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK KELUAR MASUK (24 FITUR DILAKUKAN MENJADI 6 SEGMEN DAN AKAN DIKALKULASI)
	int temp = -1;
	float pf1[6], nf1[6], pm1[6], nm1[6];
	for (int i = 0; i < jumlah / 2;i++) {
		float num1 = 0, num2 = 0, num3 = 0, num4 = 0;
		for (int j = 0; j < jumlah / 3;j++) {
			temp++;
			//ARAH
			num1 += pf[temp], num2 += nf[temp];
			//JARAK
			num3 += pm[temp], num4 += nm[temp];
		}
		pf1[i] = num1 / 4, nf1[i] = num2 / 4;
		pm1[i] = num3 / 4, nm1[i] = num4 / 4;
	}

	//##################### METHOD III #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK KELUAR MASUK (KALKULASI SETIAP FRAME 1-10)

	float pf2[9], nf2[9], pm2[9], nm2[9];
	for (int i = 0; i < 9;i++) {
		float num1 = 0, num2 = 0, num3 = 0, num4 = 0;
		for (int j = 0; j < jumlah * 2; j++) {
			if (direction[j][i] == 1) {
				num1++;
				num3 += lengthDiffirence[i][j];
			}
			else {
				num2++;
				num4 += lengthDiffirence[i][j];
			}
		}
		pf2[i] = num1 / 24, nf2[i] = num2 / 24;
		pm2[i] = num3, nm2[i] = num4;
	}

	//##################### METHOD IV #####################
	//PENGAMBILAN FITUR 24 ARAH DAN 24 JARAK  (ILUSTRASI 0.6 MASUK DAN 0.4 KELUAR MAKA DATA 1 (MASUK)
	float flow[24], move[24];
	for (int j = 0; j < jumlah * 2;j++) {
		if (pf[j] > nf[j] && pm[j] > nm[j]) {
			flow[j] = int(1);
			move[j] = pm[j] - nm[j];
		}
		else if (pf[j] < nf[j] && pm[j] > nm[j]) {
			flow[j] = int(1);
			move[j] = pm[j] - nm[j];
		}
		else {
			flow[j] = int(0);
			move[j] = nm[j] - pm[j];
		}
	}
}


void ExtractionMethodII() {
	//##################### METHOD I #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK KELUAR MASUK (24 FITUR YANG DIKALKULASI)
	float pfUP[24], pfDOWN[24], nfUP[24], nfDOWN[24], pmUP[24], pmDOWN[24], nmUP[24], nmDOWN[24];
	for (int j = 0; j < jumlah * 2;j++) {
		float num1 = 0, num2 = 0, num3 = 0, num4 = 0, num5 = 0, num6 = 0, num7 = 0, num8 = 0;
		for (int i = 0; i < 9;i++) {
			if (directionI[j][i] == 1) {
				num1++;
				num5 += lengthDiffirence[i][j];
			}
			else if (directionI[j][i] == 2) {
				num2++;
				num6 += lengthDiffirence[i][j];
			}
			else if (directionI[j][i] == 3) {
				num3++;
				num7 += lengthDiffirence[i][j];
			}
			else {
				num4++;
				num8 += lengthDiffirence[i][j];
			}
		}
		pfUP[j] = num1 / 9, pfDOWN[j] = num2 / 9;
		nfUP[j] = num3 / 9, nfDOWN[j] = num4 / 9;
		pmUP[j] = num5, pmDOWN[j] = num6;
		nmUP[j] = num7, nmDOWN[j] = num8;
	}

	//##################### METHOD II #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK KELUAR(+-) MASUK(+-) (24 FITUR DILAKUKAN MENJADI 6 SEGMEN DAN AKAN DIKALKULASI)
	int temp = -1;
	float pfUP1[24], pfDOWN1[24], nfUP1[24], nfDOWN1[24], pmUP1[24], pmDOWN1[24], nmUP1[24], nmDOWN1[24];
	for (int i = 0; i < jumlah / 2;i++) {
		float num1 = 0, num2 = 0, num3 = 0, num4 = 0, num5 = 0, num6 = 0, num7 = 0, num8 = 0;
		for (int j = 0; j < jumlah / 3;j++) {
			temp++;
			//ARAH
			num1 += pfUP[temp], num2 += pfDOWN[temp];
			num3 += nfUP[temp], num4 += nfDOWN[temp];
			//JARAK
			num5 += pmUP[temp], num6 += pmDOWN[temp];
			num7 += nmUP[temp], num8 += nmDOWN[temp];
		}
		//ARAH KELUAR(+-) MASUK(+-)
		pfUP1[i] = num1 / 4, pfDOWN1[i] = num2 / 4;
		nfUP1[i] = num3 / 4, nfDOWN1[i] = num4 / 4;
		//JARAK KELUAR(+-) MASUK(+-)
		pmUP1[i] = num5 / 4, pmDOWN1[i] = num6 / 4;
		nmUP1[i] = num7 / 4, nmDOWN1[i] = num8 / 4;
	}

	//##################### METHOD III #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK  KELUAR(+-) MASUK(+-) (KALKULASI SETIAP FRAME 1-10)
	float pfUP2[24], pfDOWN2[24], nfUP2[24], nfDOWN2[24], pmUP2[24], pmDOWN2[24], nmUP2[24], nmDOWN2[24];
	for (int i = 0; i < 9;i++) {
		float num1 = 0, num2 = 0, num3 = 0, num4 = 0, num5 = 0, num6 = 0, num7 = 0, num8 = 0;
		for (int j = 0; j < jumlah * 2; j++) {
			if (directionI[j][i] == 1) {
				num1++;
				num5 += lengthDiffirence[i][j];
			}
			else if (directionI[j][i] == 2) {
				num2++;
				num6 += lengthDiffirence[i][j];
			}
			else if (directionI[j][i] == 3) {
				num3++;
				num7 += lengthDiffirence[i][j];
			}
			else {
				num4++;
				num8 += lengthDiffirence[i][j];
			}
		}
		//ARAH KELUAR(+-) MASUK(+-)
		pfUP2[i] = num1 / 24, pfDOWN2[i] = num2 / 24;
		nfUP2[i] = num3 / 24, nfDOWN2[i] = num4 / 24;
		//JARAK KELUAR(+-) MASUK(+-)
		pmUP2[i] = num5 / 24, pmDOWN2[i] = num6 / 24;
		nmUP2[i] = num7 / 24, nmDOWN2[i] = num8 / 24;
	}
}


void ExtractionMethodIII() {
	//##################### METHOD I #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK KELUAR MASUK (24 FITUR ARAH DAN MASUK YANG DIKALKULASI (FLOW = FLOW+/FLOW-) dan (MOVE=MOVE+/MOVE-))
	float flow[24], move[24];
	for (int j = 0; j < jumlah * 2;j++) {
		float num1 = 1, num2 = 1, num3 = 1, num4 = 1;
		for (int i = 0; i < 9;i++) {
			if (direction[j][i] == 1) {
				num1++;
				num3 += lengthDiffirence[i][j];
			}
			else {
				num2++;
				num4 += lengthDiffirence[i][j];
			}
		}
		flow[j] = num1 / num2;
		move[j] = num3 / num4;
	}

	//##################### METHOD II #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK KELUAR MASUK (24 FITUR DILAKUKAN MENJADI 6 SEGMEN DAN AKAN DIKALKULASI, SEHINGGA MEMILIKI 6 FITUR ARAH DAN JARAK)
	int temp = -1;
	float flow1[6], move1[6];
	for (int i = 0; i < jumlah / 2;i++) {
		float num1 = 1, num2 = 1;
		for (int j = 0; j < jumlah / 3;j++) {
			temp++;
			//ARAH
			num1 += flow[temp];
			//JARAK
			num2 += move[temp];
		}
		flow1[i] = num1 / 4;
		move1[i] = num2 / 4;
	}

	//##################### METHOD III #####################
	//PENGAMBILAN FITUR ARAH DAN JARAK KELUAR MASUK (KALKULASI SETIAP FRAME 1-10, SEHINGGA MEMILIKI 9 FITUR ARAH DAN JARAK)
	float flow2[9], move2[9];
	for (int i = 0; i < 9;i++) {
		float num1 = 1, num2 = 1, num3 = 1, num4 = 1;
		for (int j = 0; j < jumlah * 2; j++) {
			if (direction[j][i] == 1) {
				num1++;
				num3 += lengthDiffirence[i][j];
			}
			else {
				num2++;
				num4 += lengthDiffirence[i][j];
			}
		}
		flow2[i] = num1 / num2;
		move2[i] = num3 / num4;
	}
}

void ExtractionMethodIV() {
	//##################### ExtractionMethodIV #####################
	//Ekstraksi Fitur Menggunakan Metode Perhitungan Jarak Antar Dinding Sistole Menuju Diastole
	vector<pair<float, float>> vect1[10];vector<pair<float, float>> vect2[10];
	vector<float> lengthdif[20];
	Point2f coordinate[50][50]; Point2f koordinat[50][50]; Point2f koordinat1[400][50]; Point2f koordinat2[50][50]; Point2f rect_points[4];
	Scalar color = Scalar(rng.uniform(256, 256), rng.uniform(256, 256), rng.uniform(256, 256));
	Mat garis = Mat::zeros(res.size(), res.type());
	Mat hasil = Mat::zeros(res.size(), res.type());
	Mat kontur = Mat::zeros(res.size(), res.type());
	int temp1,temp2,temp3;
	float degree,degreeA,degreeB;
	int garisan = 20;

	//Melakukan Pengurutan data tracking bagian sisi kiri dan kanan
	for (int i = 0; i < 10;i++) {
		for (int j = 0; j < jumlah; j++) {
			vect1[i].push_back(make_pair(goodFeatures[i][j].x, goodFeatures[i][j].y));
			vect2[i].push_back(make_pair(goodFeatures[i][j + jumlah].x, goodFeatures[i][j + jumlah].y));
		}
		sort(vect1[i].begin(), vect1[i].end(), sortbysec);
		sort(vect2[i].begin(), vect2[i].end(), sortbysec);
	}
	//Memindahkan data pengurutan di variabel coordinate
	for (int i = 0; i < 10;i++) {
		temp1 = -1;
		for (int j = jumlah - 1; j >= 0; j--) {
			temp1++;
			coordinate[i][temp1] = Point2f(vect1[i][j].first, vect1[i][j].second);
			if (j == jumlah-1) {
				for (int j = 0; j < jumlah; j++) {
					coordinate[i][j+jumlah] = Point2f(vect2[i][j].first, vect2[i][j].second);
				}
			}
		}
	}
	//Proses menggambar kontur 1-10 (Kontur didapatkan dari hasil tracking)
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < (jumlah * 2) - 1; j++) {
			line(kontur, coordinate[i][j], coordinate[i][j + 1], Scalar(255, 255, 255), 1, 8);
		}
		line(kontur, coordinate[i][(jumlah * 2) - 1], coordinate[i][0], Scalar(255), 1);
	
		//Pencarian Minimum Area Ractangle
		findContours(kontur, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<RotatedRect> minRect(contours.size());

		for (size_t i = 0; i < contours.size();i++) {
			minRect[i] = minAreaRect(contours[i]);
		}

		for (size_t i = 0; i < contours.size();i++) {
			drawContours(kontur, contours, (int)i, color);
			minRect[i].points(rect_points);
		}
		
		//Mendapatkan nilai sudut dari Minimum Area Ractengle
		degree = slope(rect_points[3].x, rect_points[3].y, rect_points[0].x, rect_points[0].y);

		int kondisi1 = rect_points[3].x - rect_points[0].x;
		int kondisi2 = rect_points[0].y - rect_points[3].y;
		
		if (kondisi1 < kondisi2) {
			// KONTOUR JANTUNG BAGIAN KEMIRINGAN KANAN
			//cout << "KANAN" << endl;
			//Membuat garis tegak lurus dinding atas dan bawah
			degreeA = 90 -(degree);

			//Mencari titik tengah dari kontur
			Moments m = moments(kontur, true);
			Point cen(m.m10 / m.m00, m.m01 / m.m00);
			double s = sin(degreeA*CV_PI / 180); double c = cos(degreeA*CV_PI / 180);
			Point p2(cen.x + s * 500, cen.y + c * 500); Point p1(cen.x + s * -500, cen.y + c * -500);
			line(garis, p1, p2, Scalar(255), 1, 1, 0);

			//Mendapatkan garis tegak lurus dari kontur dengan melakukan pencarian area atas dan bawa matriks citra
			garis = garis & kontur;
			temp1 = 0;
			for (int x = 0; x < garis.cols; x++) {
				for (int y = 0; y < garis.rows; y++) {
					if (garis.at<uchar>(y, x) > 0) {
						temp1++;
						koordinat1[temp1][0] = Point2f((float)x, (float)y);
						break;
					}
				}
				if (temp1 > 0) {
					break;
				}
			}

			for (int x = garis.cols - 1; x > 0; x--) {
				for (int y = garis.rows - 1; y > 0; y--) {
					if (garis.at<uchar>(y, x) > 0) {
						temp1++;
						koordinat1[temp1][0] = Point2f((float)x, (float)y);
						break;
					}
				}
				if (temp1 == 2) {
					break;
				}
			}

			garis = Scalar::all(0);
			temp1 = 0;
			line(garis, koordinat1[1][0], koordinat1[2][0], Scalar(255), 1, 1, 0);

			//Proses Garis Tegak Lurus Baru
			for (int y = garis.rows - 1; y > 0; y--) {
				for (int x = garis.cols - 1; x > 0; x--) {
					if (garis.at<uchar>(y, x) > 0) {
						temp1++;
						koordinat1[temp1][0] = Point2f((float)x, (float)y);
					}
				}
			}
			//Melakukan Pembagian range dengan melakukan pembagian sebanyak jumlah garis, hasil pembagian dijadikan range
			float batasan = temp1;
			//Membuat pembagian sebanyak 20 garis
			float data = (float)temp1 / (garisan + 1);
			temp1 = 0;
			temp2 = 0;
			for (float i = data; i <= batasan; i += data) {
				temp1++;
				temp2 = (int)round(i);
				koordinat1[temp1][1] = koordinat1[temp2][0];
				//Mengabaikan Garis Ke 20
				if (temp1 == garisan) {
					break;
				}
			}

			garis = Scalar::all(0);
			temp1 = 0;
			temp2 = garisan;
			//Garis Tegak Lurus Dinding Kiri dan Kanan
			degreeB = -(degree);

			for (int j = 1; j <= garisan; j++) {
				s = sin(degreeB*CV_PI / 180); c = cos(degreeB*CV_PI / 180);
				Point p4(koordinat1[j][1].x + s * 500, koordinat1[j][1].y + c * 500); Point p3(koordinat1[j][1].x + s * -500, koordinat1[j][1].y + c * -500);
				line(garis, p4, p3, Scalar(255), 1, 1, 0);
				hasil = garis & kontur; 
				temp3 = 0;
				//MENDAPATKAN COORDINATE DARI DINDING JANTUNG KANAN KE KIRI
				for (int y = hasil.rows - 1; y > 0; y--) {
					for (int x = hasil.cols - 1; x > 0; x--) {
						if (hasil.at<uchar>(y, x) > 0) {
							temp1++;
							temp3++;
							koordinat[temp1][i] = Point2f((float)x, (float)y);
							break;
						}
					}
					if (temp3 > 0) {
						break;
					}
				}
				//MENDAPATKAN COORDINATE DARI DINDING JANTUNG KIRI KE KANAN
				for (int y = 0; y < hasil.rows; y++) {
					for (int x = 0; x < hasil.cols; x++) {
						if (hasil.at<uchar>(y, x) > 0) {
							temp2++;
							koordinat[temp2][i] = Point2f((float)x, (float)y);
							hasil = Scalar::all(0);
							garis = Scalar::all(0);
							break;
						}
					}
				}
			}
			//Menyusun Bagian Koordinat Kontur Arah Kanan (Agar Seperti Bagian Koordinate Kontur Arah Kiri)
			temp1 = 0;
			temp2 = garisan;
			for (int j = garisan; j >= 1; j--) {
				temp1++;
				koordinat2[temp1][i] = koordinat[j][i];
				if (temp1 == 1) {
					for (int k = garisan*2; k >= garisan; k--) {
						temp2++;
						koordinat2[temp2][i] = koordinat[k][i];
					}
				}
			}
		}
		else {
			// KONTOUR JANTUNG BAGIAN KEMIRINGAN KIRI
			//cout << "KIRI" << endl;
			//Membuat garis tegak lurus dinding atas dan bawah
			degreeA = -(degree);

			//Mencari titik tengah dari kontur
			Moments m = moments(kontur, true);
			Point cen(m.m10 / m.m00, m.m01 / m.m00);
			double s = sin(degreeA*CV_PI / 180); double c = cos(degreeA*CV_PI / 180);
			Point p2(cen.x + s * 500, cen.y + c * 500); Point p1(cen.x + s * -500, cen.y + c * -500);
			line(garis, p1, p2, Scalar(255), 1, 1, 0);
			

			//Mendapatkan garis tegak lurus dari kontur dengan melakukan pencarian area atas dan bawa matriks citra
			garis = garis & kontur; 
			temp1 = 0;
			for (int x = 0; x < garis.cols; x++) {
				for (int y = 0; y < garis.rows; y++) {
					if (garis.at<uchar>(y, x) > 0) {
						temp1++;
						koordinat1[temp1][0] = Point2f((float)x, (float)y);
						break;
					}
				}
				if (temp1 > 0) {
					break;
				}
			}

			for (int x = garis.cols - 1; x > 0; x--) {
				for (int y = garis.rows - 1; y > 0; y--) {
					if (garis.at<uchar>(y, x) > 0) {
						temp1++;
						koordinat1[temp1][0] = Point2f((float)x, (float)y);
						break;
					}
				}
				if (temp1 == 2) {
					break;
				}
			}

			garis = Scalar::all(0);
			temp1 = 0;
			line(garis, koordinat1[1][0], koordinat1[2][0], Scalar(255), 1, 1, 0);
			//Proses Garis Tegak Lurus Baru
			for (int x = 0; x < garis.cols; x++) {
				for (int y = 0; y < garis.rows; y++) {
					if (garis.at<uchar>(y, x) > 0) {
						temp1++;
						koordinat1[temp1][0] = Point2f((float)x, (float)y);
					}
				}
			}

			//Melakukan Pembagian range dengan melakukan pembagian sebanyak jumlah garis, hasil pembagian dijadikan range
			float batasan = temp1;
			//Membuat pembagian sebanyak 20 garis
			float data = (float)temp1 / (garisan+1);
			temp1 = 0;
			for (float i = data; i <= batasan; i += data) {
				temp1++;
				temp2 = (int)round(i);
				koordinat1[temp1][1] = koordinat1[temp2][0];
				//Mengabaikan Garis Ke 20
				if (temp1 == garisan) {
					break;
				}
			}

			garis = Scalar::all(0);
			temp1 = 0;
			temp2 = garisan;
			//Garis Tegak Lurus Dinding Kiri dan Kanan
			degreeB = 90 - (degree);

			for (int j = 1; j <= garisan; j++) {
				s = sin(degreeB*CV_PI / 180); c = cos(degreeB*CV_PI / 180);
				Point p4(koordinat1[j][1].x + s * 500, koordinat1[j][1].y + c * 500); Point p3(koordinat1[j][1].x + s * -500, koordinat1[j][1].y + c * -500);
				line(garis, p4, p3, Scalar(255), 1, 1, 0);
				hasil = garis & kontur;
				temp3 = 0;
				
				//MENDAPATKAN COORDINATE DARI DINDING JANTUNG KANAN KE KIRI
				for (int x = hasil.cols - 1; x > 0; x--) {
					for (int y = hasil.rows - 1; y > 0; y--) {
						if (hasil.at<uchar>(y, x) > 0) {
							temp1++;
							temp3++;
							koordinat2[temp1][i] = Point2f((float)x, (float)y);
							break;
						}
					}
					if (temp3 > 0) {
						break;
					}
				}
				//MENDAPATKAN COORDINATE DARI DINDING JANTUNG KIRI KE KANAN
				for (int x = 0; x < hasil.cols; x++) {
					for (int y = 0; y < hasil.rows; y++) {
						if (hasil.at<uchar>(y, x) > 0) {
							temp2++;
							koordinat2[temp2][i] = Point2f((float)x, (float)y);
							hasil = Scalar::all(0);
							garis = Scalar::all(0);
							break;
						}
					}
				}
			}
		}
		kontur = Scalar::all(0);
	}
	//Menghitung Jarak Antar Dinding Sistole Menuju Diastole
	for (int i = 0; i < 10;i++) {
		for (int j = 1; j <= garisan; j++) {
			float length = sqrt (pow(koordinat2[j][i].x - koordinat2[j + garisan][i].x, 2) + pow(koordinat2[j][i].y - koordinat2[j + garisan][i].y, 2));
			lengthdif[i].push_back(length);
		}
	}
}

