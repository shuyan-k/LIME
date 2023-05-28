#ifndef ABC_H
#define ABC_H

#include <QObject>
#include <QPushButton>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <thread>
#include <time.h>
using namespace cv;
using namespace std;

//LIME 算法总类，用于和主窗口的连接
class ABC : public QObject
{
    Q_OBJECT
public:

    explicit ABC(QObject *parent = nullptr);
    Mat img_out;
    int c1 = 0,c2 = 0,c3 = 0,n = 0,d =0;
    void img_cl();

private:

    //计时
    class Gt{

        time_t start, end;
        double usd;
    public:
        void st(){
            start = clock();
        }
        void et(){
            end = clock();
        }
        void show(std::string str){
            usd = (double) (end - start) / CLOCKS_PER_SEC;
            std::cout << str << " time:\t"<<usd << std::endl;
        }
    };

    /* * 方法3 多核 */
    class LIME_3_d {
    public:
        std::thread th1,th2;
        Mat img,Wv,Wh;
        LIME_3_d(){}
        Mat maxMAT(cv::Mat &input){


            Mat output(input.size(), CV_32FC1);
            uchar temp;
            for (int i = 0; i < input.size().height; i++) {
                for (int j = 0; j < input.size().width; j++) {
                    temp = fmax(input.at<Vec3f>(i, j)[0], input.at<Vec3f>(i, j)[1]);
                    output.at<float>(i, j) = fmax(input.at<Vec3f>(i, j)[2], temp);
                }
            }


            return output;
        }
        Mat Repeat(Mat& m){

            int he = m.size().height;
            int wi = m.size().width;
            Mat R(he, wi, CV_32FC3);
            for (int i = 0; i < he; i++) {
                for (int j = 0; j < wi; j++) {
                    R.at<Vec3f>(i, j)[0] = m.at<float>(i, j);
                    R.at<Vec3f>(i, j)[1] = m.at<float>(i, j);
                    R.at<Vec3f>(i, j)[2] = m.at<float>(i, j);
                }
            }

            return R;
        }
        void get_wx(){

            Mat grad_x;
            Scharr(img, grad_x, -1, 1, 0);
            Mat wh;
            wh = 1 / (cv::abs(grad_x) + 1);

            this->Wh = wh;
        }
        void get_wy(){

            Mat grad_y;
            Scharr(img, grad_y, -1, 0, 1);
            Mat wv;
            wv = 1 / (cv::abs(grad_y) + 1);

            this->Wv = wv;
        }
        Mat get_t(Mat &t){
            int he = t.size().height;
            int wi = t.size().width;



            this->th1.join();
            this->th2.join();

            Mat re = Wv + Wh;
            cv::Mat I = cv::Mat::ones(he, wi, CV_32F);
            re = I + re;
            re = re + img;

            re = re / 255;
            return re;
        }
        Mat run(cv::Mat &img1){



            img1.convertTo(img1, CV_32FC3);
            this->img = maxMAT(img1);

            this->th1= std::thread(&LIME_3_d::get_wx, this);
            this->th2= std::thread(&LIME_3_d::get_wy, this);


            Mat re = get_t(this->img);


            cv::pow(re, 0.6, re);
            Mat I_out = Repeat(re);

            Mat Re = img1 / (I_out);

            return Re/255;
        }
    };

    /* * 方法3单核 */
    class LIME_3 {
    public:
        Mat img,Wv,Wh;
        LIME_3(){}
        Mat maxMAT(cv::Mat& input) {


            Mat output(input.size(), CV_32FC1);
            uchar temp;
            for (int i = 0; i < input.size().height; i++) {
                for (int j = 0; j < input.size().width; j++) {
                    temp = fmax(input.at<Vec3f>(i, j)[0], input.at<Vec3f>(i, j)[1]);
                    output.at<float>(i, j) = fmax(input.at<Vec3f>(i, j)[2], temp);
                }
            }


            return output;
        }
        Mat Repeat(Mat& m) {

            int he = m.size().height;
            int wi = m.size().width;
            Mat R(he, wi, CV_32FC3);
            for (int i = 0; i < he; i++) {
                for (int j = 0; j < wi; j++) {
                    R.at<Vec3f>(i, j)[0] = m.at<float>(i, j);
                    R.at<Vec3f>(i, j)[1] = m.at<float>(i, j);
                    R.at<Vec3f>(i, j)[2] = m.at<float>(i, j);
                }
            }

            return R;
        }
        void get_wx() {

            Mat grad_x;
            Scharr(img, grad_x, -1, 1, 0);
            Mat wh;
            wh = 1 / (cv::abs(grad_x) + 1);

            this->Wh = wh;
        }
        void get_wy() {

            Mat grad_y;
            Scharr(img, grad_y, -1, 0, 1);
            Mat wv;
            wv = 1 / (cv::abs(grad_y) + 1);

            this->Wv = wv;
        }
        Mat get_t(Mat &t) {
            int he = t.size().height;
            int wi = t.size().width;
            this->img = t;

            get_wy();
            get_wx();

            Mat re = Wv + Wh;
            cv::Mat I = cv::Mat::ones(he, wi, CV_32F);
            re = I + re;
            re = re + img;

            re = re / 255;
            return re;
        }
        Mat run(cv::Mat &img1) {


            img1.convertTo(img1, CV_32FC3);
            Mat imgt = maxMAT(img1);

            Mat re = get_t(imgt);


            cv::pow(re, 0.6, re);
            Mat I_out = Repeat(re);

            Mat Re = img1 / (I_out);

            return Re/255;
        }
    };

    /* * 方法3单核+neon */
    class LIME_3_n {
    public:
        LIME_3_n(){}

        Mat max_neon(Mat &A, Mat &B, Mat &C){

            Mat D(A.size().height, A.size().width, CV_8UC1);
            for (int row = 0; row < A.size().height ; row++){
                for (int col = 0; col < A.size().width; col+=16){
                    uint8x16_t vec1 = vld1q_u8(&A.at<uchar>(row,col));
                    uint8x16_t vec2 = vld1q_u8(&B.at<uchar>(row,col));
                    uint8x16_t vec3 = vld1q_u8(&C.at<uchar>(row,col));

                    uint8x16_t res_vec_1 = vmaxq_u8(vec1, vec2);
                    uint8x16_t res_vec   = vmaxq_u8(res_vec_1, vec3);
                    vst1q_u8(&D.at<uchar>(row,col), res_vec);
                }
            }

            return D;
        }

        Mat repeat_neon(Mat &A){

            Mat B(A.size(),CV_32FC1);
            Mat C(A.size(),CV_32FC1);
            Mat R;
            vector<Mat> channels;
            for(int row = 0; row < A.size().height; row++){
                for(int col = 0; col < A.size().width; col+=4){
                    float32x4_t vec1 = vld1q_f32(&A.at<float>(row,col));

                    vst1q_f32(&B.at<float>(row,col),vec1);
                    vst1q_f32(&C.at<float>(row,col),vec1);
                }
            }
            channels.push_back(A);
            channels.push_back(B);
            channels.push_back(C);
            merge(channels,R);

            return R;

        }

        Mat get_wx(Mat &img){

            Mat grad_x;
            Scharr(img, grad_x, -1, 1, 0);
            Mat Wh;
            Wh = 1 / (cv::abs(grad_x) + 1);

            return Wh;

        }

        Mat get_wy(Mat &img){

            Mat grad_y;
            Scharr(img, grad_y, -1, 0, 1);
            Mat Wv;
            Wv = 1 / (cv::abs(grad_y) + 1);

            return Wv;

        }



        Mat run(Mat &img1)
        {

            int he = img1.size().height;
            int wi = img1.size().width;

            img1.convertTo(img1,CV_8UC3);

            vector<Mat> channels;
            split(img1,channels);

            Mat R = channels[0];
            Mat G = channels[1];
            Mat B = channels[2];

            Mat img = max_neon(R,G,B);

            img.convertTo(img,CV_32FC1);
            Mat Wv, Wh;
            Wv = get_wy(img);
            Wh = get_wx(img);


            Mat re = Wv + Wh;

            cv::Mat I = cv::Mat::ones(he, wi, CV_32FC1);
            re = I + re;
            re = re + img;

            re = re / 255;

            re.convertTo(re,CV_32FC1);

            cv::pow(re, 0.6, re);

            Mat I_out = repeat_neon(re);


            img1.convertTo(img1,CV_32FC3);
            Mat Re = img1 / (I_out);
            return Re/255;
        }
    };

    /* * 方法3 多核+neon */
    class LIME_3_d_n {
    public:
        std::thread th1,th2;
        Mat img,Wv,Wh;
        LIME_3_d_n(){}
        Mat max_neon(Mat &A, Mat &B, Mat &C){
            Mat D(A.size().height, A.size().width, CV_8UC1);
            for (int row = 0; row < A.size().height ; row++){
                for (int col = 0; col < A.size().width; col+=16){
                    uint8x16_t vec1 = vld1q_u8(&A.at<uchar>(row,col));
                    uint8x16_t vec2 = vld1q_u8(&B.at<uchar>(row,col));
                    uint8x16_t vec3 = vld1q_u8(&C.at<uchar>(row,col));

                    uint8x16_t res_vec_1 = vmaxq_u8(vec1, vec2);
                    uint8x16_t res_vec   = vmaxq_u8(res_vec_1, vec3);
                    vst1q_u8(&D.at<uchar>(row,col), res_vec);
                }
            }
            return D;
        }
        Mat repeat_neon(Mat &A){
            Mat B(A.size(),CV_32FC1);
            Mat C(A.size(),CV_32FC1);
            Mat R;
            vector<Mat> channels;
            for(int row = 0; row < A.size().height; row++){
                for(int col = 0; col < A.size().width; col+=4){
                    float32x4_t vec1 = vld1q_f32(&A.at<float>(row,col));

                    vst1q_f32(&B.at<float>(row,col),vec1);
                    vst1q_f32(&C.at<float>(row,col),vec1);
                }
            }
            channels.push_back(A);
            channels.push_back(B);
            channels.push_back(C);
            merge(channels,R);
            return R;

        }
        void get_wx(){
             Mat grad_x;
            Scharr(img, grad_x, -1, 1, 0);
            Mat wh;
            wh = 1 / (cv::abs(grad_x) + 1);

            this->Wh = wh;
        }
        void get_wy(){

            Mat grad_y;
            Scharr(img, grad_y, -1, 0, 1);
            Mat wv;
            wv = 1 / (cv::abs(grad_y) + 1);
            this->Wv = wv;
        }
        Mat get_t(Mat &t){
            int he = t.size().height;
            int wi = t.size().width;
            this->img = t;

            th1= std::thread(&LIME_3_d_n::get_wx, this);
            th2= std::thread(&LIME_3_d_n::get_wy, this);

            th1.join();
            th2.join();

            Mat re = Wv + Wh;
            cv::Mat I = cv::Mat::ones(he, wi, CV_32F);
            re = I + re;
            re = re + img;

            re = re / 255;
            return re;
        }
        Mat run(cv::Mat &img1){

            img1.convertTo(img1, CV_8UC3);
            vector<Mat> channels;
            split(img1,channels);

            Mat R = channels[0];
            Mat G = channels[1];
            Mat B = channels[2];

            Mat imgt = max_neon(R,G,B);
            imgt.convertTo(imgt,CV_32FC1);
            Mat re = get_t(imgt);

            cv::pow(re, 0.6, re);
            Mat I_out = repeat_neon(re);

            img1.convertTo(img1, CV_32FC3);
            Mat Re = img1 / (I_out);

            return Re/255;
        }
    };



    /* * 方法2 多核 */
    class LIME_2_d {
    public:
        std::thread th1,th2;
        Mat img,Wv,Wh;
        LIME_2_d(){}
        Mat maxMAT(cv::Mat &input) {


            Mat output(input.size(), CV_32FC1);
            uchar temp;
            for (int i = 0; i < input.size().height; i++) {
                for (int j = 0; j < input.size().width; j++) {
                    temp = fmax(input.at<Vec3f>(i, j)[0], input.at<Vec3f>(i, j)[1]);
                    output.at<float>(i, j) = fmax(input.at<Vec3f>(i, j)[2], temp);
                }
            }


            return output;
        }
        Mat Repeat(Mat& m) {

            int he = m.size().height;
            int wi = m.size().width;
            Mat R(he, wi, CV_32FC3);
            for (int i = 0; i < he; i++) {
                for (int j = 0; j < wi; j++) {
                    R.at<Vec3f>(i, j)[0] = m.at<float>(i, j);
                    R.at<Vec3f>(i, j)[1] = m.at<float>(i, j);
                    R.at<Vec3f>(i, j)[2] = m.at<float>(i, j);
                }
            }

            return R;
        }
        void get_wx() {


            Mat temp1, temp3;
            Scharr(img, temp1, CV_32FC1, 1, 0);

            temp3 = 1 / (abs(temp1) + 1);

            this->Wh = temp3 / (abs(temp1) + 1);


        }
        void get_wy() {

            Mat temp1, temp3;
            Scharr(img, temp1, CV_32FC1, 0, 1);

            temp3 = 1 / (abs(temp1) + 1);

            this->Wv = temp3 / (abs(temp1) + 1);

        }
        Mat get_t(Mat &img1) {


            this->img = maxMAT(img1);

            this->img.convertTo(this->img, CV_32FC1);


            th1= std::thread(&LIME_2_d::get_wx, this);
            th2= std::thread(&LIME_2_d::get_wy, this);

            th1.join();
            th2.join();

            Mat tx, ty;
            Scharr(this->img, tx, -1, 1, 0);
            Scharr(this->img, ty, -1, 0, 1);
            tx = tx.mul(tx);
            ty = ty.mul(ty);

            float alpha = 0.015;


            this->img = this->img.mul(this->img) + alpha * Wh.mul(tx) + alpha * Wv.mul(ty);
            pow(this->img, 0.5, this->img);


            pow(this->img, 0.6, this->img);
            normalize(this->img, this->img, 0, 255, NORM_MINMAX);
            this->img.convertTo(this->img, CV_32FC1);


            return this->img;
        }
        Mat run(cv::Mat &img1) {
            img1.convertTo(img1, CV_32FC3);
            Mat t = get_t(img1);
            Mat r = Repeat(t);
            img.convertTo(img, CV_32FC3);
            r = img1 / (r);
            return r;
        }
    };

    /* * 方法2 单核 */
    class LIME_2 {
    public:
        Mat img,Wv,Wh;
        LIME_2(){}
        Mat maxMAT(cv::Mat &input) {


            Mat output(input.size(), CV_32FC1);
            uchar temp;
            for (int i = 0; i < input.size().height; i++) {
                for (int j = 0; j < input.size().width; j++) {
                    temp = fmax(input.at<Vec3f>(i, j)[0], input.at<Vec3f>(i, j)[1]);
                    output.at<float>(i, j) = fmax(input.at<Vec3f>(i, j)[2], temp);
                }
            }

            return output;
        }
        Mat Repeat(Mat& m) {

            int he = m.size().height;
            int wi = m.size().width;
            Mat R(he, wi, CV_32FC3);
            for (int i = 0; i < he; i++) {
                for (int j = 0; j < wi; j++) {
                    R.at<Vec3f>(i, j)[0] = m.at<float>(i, j);
                    R.at<Vec3f>(i, j)[1] = m.at<float>(i, j);
                    R.at<Vec3f>(i, j)[2] = m.at<float>(i, j);
                }
            }

            return R;
        }
        void get_wx() {


            Mat temp1, temp3;
            Scharr(img, temp1, CV_32FC1, 1, 0);

            temp3 = 1 / (abs(temp1) + 1);

            this->Wh = temp3 / (abs(temp1) + 1);


        }
        void get_wy() {

            Mat temp1, temp3;
            Scharr(img, temp1, CV_32FC1, 0, 1);

            temp3 = 1 / (abs(temp1) + 1);

            this->Wv = temp3 / (abs(temp1) + 1);


        }
        Mat get_t(Mat &img1) {


            this->img = maxMAT(img1);

            this->img.convertTo(this->img, CV_32FC1);


            get_wx();
            get_wy();

            Mat tx, ty;
            Scharr(this->img, tx, -1, 1, 0);
            Scharr(this->img, ty, -1, 0, 1);
            tx = tx.mul(tx);
            ty = ty.mul(ty);

            float alpha = 0.015;


            this->img = this->img.mul(this->img) + alpha * Wh.mul(tx) + alpha * Wv.mul(ty);
            pow(this->img, 0.5, this->img);


            pow(this->img, 0.6, this->img);
            normalize(this->img, this->img, 0, 255, NORM_MINMAX);
            this->img.convertTo(this->img, CV_32FC1);


            return this->img;
        }
        Mat run(cv::Mat &img1) {
            img1.convertTo(img1, CV_32FC3);
            Mat t = get_t(img1);
            Mat r = Repeat(t);
            img.convertTo(img, CV_32FC3);
            r = img1 / (r);
            return r;
        }
    };

    /* * 方法2 单核+neon */
    class LIME_2_n{
    public:
        LIME_2_n(){}
        Mat max_neon(Mat &A, Mat &B, Mat &C){



            Mat D(A.size().height, A.size().width, CV_8UC1);
            for (int row = 0; row < A.size().height ; row++){
                for (int col = 0; col < A.size().width; col+=16){
                    uint8x16_t vec1 = vld1q_u8(&A.at<uchar>(row,col));
                    uint8x16_t vec2 = vld1q_u8(&B.at<uchar>(row,col));
                    uint8x16_t vec3 = vld1q_u8(&C.at<uchar>(row,col));

                    uint8x16_t res_vec_1 = vmaxq_u8(vec1, vec2);
                    uint8x16_t res_vec   = vmaxq_u8(res_vec_1, vec3);
                    vst1q_u8(&D.at<uchar>(row,col), res_vec);
                }
            }

            return D;
        }

        Mat repeat_neon(Mat &A){



            Mat B(A.size(),CV_32FC1);
            Mat C(A.size(),CV_32FC1);
            Mat R;
            vector<Mat> channels;
            for(int row = 0; row < A.size().height; row++){
                for(int col = 0; col < A.size().width; col+=4){
                    float32x4_t vec1 = vld1q_f32(&A.at<float>(row,col));

                    vst1q_f32(&B.at<float>(row,col),vec1);
                    vst1q_f32(&C.at<float>(row,col),vec1);
                }
            }
            channels.push_back(A);
            channels.push_back(B);
            channels.push_back(C);
            merge(channels,R);


            return R;

        }


        Mat get_wx(Mat &img) {

            Mat wx, temp1, temp3;
            Scharr(img, temp1, CV_32FC1, 1, 0);

            temp3 = 1 / (abs(temp1) + 1);

            wx = temp3 / (abs(temp1) + 1);

            return wx;
        }

        Mat get_wy(Mat &img) {

            Mat wy, temp1, temp3;
            Scharr(img, temp1, CV_32FC1, 0, 1);

            temp3 = 1 / (abs(temp1) + 1);

            wy = temp3 / (abs(temp1) + 1);

            return wy;
        }

        Mat get_t(Mat &img){


            vector<Mat> channels;
            split(img,channels);

            Mat R = channels[0];
            Mat G = channels[1];
            Mat B = channels[2];

            Mat t = max_neon(R,G,B);

            t.convertTo(t,CV_32FC1);
            Mat wx = get_wx(t);
            Mat wy = get_wy(t);

            Mat tx, ty;
            Scharr(t, tx, -1, 1, 0);
            Scharr(t, ty, -1, 0, 1);
            tx = tx.mul(tx);
            ty = ty.mul(ty);

            double alpha = 0.015;
            t = t.mul(t) + alpha * wx.mul(tx) + alpha * wy.mul(ty);

            t.convertTo(t, CV_64FC1);
            pow(t, 0.5, t);




            return t;
        }
        Mat run(Mat &img){

            Mat im = img;
            im.convertTo(im, CV_8UC3);

            Mat t = get_t(im);


            pow(t, 0.6, t);

            normalize(t, t, 0, 255, NORM_MINMAX);
            t.convertTo(t, CV_32FC1);


            Mat r = repeat_neon(t);
            img.convertTo(img,CV_32FC3);
            r = img / (r);


            return r;

        }
    };

    /* * 方法2 多核+neon */
    class LIME_2_d_n{
    public:
        std::thread th1,th2;
        Mat img,Wv,Wh;
        LIME_2_d_n(){}

        Mat max_neon(Mat &A, Mat &B, Mat &C){



            Mat D(A.size().height, A.size().width, CV_8UC1);
            for (int row = 0; row < A.size().height ; row++){
                for (int col = 0; col < A.size().width; col+=16){
                    uint8x16_t vec1 = vld1q_u8(&A.at<uchar>(row,col));
                    uint8x16_t vec2 = vld1q_u8(&B.at<uchar>(row,col));
                    uint8x16_t vec3 = vld1q_u8(&C.at<uchar>(row,col));

                    uint8x16_t res_vec_1 = vmaxq_u8(vec1, vec2);
                    uint8x16_t res_vec   = vmaxq_u8(res_vec_1, vec3);
                    vst1q_u8(&D.at<uchar>(row,col), res_vec);
                }
            }

            return D;
        }

        Mat repeat_neon(Mat &A){



            Mat B(A.size(),CV_32FC1);
            Mat C(A.size(),CV_32FC1);
            Mat R;
            vector<Mat> channels;
            for(int row = 0; row < A.size().height; row++){
                for(int col = 0; col < A.size().width; col+=4){
                    float32x4_t vec1 = vld1q_f32(&A.at<float>(row,col));

                    vst1q_f32(&B.at<float>(row,col),vec1);
                    vst1q_f32(&C.at<float>(row,col),vec1);
                }
            }
            channels.push_back(A);
            channels.push_back(B);
            channels.push_back(C);
            merge(channels,R);


            return R;

        }

        void get_wx(){


            Mat temp1, temp3;
            Scharr(this->img, temp1, CV_32FC1, 1, 0);

            temp3 = 1 / (abs(temp1) + 1);

            this->Wh = temp3 / (abs(temp1) + 1);

        }
        void get_wy(){

            Mat temp1, temp3;
            Scharr(this->img, temp1, CV_32FC1, 0, 1);

            temp3 = 1 / (abs(temp1) + 1);

            this->Wv = temp3 / (abs(temp1) + 1);


        }
        Mat get_t(Mat &img1){


            img1.convertTo(img1, CV_8UC3);
            vector<Mat> channels;
            split(img1,channels);

            Mat R = channels[0];
            Mat G = channels[1];
            Mat B = channels[2];

            this->img = max_neon(R,G,B);

            this->img.convertTo(this->img, CV_32FC1);


            this->th1= std::thread(&LIME_2_d_n::get_wx, this);
            this->th2= std::thread(&LIME_2_d_n::get_wy, this);

            this->th1.join();
            this->th2.join();

            Mat tx, ty;
            Scharr(this->img, tx, -1, 1, 0);
            Scharr(this->img, ty, -1, 0, 1);
            tx = tx.mul(tx);
            ty = ty.mul(ty);

            float alpha = 0.015;


            this->img = this->img.mul(this->img) + alpha * Wh.mul(tx) + alpha * Wv.mul(ty);
            pow(this->img, 0.5, this->img);


            pow(this->img, 0.6, this->img);
            normalize(this->img, this->img, 0, 255, NORM_MINMAX);
            this->img.convertTo(this->img, CV_32FC1);



            return this->img;
        }
        Mat run(cv::Mat &img1){

            Mat t = get_t(img1);
            Mat r = repeat_neon(t);

            img1.convertTo(img1, CV_32FC3);
            r = img1 / (r);
            return r;
        }
    };



    class LIME_1{
    public:
        Mat L;//原图
        Mat _T;//初始估计照度图
        Mat Dv, Dh;//托普勒兹矩阵
        Mat DTD;
        Mat W, Wv, Wh;//权重矩阵
        Mat X;//X矩阵

        int row, col;
        double iterations, alpha, rho, gamma, strategy, exact;



    //最大值
    Mat maxMAT(Mat &input) {
        Mat output(input.size(), CV_64F);
        double temp;
        for (int i = 0; i < input.size().height; i++) {
            for (int j = 0; j < input.size().width; j++) {
                temp = fmax((double)input.at<Vec3b>(i, j)[0], (double)input.at<Vec3b>(i, j)[1]);
                output.at<double>(i, j) = fmax((double)input.at<Vec3b>(i, j)[2], temp);
            }
        }
        return output;
    }

    //单通道转三通道
    Mat Repeat(Mat &m) {
        int he = m.size().height;
        int wi = m.size().width;
        Mat R(he, wi, CV_64FC3);
        for (int i = 0; i < he; i++) {
            for (int j = 0; j < wi; j++) {
                R.at<Vec3d>(i, j)[0] = m.at<double>(i, j);
                R.at<Vec3d>(i, j)[1] = m.at<double>(i, j);
                R.at<Vec3d>(i, j)[2] = m.at<double>(i, j);
            }
        }
        return R;
    }


    /*
     * 对单通道图像进行2D FFT变换
     * @param input: 输入图像
     * @return: 傅里叶变换后的频域图像
     */
    Mat fft2D(const Mat& input) {
        Mat complexImg;
        Mat planes[] = { Mat_<float>(input), Mat::zeros(input.size(), CV_32F) };
        cv::merge(planes, 2, complexImg);
        dft(complexImg, complexImg);
        return complexImg;
    }

    /*
     * 对单通道图像进行2D逆变换
     * @param input: 输入图像的频域表示
     * @return: 逆变换后的图像
     */
    Mat ifft2D(const Mat& input) {
        Mat output;
        idft(input, output, DFT_SCALE | DFT_REAL_OUTPUT);
        return output;
    }

    Mat create_Dv(int row)
    {
        Mat Dv = Mat::zeros(row, row, CV_64FC1);

        double* ptr = Dv.ptr<double>();
        for (int i = 0; i < row - 1; i++)
        {
            ptr[i * row + i] = -1;
            ptr[i * row + i + 1] = 1;
        }
        Dv.at<double>(row-1,row-1) = -1;
        return Dv;
    }

    Mat create_Dh(int col)
    {
        Mat Dh = Mat::zeros(col, col, CV_64FC1);

        for (int i = 0; i < col; i++) {
            Dh.at<double>(i, i) = -1;
            if (i < col - 1) {
                Dh.at<double>(i + 1, i) = 1;
            }
        }


        return Dh;
    }

    void rescale_intensity(Mat& src, Mat& dst, double in_low, double in_high, double out_low, double out_high)
    {
        if (src.channels() == 1)  // 灰度图像
        {
            double min = 0, max = 0;
            minMaxLoc(src, &min, &max);
            double alpha = (out_high - out_low) / (in_high - in_low);
            double beta = out_low - alpha * in_low;
            src.convertTo(dst, CV_64F);
            dst = alpha * (dst) + beta;
        }
        else if (src.channels() == 3)  // 彩色图像
        {
            vector<Mat> channels;
            split(src, channels);
            for (int i = 0; i < channels.size(); ++i)
            {
                Mat channel_rescaled;
                double min = 0, max = 0;
                minMaxLoc(channels[i], &min, &max);
                double alpha = (out_high - out_low) / (in_high - in_low);
                double beta = out_low - alpha * in_low;
                channels[i].convertTo(channel_rescaled, CV_64F);
                channel_rescaled = alpha * (channel_rescaled - min) + beta;
                channels[i].convertTo(channels[i], CV_8UC1);
            }
            cv::merge(channels, dst);
        }
    }

        //initiate parameters
        LIME_1(int iterations1, double alpha1, double rho1, double gamma1, double strategy1, bool exact1){
            iterations = iterations1;
            alpha = alpha1;
            rho = rho1;
            gamma = gamma1;
            strategy = strategy1;
            exact = exact1;
            col = 0;
            row = 0;



        }

        void load(Mat img) {
            L = img;
            row = L.size().height;
            col = L.size().width;

            Mat temp = maxMAT(L);

            _T = temp;
            Dv = create_Dv(row);   //生成row阶Toeplitz 矩阵
            Dh = create_Dh(col);  //生成col阶Toeplitz 矩阵
            get_DTD();
            W = Strategy();
        }

        void get_DTD(){
            Mat dx = Mat::zeros(row, col, CV_64FC1);
            Mat dy = Mat::zeros(row, col, CV_64FC1);

            dx.at<double>(1, 0) = 1;
            dx.at<double>(1, 1) = -1;
            dy.at<double>(0, 1) = 1;
            dy.at<double>(1, 1) = -1;

            Mat dxf = fft2D(dx);
            Mat dyf = fft2D(dy);


            Mat DTD1, DTD2;

            cv::mulSpectrums(dxf, dxf, DTD1, 0, true);
            cv::mulSpectrums(dyf, dyf, DTD2, 0, true);

            DTD = DTD1 + DTD2;
        }
        //获得权重矩阵 返回W
        Mat Strategy() {
            if (strategy == 2) {
                Mat c1, c2;
                cv::gemm(Dv, (_T / 255), 1, Mat(), 0, c1);
                cv::gemm((_T / 255), Dh, 1, Mat(), 0, c2);
                Mat Wv = 1.0 / (cv::abs(c1) + 1);
                Mat Wh = 1.0 / (cv::abs(c2) + 1);

                Mat res(row * 2, col, CV_64FC1);
                cv::vconcat(Wv, Wh, res);
                return res;
            }
            else {
                Mat res = Mat::zeros(row * 2, col, CV_64FC1);
                return res;
            }
        }

        Mat T_sub(Mat G1, Mat Z1, double miu1) {
            Mat Xv, Xh;//储存分割后的矩阵
            Mat Re;//返回值

            X = G1 - Z1 / miu1;
            Xv = X(Rect(0, 0, X.cols, row));
            Xh = X(Rect(0, row, X.cols, row));


            Mat c1, c2;
            cv::gemm(Dv, Xv, 1, Mat(), 0, c1);
            cv::gemm(Xh, Dh, 1, Mat(), 0, c2);

            Mat term1 = 2.0 * _T;
            Mat term2 = miu1 * (c1 + c2);
            Mat res = (term1 + term2) / 255;


            cv::Mat numerator_fft = fft2D(res);


            cv::Mat miu1_mat = cv::Mat::ones(DTD.rows, DTD.cols, DTD.type()) * miu1;
            cv::Mat denominator = DTD.mul(miu1_mat) + 2;


            // 计算n/d
            //Mat k = numerator_fft / denominator;
            Mat k(row, col, CV_32FC2);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    double den = denominator.at<std::complex<float>>(i, j).real();
                    k.at<Vec2f>(i, j) = numerator_fft.at<Vec2f>(i, j) / den;
                }
            }


            // IFFT2操作
            cv::dft(k, k, DFT_INVERSE | DFT_SCALE);    // 进行IFFT2操作

            // 提取实部
            Mat T(row, col, CV_32FC1);
            Mat channels[2];
            split(k, channels);
            Mat T_real1(channels[0]);   // 提取实部
            T_real1.convertTo(T, CV_32FC1);

            rescale_intensity(T, T, 0, 1, 0.001, 1);
            Re = T;

            return Re;
        }


        // G subproblem
        Mat G_sub(Mat T_mx1, Mat Z1, double miu1, Mat W1) {
            Mat temp, temp1, temp3, Re, epsilon;
            epsilon = alpha * W1 / miu1;

            cv::gemm(Dv, T_mx1, 1, Mat(), 0, temp);
            cv::gemm(T_mx1, Dh, 1, Mat(), 0, temp1);

            cv::vconcat(temp, temp1, temp3);
            temp = temp3 + Z1 / miu1;

            cv::Mat temp_abs = cv::abs(temp);
            cv::Mat signMat = cv::Mat::ones(temp.size(), temp.type());

            cv::Mat mask_positive = (temp > 0);
            cv::Mat mask_negative = (temp < 0);
            signMat.setTo(1, mask_positive);
            signMat.setTo(-1, mask_negative);
            signMat.setTo(0, (temp_abs < FLT_EPSILON));

            cv::Mat abs_temp = cv::abs(temp);
            cv::Mat diff = abs_temp - epsilon;
            cv::Mat max_mat = cv::Mat::zeros(diff.size(), diff.type());
            for (int i = 0; i < diff.size().height; i++) {
                for (int j = 0; j < diff.size().width; j++) {
                    if (diff.at<double>(i, j) > 0) {
                        max_mat.at<double>(i, j) = diff.at<double>(i, j);
                    }
                }
            }
            Re = signMat.mul(max_mat);

            return Re;
        }

        //Z subproblem
        Mat Z_sub(Mat T_mx1, Mat G1, Mat Z1, double miu1) {
            Mat temp1, temp2, temp3, Re;
            cv::gemm(Dv, T_mx1, 1, Mat(), 0, temp1);
            cv::gemm(T_mx1, Dh, 1, Mat(), 0, temp2);
            cv::vconcat(temp1, temp2, temp3);
            temp3 = temp3 - G1;
            Re = Z1 + miu1 * temp3;
            return Re;
        }

        //miu subproblem  u
        double miu_sub(double miu1) {
            double Re;
            Re = miu1 * rho;
            return Re; //rho rou
        }

        Mat run() {
            Mat T;
            double miu;
            Mat Re(L.size().height, L.size().width, CV_64FC3);
            Mat T_mx, G, Z, Re_mx;
            if (exact) {

                T_mx = Mat::zeros(row, col, CV_64FC1);
                G = Mat::zeros(row * 2, col, CV_64FC1);
                Z = Mat::zeros(row * 2, col, CV_64FC1);
                miu = 1;
                for (int i = 0; i < iterations; i++) {
                    T_mx = T_sub(G, Z, miu);
                    G = G_sub(T_mx, Z, miu, W);
                    Z = Z_sub(T_mx, G, Z, miu);
                    miu = miu_sub(miu);
                }

                cv::pow(T_mx, gamma, T_mx);

                T = Repeat(T_mx);
                L.convertTo(L, CV_64FC3);

                imwrite("T_3.jpg",T*255);
                Re = L / T;

                rescale_intensity(Re, Re, 0.001, 1, 0, 1);
                return Re;
            }
            else {
                return Re;
            }
            return Re;
        }

    };


//与主窗口通信信号和槽
signals:
    void emit_img_out(Mat);

private slots:
    void re_img_in(Mat);
    void re_c1(int);
    void re_c2(int);
    void re_c3(int);
    void re_n(int);
    void re_d(int);

};


#endif // ABC_H
