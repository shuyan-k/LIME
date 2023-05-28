#include "LIME.h"

void sp(int flag = 0,std::string str = "moon.bmp"){
    if(flag == 0){ // 图片处理，改变lime类型可以使用不同的方法
        LIME_2_n lime;
        Mat img_in = imread(str);
        Mat re = lime.run(img_in);
        imwrite("re.jpg",re*255);
        waitKey(0);
    }else if(flag == 1){ //实时视频处理 
        VideoCapture cap(0);
        LIME_3 lime;
        Mat img_in,re;
        while(cap.read(img_in)){
            re = lime.run(img_in);
            imshow("re",re);
            if(waitKey(33) == 27) break;
        }
        destroyAllWindows();
    }else if(flag == 2){ //本地视频处理
        VideoCapture cap(str);
        LIME_3 lime;
        Mat img_in,re;
        while(cap.read(img_in)){
            re = lime.run(img_in);
            imshow("re",re);
            if(waitKey(33) == 27) break;
        }
        destroyAllWindows();
    }else if(flag == 3){ //算法一，用于高精度要求
        LIME_1 limecore = LIME_1(1, 0.15, 1.1, 0.8, 2, true);
        Mat img_in = imread(str);
        limecore.load(img_in);
        Mat re = limecore.run();
        imwrite("re.jpg",re);
        waitKey(0);
    }else{
        cout << "flag error!!!" << endl;
    }
}
int main() {
    while(1){
    cout << "Please select modules and enter the filePath when necessery" <<endl;
    std::string str;
    int flag;
    cin >> flag >> str;
    sp(flag,str);
    }
    return 0;
}
