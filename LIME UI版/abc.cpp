#include "abc.h"
#include <QDebug>
#include <QPushButton>

ABC::ABC(QObject *parent) : QObject(parent)
{



}


//LIME 判断函数，根据不同的选择使用不同的方法
void ABC::img_cl()
{
    cout << c1 << c2 << c3 << n << d <<endl;
    if(c1 == 2){
        LIME_1 limecore = LIME_1(60, 0.15, 1.1, 0.6, 2, true);
        limecore.load(this->img_out);
        this->img_out = limecore.run();

        emit emit_img_out(this->img_out);
    }else if(c2 == 2 && n == 0 && d ==0){
        LIME_2 lime;
        this->img_out = lime.run(this->img_out);

        emit emit_img_out(this->img_out);
    }else if(c2 == 2 && n == 2 && d ==0){
        LIME_2_n lime;
        this->img_out = lime.run(this->img_out);

        emit emit_img_out(this->img_out);
    }else if(c2 == 2 && n == 0 && d ==2){
        LIME_2_d lime;
        this->img_out = lime.run(this->img_out);

        emit emit_img_out(this->img_out);
    }else if(c2 == 2 && n == 2 && d ==2){
        LIME_2_d_n lime;
        this->img_out = lime.run(this->img_out);

        emit emit_img_out(this->img_out);
    }else if(c3 == 2 && n == 0 && d ==0){
        LIME_3 lime;
        this->img_out = lime.run(this->img_out);

        emit emit_img_out(this->img_out);
    }else if(c3 == 2 && n == 2 && d ==0){
        LIME_3_n lime;
        this->img_out = lime.run(this->img_out);

        emit emit_img_out(this->img_out);
    }else if(c3 == 2 && n == 0 && d ==2){
        LIME_3_d lime;
        this->img_out = lime.run(this->img_out);

        emit emit_img_out(this->img_out);
    }else if(c3 == 2 && n == 2 && d ==2){
        LIME_3_d_n lime;
        this->img_out = lime.run(this->img_out);

        emit emit_img_out(this->img_out);
    }else if(img_out.empty()){
        cerr << "no image" << endl;
    }else{
        cerr << "error" << endl;
    }

}

void ABC::re_img_in(Mat img_in)
{
    this->img_out = img_in;
    img_cl();
}

void ABC::re_c1(int q)
{
    this->c1 = q;
}

void ABC::re_c2(int q)
{
    this->c2 = q;
}

void ABC::re_c3(int q)
{
    this->c3 = q;
}

void ABC::re_n(int q)
{
    this->n = q;
}

void ABC::re_d(int q)
{
    this->d = q;
}



