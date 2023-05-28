#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QPushButton>
#include <QFileDialog>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    this->resize(800,480);
    ui->setupUi(this);

    chuli = new QPushButton(this);
    chuli->setText("LIME RUN");
    chuli->resize(100,50);
    chuli->move(170,190);

    open = new QPushButton(this);
    open->setText("OPEN");
    open->resize(100,50);
    open->move(10,190);

    close_bt =new QPushButton(this);
    close_bt->setText("EXIT");
    close_bt->resize(100,50);
    close_bt->move(320,190);


    c1 = new QCheckBox("iteration",this);
    c1->setCheckState(Qt::Unchecked);
    c1->resize(100,50);
    c1->move(30,50);

    c2 = new QCheckBox("acceleration",this);
    c2->setCheckState(Qt::Unchecked);
    c2->resize(100,50);

    c2->move(180,50);

    c3 = new QCheckBox("final",this);
    c3->setCheckState(Qt::Unchecked);
    c3->resize(100,50);

    c3->move(330,50);

    n = new QCheckBox("Neon",this);
    n->setCheckState(Qt::Unchecked);
    n->resize(100,50);

    n->move(30,130);

    d = new QCheckBox("MulCore",this);
    d->setCheckState(Qt::Unchecked);
    d->resize(100,50);

    d->move(330,130);




    abc = new ABC();
    inisi();

}



MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::inisi()
{


    connect(chuli,SIGNAL(clicked(bool)),this,SLOT(chuli_img()));

    connect(abc,SIGNAL(emit_img_out(Mat)),this,SLOT(get_chuli_img(Mat)));
    connect(this,SIGNAL(emit_img(Mat)),abc,SLOT(re_img_in(Mat)));

    connect(close_bt,SIGNAL(clicked(bool)),this,SLOT(close_my()));
    connect(close_bt,SIGNAL(clicked(bool)),this,SLOT(close()));
    connect(c1,SIGNAL(stateChanged(int)),abc,SLOT(re_c1(int)));
    connect(c2,SIGNAL(stateChanged(int)),abc,SLOT(re_c2(int)));
    connect(c3,SIGNAL(stateChanged(int)),abc,SLOT(re_c3(int)));
    connect(n,SIGNAL(stateChanged(int)),abc,SLOT(re_n(int)));
    connect(d,SIGNAL(stateChanged(int)),abc,SLOT(re_d(int)));
    connect(open,SIGNAL(clicked()),this,SLOT(openFile()));

}

void MainWindow::chuli_img()
{
    if(img_in.empty()){
        cerr << " no image " << endl;
        emit open->clicked();
    }else{
        emit emit_img(this->img_in);
    }


}


void MainWindow::get_chuli_img(Mat img)
{
    this->img_chuli = img;
    namedWindow("img_out",WINDOW_NORMAL);
    resizeWindow("img_out",cv::Size(400,400));
    imshow("img_out",img);
    imwrite("/home/user/Desktop/output/img_chuli.jpg",img_chuli*255);
    qDebug() << "finsh" <<endl;
}

void MainWindow::close_my()
{
    destroyAllWindows();
    file.close();
    close();
}

void MainWindow::openFile()
{
    filename= QFileDialog::getOpenFileName(this,"Select a Picture","/home/user/Desktop/input/");
    this->img_in = imread(filename.toStdString());
    namedWindow("img_in",WINDOW_NORMAL);
    resizeWindow("img_in",cv::Size(400,400));
    imshow("img_in",img_in);
    file.setFileName(filename);
    file.open(QIODevice::ReadWrite);

    //file.close();

}

