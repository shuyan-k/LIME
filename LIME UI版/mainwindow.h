#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QCheckBox>
#include <QThread>
#include "abc.h"
#include <QFile>
#include <opencv2/core.hpp>
#include <iostream>
#include <QLabel>
#include <QString>
#include <QImage>
#include <QPixmap>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    Mat img_in,img_chuli;
    QString filename;
    QFile file;
    ABC *abc;
    bool in;
    void inisi();
    QPushButton *close_bt, *chuli, *open;
    QCheckBox *c1,*c2,*c3,*n,*d;
    int st[3];

private:
    Ui::MainWindow *ui;

signals:
    void emit_img(Mat);

private slots:
    void chuli_img();
    void get_chuli_img(Mat);
    void close_my();
    void openFile();

};
#endif // MAINWINDOW_H
