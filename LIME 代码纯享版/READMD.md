/* 全国大学生集成电路创新创业大赛 DJF */
此为LIME低照度处理算法整合版，含neon，多核加速
src 为源代码文件
build 文件夹，内置一张测试图片
CMakeLists.txt CMake配置文件。

使用方法：

cd build
cmake ..
make -j8
./all1

输入 模式 以及文件地址以空格分隔
0 xxx 图片处理
1 xxx 实时视频处理
2 xxx 第三版图片处理
3 xxx 第一版图片处理

如 0 3.jpg 表示对build下的3.jpg进行处理
如需其它版本的处理方式，请修改main.cpp中第22行的类型声明。