#include <time.h>
#include <iostream>
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