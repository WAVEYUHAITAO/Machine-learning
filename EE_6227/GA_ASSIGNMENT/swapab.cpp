#include<iostream>
using namespace std;            
void swap(int &a,int &b) //void swap(int a,int b),如果这样写则不能实现交换两个数的目的的。因为传递的方式为值传递(单向传递)
{   
     int tmp;   
     tmp = a;    
     a = b;    
     b = tmp;    
 }   
 int main(){   
     int a = 1;    
     int b = 2;    
     swap(a, b);    
     cout<< "a = " << a << endl;   
     cout<< "b = " << b << endl;   
     system("pause");   
     return 0;   
 }