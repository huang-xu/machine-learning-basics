# 例子
## 例1 房价预测
一份简单的房价数据作为示例，来自[selling price of houses](http://people.sc.fsu.edu/~jburkardt/datasets/regression/x27.txt)。

数据包含以下13列，其中第一列为行号，最后一列为房价，其余数据为房子的特性描述。
```
A0, One
A1, the local selling prices, in hundreds of dollars;
A2, the number of bathrooms;
A3, the area of the site in thousands of square feet;
A4, the size of the living space in thousands of square feet;
A5, the number of garages;
A6, the number of rooms;
A7, the number of bedrooms;
A8, the age in years;
A9, construction type
A10, architecture type
A11, number of fire places.
B, selling price 
```

```
 1  1   4.9176  1.0   3.4720  0.998   1.0   7  4  42  3  1  0  25.9
 2  1   5.0208  1.0   3.5310  1.500   2.0   7  4  62  1  1  0  29.5
 3  1   4.5429  1.0   2.2750  1.175   1.0   6  3  40  2  1  0  27.9
 4  1   4.5573  1.0   4.0500  1.232   1.0   6  3  54  4  1  0  25.9
 5  1   5.0597  1.0   4.4550  1.121   1.0   6  3  42  3  1  0  29.9
 6  1   3.8910  1.0   4.4550  0.988   1.0   6  3  56  2  1  0  29.9
 7  1   5.8980  1.0   5.8500  1.240   1.0   7  3  51  2  1  1  30.9
 8  1   5.6039  1.0   9.5200  1.501   0.0   6  3  32  1  1  0  28.9
 9  1  16.4202  2.5   9.8000  3.420   2.0  10  5  42  2  1  1  84.9
10  1  14.4598  2.5  12.8000  3.000   2.0   9  5  14  4  1  1  82.9
11  1   5.8282  1.0   6.4350  1.225   2.0   6  3  32  1  1  0  35.9
12  1   5.3003  1.0   4.9883  1.552   1.0   6  3  30  1  2  0  31.5
13  1   6.2712  1.0   5.5200  0.975   1.0   5  2  30  1  2  0  31.0
14  1   5.9592  1.0   6.6660  1.121   2.0   6  3  32  2  1  0  30.9
15  1   5.0500  1.0   5.0000  1.020   0.0   5  2  46  4  1  1  30.0
16  1   5.6039  1.0   9.5200  1.501   0.0   6  3  32  1  1  0  28.9
17  1   8.2464  1.5   5.1500  1.664   2.0   8  4  50  4  1  0  36.9
18  1   6.6969  1.5   6.9020  1.488   1.5   7  3  22  1  1  1  41.9
19  1   7.7841  1.5   7.1020  1.376   1.0   6  3  17  2  1  0  40.5
20  1   9.0384  1.0   7.8000  1.500   1.5   7  3  23  3  3  0  43.9
21  1   5.9894  1.0   5.5200  1.256   2.0   6  3  40  4  1  1  37.5
22  1   7.5422  1.5   4.0000  1.690   1.0   6  3  22  1  1  0  37.9
23  1   8.7951  1.5   9.8900  1.820   2.0   8  4  50  1  1  1  44.5
24  1   6.0931  1.5   6.7265  1.652   1.0   6  3  44  4  1  0  37.9
25  1   8.3607  1.5   9.1500  1.777   2.0   8  4  48  1  1  1  38.9
26  1   8.1400  1.0   8.0000  1.504   2.0   7  3   3  1  3  0  36.9
27  1   9.1416  1.5   7.3262  1.831   1.5   8  4  31  4  1  0  45.8
28  1  12.0000  1.5   5.0000  1.200   2.0   6  3  30  3  1  1  41.0
```

一般而言，同一地区的房子售价区间主要由房子大小决定，除此之外的若干属性或多或少的影响价格。
后续几节的模型里，$$A_0, ..., A_{11}$$ 将作为输入变量，即模型中的 $$x$$。$$B$$ 是模型的预测目标。