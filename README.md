# sevt-
就是存储库啊
I=imread('C:\Users\hgh\Pictures\PS\QQ截图20190324131524.jpg');
I1=rgb2gray(I);

[m,n]=size(I1);                             %测量图像尺寸参数

GreyHist=zeros(1,256);                       %预创建存放灰度出现概率的向量

for k=0:255

    GreyHist(k+1)=length(find(I1==k))/(m*n);  %计算每级灰度出现的概率，将其存入GreyHist中相应位置

end

figure(3),

subplot(2,2,2);

bar(0:255,GreyHist,'g')                      %绘制直方图   

title('拉伸前灰度直方图')

xlabel('灰度值')

ylabel('出现概率')

subplot(2,2,1),imshow(I1),title('拉伸前黑白图像');

%灰度拉伸
      
I1=double(I1);

ma=double(max(max(I1)));

mi=double(min(min(I1)));

I1=(255/(ma-mi))*I1-(255*mi)/(ma-mi);

I1=uint8(I1);

%figure(4),

subplot(2,2,3);

imshow(I1);

title('灰度拉伸后黑白图像');

for k=0:255

    GreyHist(k+1)=length(find(I1==k))/(m*n);                 

end

subplot(2,2,4);

bar(0:255,GreyHist,'b')                                      

title('拉伸后的灰度直方图')

xlabel('灰度值')

ylabel('出现概率')

 

%突出目标对象

SE=strel('disk',16);%半径为r=16的圆的模板

I2=imopen(I1,SE);%开运算  用模板SE对灰度图I1进行腐蚀，再对腐蚀后的结果进行膨胀，使外边缘圆滑

figure(4),imshow(I2);title('背景图像');%输出背景图像

%用原始图像与背景图像作减法，增强图像

I3=imsubtract(I1,I2);%两幅图相减

figure(5),imshow(I3);title('增强黑白图像');%输出黑白图像

 

%Step3 取得最佳阈值，将图像二值化，此处为三分点法，取灰度范围的三分之二点

fmax1=double(max(max(I3)));%I3的最大值并输出双精度型

fmin1=double(min(min(I3)));%I3的最小值并输出双精度型

T=(fmax1-(fmax1-fmin1)/3)/255;%获得最佳阈值

bw22=im2bw(I3,T);%转换图像为二进制图像

bw2=double(bw22);

figure(6),imshow(bw2);title('图像二值化');%得到二值图像

%%%%%%%%%%%%%%%  车牌定位模块  %%%%%%%%%%%%%%%%%%%%

%%    数学形态学处理进行车牌粗定位

%%    采用彩色像素点统计，行列扫描的方式实现车牌精确定位

 

%Step4 车牌粗定位，对得到二值图像进行边缘检测和开闭操作进行数字形态学处理

sideline=edge(bw2,'canny')%用canny边缘检测算子识别二值车辆图像中的边界

figure(7),imshow(sideline);title('Canny算子图像边缘提取');%提取并显示出边缘

bg1=imclose(sideline,strel('rectangle',[5,19]));%取矩形核模的闭运算

figure(8),imshow(bg1);title('图像闭运算[5,19]');%输出闭运算的图像

bg3=imopen(bg1,strel('rectangle',[5,19]));%取[5,19]矩形核模的开运算

figure(9),imshow(bg3);title('图像开运算[5,19]');%显示开运算后的图像

bg2=imopen(bg3,strel('rectangle',[11,5]));%取[11,5]矩形核模的开运算

%bg2=bwareaopen(bg2,);%消除细小对象

figure(10),imshow(bg2);title('图像开运算[11,5]');%显示开运算后的图像

bg2=bwareaopen(bg2,5);%消除细小对象，消除间隔符

%figure(11),imshow(bg2);title('消除小对象');

 

%Step5 像素中线扫描（颜色纹理范围定义，行列扫描的方式）精定位和经验阈值分割车牌

 

%%%%%%%%%%%%%%%%  Y方向 %%%%%%%%%%%%%%%%

%进一步确定y方向（水平方向）的车牌区域

[y,x,z]=size(bg2);  %y方向对应行，x方向对应列，z方向对应深度，z=1为二值图像

myI=double(bg2);  %数据类型转换，每个方向范围在0~1  0为黑，1为白（车牌区域）

Im1=zeros(y,x);  %创建一个与图像一样大小的空矩阵，用于记录行扫描时蓝色像素点的位置

Im2=zeros(y,x);  %创建一个与图像一样大小的空矩阵，用于记录列扫描时蓝色像素点的位置

Blue_y=zeros(y,1);%创建一个列向量，用于统计行扫描某行的蓝色像素点个数

%开始行扫描，对每一个像素进行分析，统计满足条件的像素所在行对应的个数，确定车牌的上下边界

   for i=1:y      %行扫描

       for j=1:x

            if  (myI(i,j,1)==1)      %在RGB彩色模型中（0，0，1）表示蓝色，转换数据后 1为蓝色，在二值图中蓝色呈现出白色，也就是1，i,j为坐标。

               Blue_y(i,1)=Blue_y(i,1)+1;%统计第i行蓝色像素点的个数

               Im1(i,j)=1; %标记蓝色像素点的位置

           end

       end

   end

   

% Y方向车牌区域确定

[temp,MaxY]=max(Blue_y);

 

%阈值的设置是经验，采用统计分析方法和车牌的固定特征设置阈值，在规定大小的车辆图像上车牌区域的长宽经过统计，收敛于某个值

Th=5;  %阈值参数可改（要提取的蓝颜色参数经验值范围）

 

%向上追溯，直到车牌区域上边界

PY1=MaxY;

while((Blue_y(PY1,1)>=Th)&&(PY1>1))

    PY1=PY1-1;

end

 

%向下追溯，直到车牌区域的下边界

PY2=MaxY;

while((Blue_y(PY2,1)>=Th)&&(PY2<y))

    PY2=PY2+1;

end

 

%对车牌区域进行校正，加框，减少车牌区域信息丢失

PY1=PY1-2;

PY2=PY2+2;

if PY1<1

    PY1=1;

end

if PY2>y

    PY2=y;

end

 

%得到车牌区域

IY=I(PY1:PY2,:,:);

 

%%%%%%%%%  X方向 %%%%%%%%%%%

%进一步确定x方向（竖直方向）的车牌区域，确定车牌的左右边界

Blue_x=zeros(1,x);   %创建一个行向量，同于统计列扫描某行的蓝色像素点个数

%列扫描，确定车牌的左右边界

for j=1:x     

    for i=PY1:PY2

           if  (myI(i,j,1)==1)

              Blue_x(1,j)=Blue_x(1,j)+1;  %统计第j列蓝色像素点的个数

              Im2(i,j)=1; %标记蓝色像素点的位置

           end

    end

end

 

%向右追溯，直到找到车牌区域左边界

PX1=1;

Th1=3; %经验阈值的选取，可改

while((Blue_x(1,PX1)<Th1)&&(PX1<x))

    PX1=PX1+1;

end

%向左追溯，直到找到车牌区域右边界

PX2=x;

while((Blue_x(1,PX2)<Th1)&&(PX2>1))

    PX2=PX2-1;

end

% 对车牌区域进行校正，加框，减少信息丢失

PX1=PX1-2;

PX2=PX2+2;

if PX1<1

    PX1=1;

end

if PX2>x

    PX2=x;

end

 

%得到车牌区域

IX=I(:,PX1:PX2,:);

 

%分割车牌区域

Plate=I(PY1:PY2,PX1:PX2,:);

row=[PY1 PY2];

col=[PX1 PX2];

Im3=Im1+Im2;  %图像代数运算

Im3=logical(Im3);

Im3(1:PY1,:)=0;

Im3(PY2:end,:)=0;

Im3(:,1:PX1)=0;

Im3(:,PX2:end)=0;

%%%%% 显示%%%%%

figure(11);

subplot(2,2,4);imshow(IY);title('行过滤结果','FontWeight','Bold');

subplot(2,2,2);imshow(IX);title('列过滤结果','FontWeight','Bold');

subplot(2,2,1);imshow(I);title('原图像','FontWeight','Bold');

subplot(2,2,3);imshow(Plate);title('车牌区域','FontWeight','Bold');

imwrite(Plate,'Plate彩色图.jpg');

Plate1=rgb2gray(Plate);%rgb2gray转换成灰度图

imwrite(Plate1,'Plate灰度图.jpg');
%%Rando倾斜校正
   
