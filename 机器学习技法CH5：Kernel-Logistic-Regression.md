---
title: 机器学习技法CH5：Kernel Logistic Regression
date: 2021-02-17 23:46:55
index_img: /img/ml_twu2.png
tags: 机器学习
---

# CH5：Kernel Logistic Regression

## Soft-Margin SVM as Regularzied

![image-20210217182704686](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217182704686.png)

当$(x_n,y_n)$越过边界时，错误就是$1-y_n(w^Tz_n+b)$,当$(x_n,y_n)$没有越过边界时，说明他是正确的没有错误，即$\xi=0$,那么我们综上所述：可以把$\xi$换成另一种写法：$max(1-y_n(w^Tz_n+b),0)$

此时我们的SVM可以这么写:

![image-20210217183252584](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217183252584.png)



这样一来，我们把之前抽象的$\xi$概念转化为了具体的式子，同时我们现在可以通过 $b,w$的不同取值最小化下式的soft-margin SVM问题：

![image-20210217183428181](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217183428181.png)







![image-20210217183541395](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217183541395.png)

![image-20210217184542732](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217184542732.png)

因此soft-margin SVM就是一个L2 regularization的问题。



![image-20210217185054053](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217185054053.png)

他们之间有一些相互的关系：

![image-20210217185144879](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217185144879.png)



## SVM versus Logistic  Regression

首先我们对比一下0-1error 和 SVM中的error measure的区别：

![image-20210217190201303](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217190201303.png)

我们不难看出：SVM中的error measure总是大于0-1error measure。

我们在logistic regression中用到的error measure叫做：scaled cross-entropy error ，详细推导见：[logistic regression](https://chillstepp.github.io/2021/01/20/CH10%EF%BC%9ALogistic-Regression/)。

首先我们有个概念叫做：cross entropy

![image-20210217190950840](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217190950840.png)

但是为了图像对比，改进成了**scaled cross-entropy**,即$ln$换成了$log_2$



我们对比一下三者：

![image-20210217191214795](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217191214795.png)

![image-20210217191744126](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217191744126.png)

这样看来： SVM的error measure方法和logistic regression的error measure方法很相似。



SVM此时可以看作: 有着L2-regularized的logistic regression

![image-20210217191912311](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217191912311.png)



![image-20210217192206685](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217192206685.png)



因此现在我们有了一个新的想法：

![image-20210217192312492](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217192312492.png)

当我们求出SVM的解的时候，我们是否可以通过SVM的解来反映logistics regression里的几率问题呢？

## SVM for Soft Binary

 

![image-20210217210724051](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217210724051.png)

- 我们跑soft-margin SVM然后得到$b，w$, 然后传回到g(x)当作分类器。这样的方法表现得一般不错，但是**少了一些逻辑回归的特点，logistics regression中的梯度下降跑出来的结果强调的是maximum(最大)，但是现在SVM跑出来的肯定和最大有一点点的差距。**

- 我们还有一种想法就是： 我们做逻辑回归时需要选一个起始点然后去做SGD/GD(梯度下降)，那么我们可以把SVM求出来的结果当作迭代的起始点，然后再去做SGD/GD. 但是这样的做法会使得我们**SVM中的kernel等特点没法用到。**

我们想着补全两者的缺点。因此我们添加上两个自由度A,B

![image-20210217212417123](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217212417123.png)

- 用A,B去微调hyperplane 使得 其满足maximum，即logistics regression的特点

- 同时我们由于保留了$\phi$,所以kernel的特点也保留了下来。

同时我们也要注意到：

![image-20210217212635731](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217212635731.png)

A一般大于0，如果你的A小于0，说明你的$w_{SVM}$求得有问题。B一般约等于0，他只是平移hyperplane，由于我们SVM求出的已经比较准确了，所以一般不会有什么太大的动作。



那我们新的回归问题可以写作：

![image-20210217212616360](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217212616360.png)



我们观察这个式子：

![image-20210217213032621](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217213032621.png)

这个操作将一个多维的数据$x_n$转换为了一个具体的数:

$\phi_{SVM}(x_n)=w_{SVM}^T\phi(x_n)+b_{SVM}$

我们再用这个数来用A,B做调整即可，此时logistics regression此时只需要做A,B两个维度即可。



![image-20210217215153765](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217215153765.png)



![image-20210217222127484](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217222127484.png)

但是这样得到的不是Z空间的最优解，只是一个经过A,B调整后比较好的解。



## Kernel Logistic Regression

![image-20210217222608350](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217222608350.png)

最优解的$w$中也包含核技巧，并且也是$z_n$的线性组合

其实无论是SVM，PLA还是logReg by SGD都是$z_n$的线性组合：

![image-20210217222735537](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217222735537.png)



所以我们得出结论：

![image-20210217232135387](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217232135387.png)

![image-20210217232208332](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217232208332.png)

​	我们的最佳的$w_*$一定是由两部分组成的，一部分是平行于z-空间向量$z_n$们所张成的空间，另一部分是垂直于这个张成的空间。

​	我们肯定希望垂直于张成的空间的w部分，即$w_⊥=0$.

如果$w_⊥\ne 0$会得到什么结果呢？我们考虑一下下面两个部分：

![image-20210217232702845](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217232702845.png)

我们发现这部分并不会被影响因为$w_⊥z_n=0$是一个事实，垂直于$z_n$的向量和$z_n$相乘当然等于0。

![image-20210217232836583](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217232836583.png)

第二部分，当我们考虑大小时就出现了问题，可以看出向量会大一些。

但是这就出现了问题，这提示我们$w_*^Tw_*>w_∥^Tw_∥$,这说明我们的最优解不是最优解了竟然，这意味着我们的$w_⊥=0$是恒成立的，上面相当于我们在反证这个结论。

$w_⊥=0$是恒成立的就意味着：**我们的$w_*$一定是由$z_n$的线性组合得来的。**



这样一来我们就可以加上核技巧了。

![image-20210217233615892](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217233615892.png)

此时我们求解$\beta$即可。

![image-20210217233704780](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217233704780.png)

怎么求呢？ GD/SGD都是很好的选择。



另一种视角：

![image-20210217234001203](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217234001203.png)

- 相当于权重$\beta$和做feature transform后的数据的乘积。

- ![image-20210217234232883](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210217234232883.png)这部分可以看作一种特殊的正则化。

- 因此Kernel Logistic Regression可以看作$\beta$的线性模型，只不过它是通过kernel transform后的数据和一种特殊的kernel regularizer 得到的结果。

  