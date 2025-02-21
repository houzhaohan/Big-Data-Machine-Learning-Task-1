# Big-Data-Machine-Learning-Task-1
Analysis and prediction of ammonia gas in winter for geese raised in fermentation bed net 发酵床网养种鹅冬季氨气分析预测

### 一、实验目的
发酵床网养种鹅冬季氨气分析预测
根据联合国粮农组织（FAO）统计，禽肉消费比例逐年增长，从全球范围内看，禽肉将逐步取代猪肉的地位。鹅业作为我国传统养殖业，自21世纪以来，我国肉鹅出栏量一直位居世界首位，约占全球产量90%左右。
原因：氨气主要是由粪便以及潮湿垫料的发酵产生。
危害：(1)使种鹅处于亚健康状态，影响种鹅体重的增重和生产性能。
(2)机体的免疫力降低。
(3)氨气含量高，氧的含量相对较低；表现为精神萎靡，食欲减退。
(4)严重的会发生疾病：猝死症、腹水症、鹅的慢性呼吸道疾病。
(5)氨气会损害工作人员的健康，影响周围居民的环境质量。
因此，对发酵床网养种鹅冬季的氨气进行准确的分析预测具有重要意义。

### 二、实验原理
（1）数据采集
试验变送器布点
舍外：鹅舍外屋檐下安装温湿度变送器和HOBO温湿度记录仪来记录鹅舍外的温湿度情况。
舍内：Fan2和Fan3中心位置安装温湿度变送器，离地高度1.65m；在种鹅生活区中心线等间距安装5组温湿度、NH3和CO2变送器，离网床高度1m；在湿帘处安装两组无线温湿度变送器，离网床高度1m；生活区变送器间距为7.2m，所有每隔一分钟记录一次数据。
![image](https://github.com/user-attachments/assets/ab731a83-e007-4fe3-9179-f79c4814dd89)

环境监测系统
（2）数据预处理
• 小时均值处理：
整理传感器每分钟采集到的数据，将1小时内测量到的60次数据加和平均处理，便于后续的数据处理和模型建立。
小时均值处理公式：
![image](https://github.com/user-attachments/assets/f10f4d57-4642-4192-aef9-4680cea89020)

式中：xh是小时均值处理后数据，xi是每分钟各采样点数据。
• 数据归一化：
为提高算法收敛速度和精度，使模型建立、学习、训练和预测的效果更好，需要对数据进行标准化处理。
本实验采用数据归一化方法中的最大最小值归一化法，即线性函数归一化法。其原理是：通过使用数据集中数据的最大值和最小值进行标准化处理，使得处理后的数据集中在大于0小于1的区间范围内，具体公式为：
![image](https://github.com/user-attachments/assets/3b3d5e8d-aa89-44ad-ab9e-7b2ca4f447ba)

式中：X∗为归一化处理后数据，X是采集的环境参数，Xmax、Xmin是环境参数中最大值与最小值。

### 三、实验步骤
![image](https://github.com/user-attachments/assets/76288f3e-4109-41f1-bc2d-8d2132fa043a)

（1）时间序列与监督学习
在可以使用机器学习之前，时间序列预测问题必须重新构建成监督学习问题，从一个单纯的序列变成一对序列输入和输出。
定义一个名为series_to_supervised( )的新Python函数，它采用单变量或多变量时间序列，并将其作为监督学习数据集。
该函数有四个参数：
• data：序列，列表或二维数组。
• n_in：用于输入数据步数(x)。值可能介于[1,len(data)]，可选参数。
• n_out：作为输出数据步数(y)。值可能介于[1,len(data)]，可选参数。
• dropnan：用于滤除缺失数据。可选参数。默认为True。
代码实现：
• 首先使用MinMaxScaler( )函数对数据进行归一化处理。
• 然后通过series_to_supervised( )函数将数据转换为有监督的数据。
• 最后利用drop( )函数删除不预测的列。
其中删除列数为特征数(feature)-1，
起始列为n_in* feature+1。
（2）数据集划分
• 将数据集70%作为训练集，30%作为测试集。
• 通过reshape( )函数将训练集与测试集转化为3维，三个参数分别为：数据集行数(shape[0])、输入序列步数( n_in )、特征数(feature)
（3）GRU模型构建
GRU模型代码如右图所示，
• 隐藏层数为1，
• 神经元个数为35，
• 输出层维度为1，
• Epoch为1200，
• Batc_size为120，
• 损失函数为mae，
• 优化器可选sgd与adam优化器。
（4）GRU模型调参
根据需求对神经元个数及网络层数进行选择，如右图所示。
此外，可对Batc_size、学习率等参数进行优化。
为防止过拟合，可采用Dropout方法，随机选择神经层中的一些单元并将其临时隐藏。
（5）评价与制图
将RMSE与MAE作为评价指标，最后绘制预测值与真实值曲线图。

### 四、实验结果
![image](https://github.com/user-attachments/assets/7b79be42-2719-4f82-8e78-a7bfcb4172fc)

LSTM-M.py训练集和测试集运行结果
![image](https://github.com/user-attachments/assets/a8232479-1aed-4a9a-90ba-25e125a13451)

LSTM-M.py运行后NH3预测值和测量值比较结果
![image](https://github.com/user-attachments/assets/4c8b0c29-4ac9-4220-94bb-45ebccd6c6b0)

GRU.py训练集和测试集运行结果
![image](https://github.com/user-attachments/assets/e99c5ced-4efe-45d6-9242-855aca1de4c0)
在epochs=500, batch_size=120参数下GRU预测值和测量值对比图
![image](https://github.com/user-attachments/assets/948a6246-dbba-4a33-9a23-9519d524574d)
在epochs=250, batch_size=60参数下GRU预测值和测量值对比图

### 五、实验总结
本实验通过系统的数据采集、预处理、模型搭建和训练，成功地建立了基于GRU的发酵床网养种鹅冬季氨气预测模型。绘制了预测值与真实值的曲线图，从图中可以看出，epochs=250, batch_size=60参数下GRU预测值和真实值更加接近，而epochs=500, batch_size=120参数下GRU预测值和真实值相差更加大。通过训练GRU模型，能够准确地预测发酵床网养种鹅冬季的氨气浓度变化趋势。评价指标RMSE（Root Mean Square Error）和MAE（Mean Absolute Error）均较低，表明模型预测值与真实值之间的误差较小。
实验结果表明，GRU模型在处理时序数据方面表现出色，能够有效地预测鹅舍内的氨气浓度。这不仅有助于改善种鹅的养殖环境，减少氨气对种鹅健康和生产性能的影响，还能为鹅业养殖提供科学依据和技术支持。
本实验还验证了数据预处理和模型调参在深度学习中的重要性。通过小时均值处理和数据归一化处理，提高了数据的质量和算法收敛速度；通过调整模型参数和采用Dropout方法，进一步优化了GRU模型的性能。
