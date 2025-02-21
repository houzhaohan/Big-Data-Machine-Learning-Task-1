from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN
import numpy as np


# 转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):  # n_in,n_out相当于lag
    n_vars = 1 if type(data) is list else data.shape[1]  # 变量个数
    df = DataFrame(data)
    print("待转换数据")
    print(df.head())
    cols, names = list(), list()
    # 数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # 预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 拼接 put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


##数据预处理 load dataset
dataset = read_csv('C5#21.csv', header=0, index_col=0)
values = dataset.values
# # 标签编码 integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# 保证为float ensure all data is float
values = values.astype('float32')
# 归一化 normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 转成有监督数据 frame as supervised learning
reframed = series_to_supervised(scaled, 16, 32)
# 删除不预测的列 drop columns we don't want to predict
reframed.drop(reframed.columns[[65,66,67]], axis=1, inplace=True)
print(reframed.head())

# 数据准备
# 把数据分为训练数据和测试数据 split into train and test sets
values = reframed.values
v_train_size = len(values)
print('改变后的训练集数量',v_train_size)
train_size = int(len(dataset) * 0.7)  #70%作为训练集
print('训练集数量',train_size)
# # 拿一年的时间长度训练
# n_train_hours = 365 * 24
# # 划分训练数据和测试数据
train = values[:train_size, :]
test = values[train_size:, :]
# 拆分输入输出 split into input and outputs
train_X, train_y = train[:, :64], train[:, -4]  # 多步预测 train[:, :timesteps*features] train[:, -features]
test_X, test_y = test[:, :64], test[:, -4]
# reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features:维度]
train_X = train_X.reshape((train_X.shape[0], 16, 4))
test_X = test_X.reshape((test_X.shape[0], 16, 4))
print('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

##模型定义 design network
model = Sequential()
model.add(LSTM(35, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(35,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(35))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# 模型训练 fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=120, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# 输出 plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# 进行预测 make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 64))
# 预测数据逆缩放 invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -3:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
inv_yhat = np.array(inv_yhat)
print("预测",inv_yhat)
# 真实数据逆缩放 invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -3:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

#calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
MAE = mean_absolute_error(inv_y, inv_yhat)
print('Test MAE: %.3f' % MAE)

# 画出真实数据和预测数据
pyplot.plot(inv_yhat,label='prediction value',color='#FF6666')
pyplot.plot(inv_y,label='detection value',color='#6699FF')
pyplot.title('LSTM model of NH3 value in winter')
pyplot.xlabel('Time(15min)')
pyplot.ylabel('NH3 Concentration(ppm)')
pyplot.legend()
pyplot.show()



