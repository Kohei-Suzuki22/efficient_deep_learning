import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import datasets
from tensorflow.keras import backend as K


if __name__ == '__main__':
  np.random.seed(123)
  tf.random.set_seed(123)

  '''
  1. データの準備
  '''
  mnist = datasets.mnist
  (x_train,t_train),(x_test,t_test) = mnist.load_data()

  # print(x_train.shape)  #: (60000,28,28)
  # print(t_train.shape)  #: (60000,)
  # print(x_test.shape)   #: (10000,28,28)
  # print(t_test.shape)   #: (10000,)


  x_train = (x_train.reshape(-1,784) / 255).astype(np.float32)   # .shape: (6000,784)  # 最大値(255)で割ることで、全体の値を小さく保持
  x_test = (x_test.reshape(-1,784) / 255).astype(np.float32)
  # t_train = np.eye(10)[t_train].astype(np.float32)     #np.eye(10).shape: (10,10)
  # t_test = np.eye(10)[t_test].astype(np.float32)

  x_train,x_val, t_train, t_val = train_test_split(x_train,t_train, test_size=0.2)


  '''
  2. モデルの構築
  '''

  def swish(x,beta=1.):
    return x * K.sigmoid(beta * x)

  model = Sequential()
  # model.add(Dense(200,activation= lambda x: swish(x,beta=10.)))
  # model.add(Dropout(0.3))
  # model.add(Dense(200,activation= lambda x: swish(x,beta=10.)))
  # model.add(Dropout(0.3))
  # model.add(Dense(200,activation= lambda x: swish(x,beta=10.)))
  # model.add(Dropout(0.3))
  # model.add(Dense(10,activation='softmax'))
  # model.add(Dense(200))
  # model.add(LeakyReLU(0.01))
  # model.add(Dense(200))
  # model.add(LeakyReLU(0.01))
  # model.add(Dense(200))
  # model.add(LeakyReLU(0.01))
  # model.add(Dense(10, activation='softmax'))

  '''
  # 重みの初期値の最適化

  ## Heの初期化手法: (ReLU関数に対して、適切な重みの初期値を与える手法)
  Dense(hidden_dim,activation='relu', kernel_initializer='he_normal')

  ## Xavierの初期化手法: (sigmoid,tanhなどの ~対称性~ を持つ関数に対して、低季節な重みの初期化を与える手法)
  ## Dense(hidden_dim,activation='sigmoid', kernel_initializer='glorot_normal')
  '''

  # Xavierの初期化手法
  # model.add(Dense(200, activation='sigmoid',kernel_initializer='glorot_normal'))
  # model.add(Dense(200, activation='sigmoid',kernel_initializer='glorot_normal'))
  # model.add(Dense(200, activation='sigmoid',kernel_initializer='glorot_normal'))
  # model.add(Dense(10, activation='softmax'))

  # Heの初期化手法
  # model.add(Dense(200,activation='relu',kernel_initializer='he_normal'))
  # model.add(Dense(200,activation='relu',kernel_initializer='he_normal'))
  # model.add(Dense(200,activation='relu',kernel_initializer='he_normal'))
  # model.add(Dense(10,activation='softmax', kernel_initializer='he_normal'))

  '''
  # バッチの最適化

  model.add(BatchNormalization()) をつける。

  ※ バッチの正規化は活性化の前に処理する必要があるため、model.add(Dense(hidden_dim,activation='sigmoid'))のような形ではなく、次の順にそれぞれ指定しなければならない
  1.Dense
    → model.add(Dense(200), kernel_initializer='he_normal')
  2.BatchNormalization
    → model.add(BatchNormalization())
  3.Activation
    → model.add(Activation('relu'))
  4.Dropout
    → model.dd(Dropout(0.5))

  ※ タスクによっては、バッチ正規化とドロップアウトを同じモデルに取り入れると学習が不安定になることがある。
      → 学習がうまくいかない場合、バッチ正規化のみを用いるのがよい。

  '''
  model.add(Dense(200,kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(200,kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(200,kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10,kernel_initializer='he_normal', activation='softmax'))


  '''
  3. モデルの学習
  '''
  # 学習率の最適化
  '''
  ## モメンタムをoptimizerに設定(学習率の最適化)
  ## optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)      # momentumは、(<1) 通常は0.5や0.9に設定する

  ## adagradを設定
  ## optimizer = optimizers.Adagrad(learning_rate=0.01)

  ## RMSpropの設定 (learning_rateは 0.001に設定することが一般的)
  # optimizer = optimizers.RMSprop(learning_rate=0.001,rho=0.99)      # rho: 時間減衰率を表すパラメータ。通常は 0.9, 0.99 を用いる。
  
  ## Adadeltaの設定
  ## optimizer = optimizers.Adadelta(rho=0.95) # rho: 0.95に設定することが一般的
  
  ## Adamの設定 (学習率は0.001に設定することが一般的)
  ## optimizer = optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
  '''

  # AMSGradの設定
  optimizer = optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999, amsgrad=True)  # Adamの拡張。amsgrad=Trueを加える。



  model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  es = EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='auto')
  hist = model.fit(x_train,t_train, epochs=100, batch_size=100,verbose=2, validation_data=(x_val,t_val),callbacks=[es])


  '''
  4. モデルの評価
  '''
  # 検証データの誤差の可視化
  loss = hist.history['loss']
  val_loss = hist.history['val_loss']     # 誤差の履歴推移
  # val_loss = hist.history['val_accuracy'] # 正解率の履歴推移

  fig = plt.figure()              # 描写領域を準備
  plt.rc('font', family='serif')  # フォントの設定
  plt.plot(range(len(loss)),loss, color='gray', linewidth=1,label='loss')
  plt.plot(range(len(val_loss)), val_loss, color='black', linewidth=1, label='val_loss')    # データを描写する。
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  # plt.savefig('output.jpg')    # 画像をファイルで保存
  plt.show()




  # テストデータの評価
  loss,acc =model.evaluate(x_test,t_test, verbose=0)
  print('test_loss: {:.3f}, test_acc: {:.3f}'.format(loss,acc))


# accuracy結果 per 活性化関数

## Sigmoid:   70.3%
## tanh:      96.4%
## ReLU:      97.9%
## LeakyReLU: 96.9%       # 別途ライブラリインポートする必要がある。model.add()をDenseと別に定義しなければならない。
## swish:     97.1%       # 自分で実装する必要がある。βを指定する場合はラムダ式を使用する。
