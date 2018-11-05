# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

import numpy as np
import matplotlib.pylab as plt

#3.2.2ステップ関数の実装
#Numpyのハンドリングとかが全然できてない実装
def step_function(x):
    if x >= 0:
        return 1
    elif x < 0: #elseでも何も問題ない
        return 0

test_arg1 = 2.0
print(step_function(test_arg1))

#ステップ関数の実装
#Numpyのハンドリングありver
def step_function(x):
    y = x > 0
    return y.astype(np.int) #引数を希望の型に変換できる

test_arg2 = np.array([-1.0,1.0,2.0])
print(step_function(test_arg2))


#遊び1
def step_function_str(x):
    y = x > 0
    return y.astype(np.str) #引数を希望の型に変換できる

test_arg2 = np.array([-1.0,1.0,2.0])
print(step_function_str(test_arg2))#['False' 'True' 'True']と表示 bool型を文字列として表示している

#遊び2
def step_function_bool(x):
    y = x > 0
    return y.astype(np.bool) #引数を希望の型に変換できる

test_arg2 = np.array([-1.0,1.0,2.0])
print(step_function_bool(test_arg2))#[False  True  True]と表示 yがもともとbool型だからもそのまま表示と一緒のことやっている


#3.2.3ステップ関数のグラフ
def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0,5.0,0.1) #-5.0から5.0までの値を0.1刻みで作成
y = step_function(x)
plt.plot(x,y) #x,yのデータ列を与える
plt.ylim(-0.1,1.1) #y軸の範囲を指定
plt.show() #Step関数の表を表示


#3.2.4 シグモイド関数の実装
def sigmoid(x):
    return 1/ (1+np.exp(-x))#exp(-x)はe(ネイピア数)の-1乗を表現している

x = np.array([-1.0,1.0,2.0])
print(sigmoid(x))

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


#両方表示してみる
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x,y1)
plt.plot(x,y2,linestyle="--")
plt.ylim(-0.1,1.1)
plt.show

#3.2.7ReLU関数
def relu(x):
    return np.maximum(0,x)

reluY = relu(x)
plt.plot(x,reluY)
plt.show()

#3.3.1 多次元配列
A=np.array([1,2,3,4])
#次元数を取得
np.ndim(A)
#配列の形状を取得
A.shape
A.shape[0]

B=np.array([[1,2],[3,4],[5,6]])

print(B) #[[1,2],[3,4],[5,6]]が表示
print(np.ndim(B)) #2が表示される →次元数:2を示す
print(B.shape) #(3,2)と表示される 3列2行の配列を示している

#3.2.2行列の積
A=np.array([[1,2],[3,4]])

B = np.array([[5,6],[7,8]])

print(np.dot(A,B)) #[[19,22],[44,50]]と表示される

A=np.array([[1,2,3],[4,5,6]])
B=np.array([[1,2],[3,4],[5,6]])

print(np.dot(A,B)) #[[22,28],[49,64]]と表示される

C=np.array([[1,2],[3,4]])

#print(np.dot(A,C)) #エラーが発生 Aの0次元目の数とCの1次元目の数が異なるため → 行列では計算できない

A=np.array([[1,2],[3,4]])
C=2


#ニューラルネットワークの行列の積
X = np.array([1,2])
print(X.shape) #(2,)と表示

W = np.array([[1,3,5],[2,4,6]])
print(W)
print(W.shape) #(2,3)と表示

Y =np.dot(X,W)
print(Y)

#3.4.3各層における信号伝達の実装
X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])

print(W1.shape)#(2,3)と表示される
print(X.shape)#(2,)と表示される
print(B1.shape)#(3,と表示される)

A1=np.dot(X,W1)+B1
print(A1)

print(np.dot(X,W1))

Z1=sigmoid(A1)
print(Z1)

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
print(Z1.shape)#(3,)
print(W2.shape)#(3,2)
print(B2.shape)#(2,)
A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)
#恒等関数
#今回は入力をそのまま出力しているだけだが今までの流れに合わせて下記のように記載
def identity_function(X):
    return X
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])
A3=np.dot(Z2,W3)+B3
Y=identity_function(A3)
#3.4.3実装のまとめ
def sigmoid(x):
    return 1/ (1+np.exp(-x))#exp(-x)はe(ネイピア数)の-1乗を表現している
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network
def forward(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)


#3.5.1恒等関数とソフトマックス関数
a=np.array([0.3,2.9,4.0])

exp_a=np.exp(a) #指数関数
print(exp_a)

sum_exp_a=np.sum(exp_a) #指数関数の和
print(sum_exp_a)

y=exp_a/sum_exp_a
print(y)

#ソフトマックス関数の関数化 #3.5.2の注意点を考慮済み
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

#3.5.2ソフトマックス関数の実装上の注意
a = np.array([1010,1000,990])
#print(np.exp(a) / np.sum(np.exp(a))) #ソフトマックス関数の計算 → 正しく計算されない

c=np.max(a) #今回は1010
print(a-c) #[0,-10,-20]

print(np.exp(a-c) / np.sum(np.exp(a-c)))

#3.5.3ソフトマックス関数の特徴
a=np.array([0.3,2.9,4.0])
y=softmax(a)
print(y)
print(np.sum(y)) #合計が1となり、「確率」として捉えることができる

#3.6.1 MNISTデータセット
#MNIST:機械学習分野で最も有名なデータセット

(x_train,t_train), (x_test,t_test) = \
    load_mnist(flatten=True,normalize=False)

#それぞれのデータの形状を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train,t_train), (x_test,t_test) = \
    load_mnist(flatten=True,normalize=False) #flatten=Truerとすると1次元で画像が格納される

img=x_train[0]
label = t_train[0]
print(label)

print(img.shape)

img=img.reshape(28,28) #NumPyの配列の形状の変形を行なっている
print(img.shape)

img_show(img)

#3.6.2ニューラルネットワークの推論処理
def get_data():
    (x_train,t_train), (x_test,t_test) = \
    load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    
    return network

def predict(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)
    
    return y

x,t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y) #最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt +=1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


#3.6.3バッチ処理
#大きな配列を使用することで、計算の速度が上がるから便利
#変更点は以下の部分だけ

x,t = get_data()
network = init_network()

batch_size = 100 #バッチの数
accuracy_cnt = 0
for i in range(0,len(x),batch_size): #range(srart,end,step)だと1項目と2項目の差がstepになる
    x_batch=x[i:i+batch_size]#入力データのi番目からi+batch_n番目までのデータを取り出す
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)#配列のaxis次元目の要素ごとに最大値のインデックスを見つける
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
