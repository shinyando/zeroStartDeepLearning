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

