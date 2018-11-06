#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:17:43 2018

@author: andoshinya
"""

import numpy as np
from dataset.mnist import load_mnist
import sys,os
sys.path.append(os.pardir)
import matplotlib.pylab as plt


#4.1データから学習する
#重みのパラメータをデータから自動で決定できる！
#時に数億にも及ぶパラメータを自動で決めてくれる

#4.2.1 2乗和誤差

#2乗和誤差の関数
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

#正解は2とする
t = [0,0,1,0,0,0,0,0,0,0]

#例1：「2」の確率が最も高い場合
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(mean_squared_error(np.array(y),np.array(t)))
#0.09750000000000003

#例2「7」の確率が最も高い場合
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(mean_squared_error(np.array(y),np.array(t)))
#0.5975

#例1の方が損失関数の結果が小さくなる
# → 例1の方が出力結果が教師データにより適合していることを示している

#4.2.2交差エントロピー誤差
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


t = [0,0,1,0,0,0,0,0,0,0]
#例1：「2」の確率が最も高い場合
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(cross_entropy_error(np.array(y),np.array(t)))

#例2「7」の確率が最も高い場合
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(cross_entropy_error(np.array(y),np.array(t)))


#4.2.3ミニバッチ学習
#数万、数億のデータを使用して学習を毎回することはできない → 小さな塊(ミニバッチ)を使用して学習をする
(x_train, t_train),(x_test,t_test) =\
    load_mnist(normalize = True,one_hot_label=True)

print(x_train.shape) #(60000,784)
print(t_train.shape) #(60000,10)

#上記のデータから10枚だけをランダムに抜き出す
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


#4.3.1 微分
def numerical_diff(f,x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h)) / (2*h)

#4.3.2数値微分の例
#y = 0.01x^2 + 0.1x
def function_1(x):
    return 0.01*x**2+0.1*x

x=np.arange(0.0,20.0,0.1) #0から20までを0.1刻みのx配列
y=function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()


#x=5で微分
print("x=5で微分 " + str(numerical_diff(function_1,5)))

#x=10で微分
print("x=10で微分 " + str(numerical_diff(function_1,10)))

#接戦も表示した表の出力
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

tf = tangent_line(function_1, 10)
y3 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()

#4.3.3 偏微分 複数の変数がある場合の微分
#問い1 x0 = 3,x1 = 4の時、x0に対する偏微分を求める

def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1,3.0))

#問い2 x0=3,x1=4のx１に対する偏微分
def function_tmp2(x1):
    return 3.0**2 + x1 ** 2
print(numerical_diff(function_tmp2,4.0))


#4.4 勾配 gradient 全ての変数の偏微分をベクトルとしてまとめたもの
def numerical_gradient(f,x):
    h =1e-4 #0.0001
    grad = np.zeros_like(x) #xと同じ形状の配列を生成
    
    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        #f(x-h)の計算
        x[idx] = tmp_val -h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #値を元に戻す
        
    return grad

print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([0.0,2.0])))
print(numerical_gradient(function_2,np.array([3.0,0.0])))


#4.4.1 勾配法
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
        
    return x
        
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))

#勾配法の更新プロセスを表示(サンプルコードより拝借)
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()


def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
        
    return x
#学習率が大きすぎる例 lr = 10.0
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=10.0,step_num=100)) #大きい値に発散

#学習率が小さすぎる例
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=1e-10,step_num=100)) #ほぼ変化なし













