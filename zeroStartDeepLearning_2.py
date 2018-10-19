#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:49:04 2018

@author: andoshinya
"""

import numpy as np

def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=x1*w1+x2*w2
    if tmp <= theta:
        return 0
    elif tmp>theta:
        return 1

print("ANDゲートの結果")
print(AND(0,0)) #0が表示される
print(AND(1,0)) #0が表示される
print(AND(0,1)) #0が表示される
print(AND(1,1)) #1が表示される


print("新しいANDゲートの記載方法")

x=np.array([0,1])#入力
w=np.array([0.5,0.5])#重み
b=-0.7#バイアス

def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(0,0)) #0が表示される
print(AND(1,0)) #0が表示される
print(AND(0,1)) #0が表示される
print(AND(1,1)) #1が表示される

print("NANDゲート")
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])#重みとバイアスがANDと異なる
    b=0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1

print(NAND(1,1)) #0が返却
print(NAND(1,0)) #1が返却
print(NAND(0,1)) #1が返却
print(NAND(0,0)) #1が返却

print("ORゲート")
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5]) #重みとバイアスだけがANDと異なる
    b=-0.3
    tmp=sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1

print(OR(1,1)) #1が返却
print(OR(1,0)) #1が返却
print(OR(0,1)) #1が返却
print(OR(0,0)) #0が返却

print("XORゲート")
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    
    y=AND(s1,s2)
    
    return y

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
