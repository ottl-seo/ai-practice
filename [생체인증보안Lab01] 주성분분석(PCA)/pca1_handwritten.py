# -*- coding: utf-8 -*-
"""PCA1-handwritten.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17oK1axA1RVKlyDyh-MuN3aJKCoAwliVC

# 1. HandWritten 숫자 데이터 분석 과제

### **패키지 import**
"""

import numpy as np

import matplotlib.pyplot as pylab  # matplot (시각화에 쓰임)

from sklearn.cluster import KMeans  # k-means 군집화 모델

from sklearn.metrics import pairwise_distances_argmin

from skimage.io import imread

from sklearn.utils import shuffle

from skimage import img_as_float  # 수기 이미지를 숫자로 바꿔주는 모델

"""### 데이터를 **shape 함수로 분석**"""

from sklearn.datasets import load_digits  # 숫자 데이터셋 불러오기

digits = load_digits() # 숫자 불러와 변수로 정의
print(digits.data.shape) # digits이 가리키는 숫자를 shape 함수로 분석

j=1
np.random.seed(1) # 랜덤 씨드 지정
fig = pylab.figure(figsize=(3,3))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

"""#### 진행과정 눈으로 확인하기 위해 subplot 이용   
--> 다시 8x8 이미지로 변경
"""

for i in np.random.choice(digits.data.shape[0], 25):
  pylab.subplot(5, 5, j), pylab.axis('off')
  pylab.imshow(np.reshape(digits.data[i, : ], (8,8)), cmap='binary') # reshape-> 다시 8x8 데이터(이미지)로 바꿔준다
  j+=1

pylab.show() # 화면 나타내기

"""### **시각화** 진행 (matplotlib)"""

from sklearn.decomposition import PCA

pca_digits = PCA(2)
digits.data_proj = pca_digits.fit_transform(digits.data)

pylab.figure(num='', figsize=(15,10))
pylab.scatter(digits.data_proj[:,0], digits.data_proj[:,1], lw=0.25, \
              c=digits.target, edgecolor='k', s=100, \
              cmap=pylab.cm.get_cmap('cubehelix',10))

pylab.xlabel('PC1', size=20), pylab.ylabel('PC2', size=20) # 그래프 x축, y축에 표시할 내용
pylab.title('2D Projection of handwritten digits with PCA', size=25) # 제목 지정
pylab.colorbar(ticks=range(10), label='digit value')
pylab.clim(-0.5, 9.5)

pylab.show() # show 꼭 해줄 것