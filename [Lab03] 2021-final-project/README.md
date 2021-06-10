# Final Project Report
사이버보안전공 1971063 김윤서

### 목차

1. Inception module 구현 및 설명
2. GoogLeNet 모델 구현 및 설명
3. 직접 구현한 모델(MyNet) 설명
4. 성능 개선 일지 — hyper-parameter Tuning 과정
5. 최종 결과

---

## 1. Inception module 구현

1x1, 3x3 등의 작은 Convolutional layer 여러 개를 한 층에서 구성하는 형태를 취하는 Inception module 을 구현했다.

→ 여러 층의 Inception module을 구성함으로써, 전체적인 연산량을 줄이고 정확도를 높여주었다.

![Untitled](https://user-images.githubusercontent.com/61778930/121559619-b56e5280-ca51-11eb-8ab8-4ca5e3a37560.png)

*구현한 Inception module 그림  ↑*

### 코드

```python
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)
```

---

## 2. Inception module을 이용해 GoogLeNet 구현

이미지 인식에 좋은 성능을 내는 **GoogLeNet**을 참고하였다.
![Untitled 1](https://user-images.githubusercontent.com/61778930/121559646-bbfcca00-ca51-11eb-9f05-d0b9cf698977.png)

- 특징

→ 22개 층으로 구성

→ 위에서 정의한 Inception module을 여러 개 쌓아 모델 구현

- 모델 structure
![Untitled 2](https://user-images.githubusercontent.com/61778930/121559674-c1f2ab00-ca51-11eb-9cfe-1ca81ca9cea2.png)

### GoogLeNet 코드

```python
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)
      

    def forward(self, x):
        
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
```

### GoogLeNet 결과 ⇒ 74%
![Untitled 3](https://user-images.githubusercontent.com/61778930/121559699-c7e88c00-ca51-11eb-8593-cdd33783b09f.png)

---

## 3. MyNet 모델 정의

GoogLeNet 아키텍처를 기반으로 MyNet 모델을 구현하였다.

먼저 GoogLeNet 모델은 336 * 336 px 크기의 이미지 분류에 특화되어 있기 때문에, 

1) 해당 데이터셋에 알맞게 **Inception 순서를 조정**하고, 
2) **Conv layer 크기**를 아래와 같이 변경해주었다.

- MyNet 모델 구조
    1. Conv2d
    2. BatchNorm2d
    3. ReLU
    4. Inception
        - n1x1(48), n3x3(8), n5x5(16), pool(16)
    5. Inception
        - n1x1(96), n3x3(16), n5x5(32), pool(32)
    6. MaxPool2d
    7. Inception
        - n1x1(160), n3x3(256), n5x5(64), pool(64)
    8. Inception
        - n1x1(256), n3x3(256), n5x5(128), pool(128)
    9. Inception
        - n1x1(256), n3x3(256), n5x5(128), pool(128)
    10. MaxPool2d
    11. Inception
        - n1x1(256), n3x3(512), n5x5(128), pool(128)
    12. Inception
        - n1x1(384), n3x3(384), n5x5(128), pool(128)
    13. AvgPool2d
    14. Linear

### MyNet 코드

```python
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(True),        #활성화 함수 ReLU 사용
        )

        self.a3 = Inception(128,  32,  48,  64,  8, 16, 16)
        self.b3 = Inception(128,  64,  96, 128, 16, 32, 32)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(256, 160,  96, 256, 16,  64,  64)
        self.b4 = Inception(544, 256, 128, 256, 64, 128, 128)
        self.c4 = Inception(768, 256, 128, 256, 64, 128, 128)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a5 = Inception(768, 256, 256, 512, 64, 128, 128)
        self.b5 = Inception(1024, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)  # Average Pooling 사용
        self.linear = nn.Linear(1024, 10)
        

    def forward(self, x):
        
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
```

### MyNet 결과 ⇒ **76%**
![Untitled 4](https://user-images.githubusercontent.com/61778930/121559736-d2a32100-ca51-11eb-8b0c-764652075292.png)

---

## 4. 성능 개선 일지
![Untitled 5](https://user-images.githubusercontent.com/61778930/121559764-d9ca2f00-ca51-11eb-9757-9274209df8d2.png)

### **시도 1**

*epochs=**4*** 로 변경 

- 결과 ⇒ **82%**
![Untitled 6](https://user-images.githubusercontent.com/61778930/121559789-dfc01000-ca51-11eb-8ee8-08e60613c3f1.png)
![Untitled 7](https://user-images.githubusercontent.com/61778930/121559796-e2226a00-ca51-11eb-80e4-80bf86936333.png)

### **시도 2**

*lr=**0.0001*** 로 변경 
![Untitled 8](https://user-images.githubusercontent.com/61778930/121559824-e9497800-ca51-11eb-9418-4dddbee0ce37.png)

- 결과: **75%** 로 더 낮아짐
→ learning rate는 0.001로 유지하도록 한다.
![Untitled 9](https://user-images.githubusercontent.com/61778930/121559847-ee0e2c00-ca51-11eb-8659-0899efba96aa.png)


### **시도 3**

*batch size=**16*** 으로 변경
![Untitled 10](https://user-images.githubusercontent.com/61778930/121559867-f4040d00-ca51-11eb-8142-72c5aeaa7cfd.png)


- 결과 ⇒ **83%**

    → 변화 적음
    → *batch size*를 더 바꿔보며 적절한 값을 찾자.
![Untitled 11](https://user-images.githubusercontent.com/61778930/121559901-f9f9ee00-ca51-11eb-93b1-fcf44584bdf0.png)

### 시도 4

*batch size*=**8** 

- 결과 ⇒ **83%**
![Untitled 12](https://user-images.githubusercontent.com/61778930/121559956-067e4680-ca52-11eb-8f1a-8acb253d79be.png)


⇒ *batch size*를 4, 8, 16 으로 설정했을 때 결과 거의 비슷

→ *batch size*를 좀 더 늘려보자.
→ *epoch=*2 →4로 바꿨을 때도 성능이 올라갔으므로 epoch도 좀 더 바꿔보자.

### 시도 5

*batch size*=**32** 

- 결과 ⇒ **81%**
![Untitled 13](https://user-images.githubusercontent.com/61778930/121559977-0da55480-ca52-11eb-9283-21114cfedbc6.png)


⇒ *batch size* 높여도 성능 변화 크지 않으므로, *batch size= 8*로 유지하도록 함

### 시도 6

*epochs = **6** 
batch_size **= 8**
lr = **0.0005***  조합
![Untitled 14](https://user-images.githubusercontent.com/61778930/121559999-13029f00-ca52-11eb-8ae6-058996e230d7.png)


- 결과 ⇒ **84%**
![Untitled 15](https://user-images.githubusercontent.com/61778930/121560034-1bf37080-ca52-11eb-8212-80bbd024dc90.png)


### 시도 7

*epochs = **6**
batch size = **8**
lr = **0.001***  조합

- 결과 ⇒ **87%**
![Untitled 16](https://user-images.githubusercontent.com/61778930/121560073-23b31500-ca52-11eb-8529-f4adebc7385f.png)

![Untitled 18](https://user-images.githubusercontent.com/61778930/121560107-2b72b980-ca52-11eb-9d9b-22740929c325.png)


  **Accuracy** 그래프

---

## 5. 최종 결과

⇒ 정확도 **87%** 도출
![Untitled 19](https://user-images.githubusercontent.com/61778930/121560141-34638b00-ca52-11eb-923e-5e2019dc9521.png)


사용한 hyper-parameter 요약: 

- Loss 함수: CrossEntropy
- Optimizer: Gradient descent with lr = 0.001 (learning rate)
- Epochs= 6
- Batch_size = 8
- Activation 함수: ReLU

### 최종 결과 시각화 (learning curve)
![Untitled 20](https://user-images.githubusercontent.com/61778930/121560171-3a596c00-ca52-11eb-9291-dad7719d25fb.png)
