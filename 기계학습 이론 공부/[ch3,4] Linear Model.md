# Linear Model 핵심 내용

공부하며 문답하기


### 1. linear regression에서, f가 deterministic 하지 않은 이유

> f= y+ ε 으로 나타내기 때문이다.   
> noise인 ε가 더해지기 때문에 우리가 정한 f와 실제 점 y 사이 오차 ε를 줄이는 방식으로 f를 수정해나간다.

### 2. PLA 란 무엇인가?

> 가중치 w와 input 벡터 x 값을 곱하여 모두 더한 값 (= weighted sum)이 threshold 이상이면 Approve, threshold 이하이면 Deny 하는 알고리즘.   
> PLA의 목적은 최적의 g를 만족하는 가중치 w 를 찾는 것이다.

### 3. PLA 와 Pocket 알고리즘의 차이?

> **PLA**는 misclassified point를 찾을 때마다 weight 벡터 w를 수정한다 (w(t+1) <- wt . yn . xn 이렇게)   
> **Pocket 알고리즘**은 가장 좋은 성능을 내는 w를 찾으면, 그 값을 저장하여 유지한다.더 좋은 성능을 내는 w가 나왔을 때만 값을 수정하기 때문에, error 그래프를 보면 성능이 유지되는 것을 확인할 수 있다.

![https://blog.kakaocdn.net/dn/co38vV/btq20FzTblD/k4EzrM60K4WPtNuUvLNbJ1/img.png](https://blog.kakaocdn.net/dn/co38vV/btq20FzTblD/k4EzrM60K4WPtNuUvLNbJ1/img.png)

### 4. P.A.C란?

> **P**robably **A**pproximately **C**orrect.   
> 전체 error rate(=뮤)와 샘플의 error rate(=뮤햇)이 비슷함을 나타내는 용어이다.

### 5. 빅데이터가 중요한 이유

> Hoeffeding Inequality에 따르면, 데이터 개수 N이 커짐에 따라뮤와 뮤햇의 차이는 점점 줄어든다.   
> 즉, N이 커지면 sample error rate가 전체 에러율에 근접해지면서 좀 더 정확한 예측이 가능해지기 때문이다.

### 6. Linear Classification / Linear Regression / Logistic Regression 비교

> **Linear Classification**(선형 분류)는 1, -1처럼 바이너리 결과를 리턴한다. 어떤 기준에 대하여 기준 이상이면 참, 이하이면 거짓을 리턴하는 방식이다.    
> **Linear Regression**(선형 회귀)는 input 값 x에 따라 실수인 결과값 y을 반환한다. 결과값이 연속이며, unbounded한 값이라는 특징이 있다.   
> **Logistic Regression**은 input 값에 따른 결과값 y를 0과 1 사이의 '가능성'으로 표현하여,1에 가까운 값일 수록 가능성이 크다고 해석한다.   
> 이러한 점에서 바이너리 분류보다 soft 하므로 "soft binary classification"이라고도 불리며, 결과값이 연속(real)이면서, 0~1 사이의 값으로 bounded 되어있다는 특징이 있다.

### 7. Cross Entropy error measure란 무엇이며, 왜 사용하는가

> 크로스엔트로피는 input x를 통해 output Y가 나올 "가능성" h에 기초하여 에러를 측정하는 방식이다.   
> observed 데이터는 0,1의 hard한 데이터이고,
> 이를 통해 학습된 모델의 fitted 데이터는 0에서 1 사이의 soft한 값이다.   
> 이처럼 서로 다른 distribution의 에러를 구하기 위해 cross entropy가 사용된다.

### 8. Cross-Entropy error 방식의 이점을 Gradient Descent 와 연관지어 설명

> Ein (w) 가 convex function이므로, local minima가 존재 X   
> --> 시작점과 상관 없이 global minimum point를 찾을 수 있다. (local minimum 을 고려할 필요가 없다)

### 9. 크로스엔트로피에서, -1/N * ln(.) 와 같은 수식이 나온 이유를 설명하라

> input x를 통해 output Y가 나올 가능성(likelihood) h는 maximize될 수록 좋은 값이다.   
> 그러나 앞서 선형 분류나 선형 회귀에서, 에러는 항상 minimize하게 측정되어 왔으므로, 통일성을 위해negative 값을 로그 안으로 넣어주어, 최종 값이 minimize 하도록 수식을 바꿔주었다.

### 10. Fixed Gradient-Descent 방식의 pseudo-code 작성하기

1. Initialize t=0 to w(0)   
2. For t=0,1,2, .... do   
3. gradient 계산하기 --> ∇Ein( w(t) )   
4. direction v 정하기   
![https://blog.kakaocdn.net/dn/cCztbF/btq270jTW48/czMIyhK1c2JPTht9uj5a10/img.png](https://blog.kakaocdn.net/dn/cCztbF/btq270jTW48/czMIyhK1c2JPTht9uj5a10/img.png)

5. update w(t+1) = w(t) + n*v   

![https://blog.kakaocdn.net/dn/bcblqf/btq3c9zSaOn/1GKEQlyfmGuXzFOoxU3Bz1/img.png](https://blog.kakaocdn.net/dn/bcblqf/btq3c9zSaOn/1GKEQlyfmGuXzFOoxU3Bz1/img.png)

6. Iterate   
7. Return final w   

### 11. Variable Gradient-Descent 방식의 pseudo-code 작성하기

1. Initialize t=0 to w(0)   
2. For t=0,1,2, .... do      
3. gradient 계산하기 --> ∇Ein( w(t) )   
4. direction v 정하기    // 여기까지 동일  
 
![https://blog.kakaocdn.net/dn/dNNJQK/btq3bnTolrc/gaq34YSwLjjUXvvGf74NN1/img.png](https://blog.kakaocdn.net/dn/dNNJQK/btq3bnTolrc/gaq34YSwLjjUXvvGf74NN1/img.png)

5. Learning Rate n_t 정하기   
 
![https://blog.kakaocdn.net/dn/b4t2TS/btq3bGSIyTe/WK9N0i2A31bLwTjMp8wyKk/img.png](https://blog.kakaocdn.net/dn/b4t2TS/btq3bGSIyTe/WK9N0i2A31bLwTjMp8wyKk/img.png)

6. update w(t+1) = w(t) + n*v // n_t의 값과 cancel되며 v는 더이상 unit 벡터가 아님   
 
![https://blog.kakaocdn.net/dn/btAHQw/btq3bwCBcg2/oBkBWh1WnYi6szTcxSuf41/img.png](https://blog.kakaocdn.net/dn/btAHQw/btq3bwCBcg2/oBkBWh1WnYi6szTcxSuf41/img.png)

7. iterate   
8. Return final w

### -> 왜 v에 마이너스(-) 가 붙는가?

점이 최소값의 **왼쪽**에 있을 때는 오른쪽으로 점을 움직여야 하는데,

현재의 기울기가 음수이므로, 양의 방향으로 움직여야 하기 때문이다.

**즉, 움직여야 하는 방향은 현재 기울기의 반대 방향이다.**

(기울기가 음수이면 양의 방향으로 움직여야 최솟값 convex가 나오므로)
