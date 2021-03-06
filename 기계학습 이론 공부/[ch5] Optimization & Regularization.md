# 5. Optimization & Regularization

## Optimization

- 


## Regularization

- what is it?

    모델이 too complex해지면 noise까지 과도하게 학습시키는 overfit 발생     
    
    —> overfit 하지 않도록 모델을 조정(**constrain**)해주는 것이 정규화의 목적

    "**`constrain`** *a model so that it can avoid fitting noise"*
![Untitled](https://user-images.githubusercontent.com/61778930/115956972-c5ca7c80-a53a-11eb-8527-3b7101b30eb4.png)

### Regularization method

1. **w 등 변수**를 바꿔준다.
2. **model complexity 더해준다**. (Ein **+ Ω**)

    Ein을 minimize하려다보니 Overfit 발생   
    —> 모델 복잡도 Ω를 더해줌으로써 Ein ~= Eout 상태로 만든다.

3. **Ensemble methods** (앙상블 메서드)

### Regularization의 목표
![Untitled 1](https://user-images.githubusercontent.com/61778930/115956985-d67af280-a53a-11eb-8f11-1b19444c3371.png)

: Overfit 방지 ( case3의 상태를 case2 로 돌리는 작업)

---

### Most famous regularizers; L1 and L2

- **L1 regularizer**: aka '**lasso**'
![Untitled 2](https://user-images.githubusercontent.com/61778930/115956951-ae8b8f00-a53a-11eb-8878-1d49e2441fdb.png)

```jsx
**L1** = weight의 절댓값들의 합

— 대부분 0 값을 가져서, 일부의 weight만 남는다. (selection)
— 미분 X
— **sparse solution**
```

- **L2 regularizer**: aka '**ridge**'
![Untitled 3](https://user-images.githubusercontent.com/61778930/115956991-e09cf100-a53a-11eb-8bb3-c410cdf19386.png)

```jsx
**L2** = weight의 square 값의 합 (**weighted decay**)

— 미분 가능 (differentiable / convex) --> math friendly
— weight가 0이 되지 않음 (미세한 weight까지 다 포함시킴)
```

### 5-Cross-Validation

5-cross-validation (교차검증)이란,   
기본적으로 데이터를 test set과 train set으로 나누고,    
train set 내에서 data를 5개로 분할한 뒤 그 중 한 개를 **validation set**으로 정하는 방식이다.   

(그리하여 test data : validation data : train data의 비율은 약 1:1:8 정도가 된다)
![Untitled 4](https://user-images.githubusercontent.com/61778930/115956998-e85c9580-a53a-11eb-99f2-7acc53c14107.png)

**validation set**은 5개의 train set을 분할한 5개를 돌아가면서 선정하고, 그에 따른 에러율을 Edev_n 으로 정한다. (n=1,2,3,4,5)

우리는 각각의 validation set에 대해 Edev_n을 계산한 뒤 test data를 넣어 에러율을 구한다.

이를 통해 가장 최적의 값을 가지는 h를 구할 수 있다.
