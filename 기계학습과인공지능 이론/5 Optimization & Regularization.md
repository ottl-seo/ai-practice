# 5. Optimization & Regularization

## Optimization

- 

---

## Regularization

- what is it?

    모델이 too complex해지면 noise까지 과도하게 학습시키는 overfit 발생
    —> overfit 하지 않도록 모델을 조정(**constrain**)해주는 것이 정규화의 목적

    "**`constrain`** *a model so that it can avoid fitting noise"*

![5%20Optimization%20&%20Regularization%200b37f38771bc410fab7ccbba70ecffa6/Untitled.png](5%20Optimization%20&%20Regularization%200b37f38771bc410fab7ccbba70ecffa6/Untitled.png)

### Regularization method

1. **w 등 변수**를 바꿔준다.
2. **model complexity 더해준다**. (Ein **+ Ω**)

    Ein을 minimize하려다보니 Overfit 발생
    —> 모델 복잡도 Ω를 더해줌으로써 Ein ~= Eout 상태로 만든다.

3. **Ensemble methods** (앙상블 메서드)

### Regularization의 목표

: Overfit 방지 ( case3의 상태를 case2 로 돌리는 작업)

![5%20Optimization%20&%20Regularization%200b37f38771bc410fab7ccbba70ecffa6/Untitled%201.png](5%20Optimization%20&%20Regularization%200b37f38771bc410fab7ccbba70ecffa6/Untitled%201.png)

---

### Most famous regularizers; L1 and L2

- **L1 regularizer**: aka '**lasso**'

    ![5%20Optimization%20&%20Regularization%200b37f38771bc410fab7ccbba70ecffa6/Untitled%202.png](5%20Optimization%20&%20Regularization%200b37f38771bc410fab7ccbba70ecffa6/Untitled%202.png)

```jsx
**L1** = weight의 절댓값들의 합

— 대부분 0 값을 가져서, 일부의 weight만 남는다. (selection)
— 미분 X
— **sparse solution**
```

- **L2 regularizer**: aka '**ridge**'

    ![5%20Optimization%20&%20Regularization%200b37f38771bc410fab7ccbba70ecffa6/Untitled%203.png](5%20Optimization%20&%20Regularization%200b37f38771bc410fab7ccbba70ecffa6/Untitled%203.png)

```jsx
**L2** = weight의 square 값의 합 (**weighted decay**)

— 미분 가능 (differentiable / convex) --> math friendly
— weight가 0이 되지 않음 (미세한 weight까지 다 포함시킴)
```