# 3.1. Linear regression and Data preparation

```python
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
```

```python
print(diabetes.data.shape, diabetes.target.shape)
```

```python
diabetes.data[0:3]
```

```python
diabetes.target[:3]
```

```python
import matplotlib.pyplot as plt
plt.scatter(diabetes.data[:,2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

```python
x = diabetes.data[:, 2]
y = diabetes.target
```

# 3.2 Graident discent

```python
w = 1.0
b = 1.0
```

predicted value for first sample with initial w and b

```python
y_hat = x[0] * w + b
print(y_hat)
```

```python
def draw_point_line(i, j, w, b):
    plt.scatter(x[i:j+1], y[i:j+1])
    pt1 = (-0.1, -0.1 * w + b)
    pt2 = (0.15, 0.15 * w + b)
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
```

```python
draw_point_line(0,0,w,b)
```

```python
print(y[0])
```

```python
# predicted value for first sample with incremented w
w_inc = w + 0.1
y_hat_inc = x[0] * w_inc + b
print(y_hat_inc)
```

```python
w_rate = (y_hat_inc - y_hat) / (w_inc - w) # ≈ d y_hat / d w at a point x[0]
print(w_rate) # w_rate tells two: to which direction and how much w should go
```

```python
w_new = w + w_rate
print(w_new)
```

```python
draw_point_line(0,0,w_new,b)
```

```python
b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc
b_rate = (y_hat_inc - y_hat) / (b_inc - b) # ≈ d y_hat / d b at a point x[0]
print(b_rate)
```

```python
b_new = b + 1
print(b_new)
```

```python
draw_point_line(0,0,w,b_new)
```

Two Problems: 
1. it gets closer to the point too slowly
2. it will only increse even after meeting the point


## Error Backpropagation

```python
err = y[0] - y_hat
w_new = w + w_rate * err
b_new = b + 1 * err # 1 is b_rate
print(w_new, b_new)
```

```python
draw_point_line(0,0,w_new,b_new)
```

adjust params with the second data point

```python
y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]
w_new = w_new + w_rate * err
b_new = b_new + 1 * err
print(w_new, b_new)
```

```python
draw_point_line(0, 1, w_new, b_new)
```

iterate over all data points

```python
for x_i, y_i in zip(x,y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
```

```python
draw_point_line(0, len(x) + 1, w, b)
```

iterate whole process several times: epoch

```python
for i in range(1, 100):
    for x_i, y_i in zip(x,y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err
```

```python
draw_point_line(0, len(x) + 1, w, b)
```

predict for new input

```python
x_new = 0.18
y_pred = x_new * w + b
print(y_pred)
```

```python
plt.scatter(x, y)
plt.scatter(x_new, y_pred, c = "red")
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
