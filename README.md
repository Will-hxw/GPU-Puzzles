# GPU 谜题

  - 作者：[Sasha Rush](http://rush-nlp.com) - [srush\_nlp](https://twitter.com/srush_nlp)

GPU 架构对于机器学习至关重要，并且其重要性与日俱增。然而，你可能是一位机器学习专家，却从未接触过 GPU 代码。通过抽象来建立直觉是很困难的。

本笔记试图以完全互动的方式教授初学者 GPU 编程。它没有提供带有概念的文本，而是直接让你进入编码和构建 GPU 内核的实践。这些练习使用 NUMBA，它可以将 Python 代码直接映射到 CUDA 内核。它看起来像 Python，但基本上与编写低级 CUDA 代码相同。
我认为，在几个小时内，你就可以从基础知识入门，到理解当今 99% 深度学习所依赖的真实算法。如果你确实想阅读手册，可以在这里找到：

[NUMBA CUDA 指南](https://numba.readthedocs.io/en/stable/cuda/index.html)

我建议在 Colab 中完成这些练习，因为它很容易上手。请务必创建自己的副本，在设置中打开 GPU 模式（`Runtime / Change runtime type`，然后将 `Hardware accelerator` 设置为 `GPU`），然后开始编码。

[](https://colab.research.google.com/github/srush/GPU-Puzzles/blob/main/GPU_puzzlers.ipynb)

（如果你喜欢这种风格的谜题，也可以看看我的 [Tensor 谜题](https://github.com/srush/Tensor-Puzzles) for PyTorch。）

[演练指南](https://www.youtube.com/watch?v=K4T-YwsOxrM)

```python
# 安装 chalk 库，该库用于在终端以不同颜色输出文本，增强可读性。
# 使用 -qqq 参数以静默模式安装，减少不必要的输出。
# git+https://github.com/danoneata/chalk@srush-patch-1 指定了从特定分支安装。
!pip install -qqq git+https://github.com/danoneata/chalk@srush-patch-1

# 下载 robot.png 图片文件和 lib.py 库文件。
# -q 参数表示静默模式，不显示下载进度。
# 这两个文件是 GPU-Puzzles 项目的资源文件。
!wget -q https://github.com/srush/GPU-Puzzles/raw/main/robot.png https://github.com/srush/GPU-Puzzles/raw/main/lib.py
```

```python
# 导入 numba 库，它是一个即时编译器 (JIT compiler)，可以将 Python 函数转换为优化的机器码，
# 特别是对于 CUDA 编程，可以将 Python 子集编译为高效的 GPU 内核代码。
import numba

# 导入 numpy 库，并使用别名 np。Numpy 是 Python 中用于科学计算的核心库，
# 提供了对多维数组和矩阵运算的支持。
import numpy as np

# 导入 warnings 库，用于控制和处理 Python 程序运行过程中产生的警告信息。
import warnings

# 从自定义的 lib 模块中导入 CudaProblem 和 Coord 类。
# CudaProblem 类可能用于封装和管理 CUDA 编程谜题的测试和验证。
# Coord 类可能用于表示坐标或维度信息，例如线程块的维度。
from lib import CudaProblem, Coord
```

```python
# 过滤掉特定类型的警告：numba.NumbaPerformanceWarning。
# 当 Numba 检测到某些代码模式可能导致性能不佳时，会发出此警告。
# action="ignore" 表示忽略这类警告，不将其显示出来。
# category=numba.NumbaPerformanceWarning 指定了要忽略的警告类别。
# module="numba" 指定了只忽略来自 numba 模块的这类警告。
warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)
```

## 谜题 1: 映射 (Map)

实现一个“内核”（GPU 函数），该函数将向量 `a` 的每个位置上的值增加 10，并将结果存储在向量 `out` 中。每个位置分配一个线程。

**警告** 这段代码看起来像 Python，但它实际上是 CUDA！你不能使用标准的 Python 工具，如列表推导，也不能请求 Numpy 的属性，如形状或大小（如果需要大小，它会作为参数给出）。这些谜题只需要执行简单的操作，基本上是 `+`、`*`、简单的数组索引、for 循环和 if 语句。允许使用局部变量。如果你遇到错误，很可能是因为你使用了一些高级的 Python 特性 :)。

*提示：可以将函数 `call` 视为每个线程运行一次。唯一的区别是 `cuda.threadIdx.x` 每次都会改变。*

```python
# 定义 map_spec 函数，用于生成期望的输出结果。
# 这个函数在 CPU 上执行，作为 GPU 内核实现的参考标准。
# 它接收一个 numpy 数组 a 作为输入。
def map_spec(a):
    # 将输入数组 a 的每个元素加 10，并返回结果数组。
    # 这是标准的 numpy 向量化操作。
    return a + 10


# 定义 map_test 函数，该函数返回一个 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入，提供了访问 CUDA 特定功能的接口。
def map_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组，用于存储结果。
    # a: 输入数组。
    def call(out, a) -> None:
        # 获取当前线程在线程块内的 x 方向索引。
        # 对于一维问题，这通常直接对应于数组元素的索引。
        local_i = cuda.threadIdx.x
        
        # 核心逻辑：将输入数组 a 中对应当前线程索引的元素值加 10，
        # 然后将结果存入输出数组 out 的相同索引位置。
        # 每个线程处理数组中的一个元素。
        out[local_i] = a[local_i] + 10
    # 返回定义的内核函数。
    return call


# 定义输入数组的大小。
SIZE = 4
# 初始化输出数组 out，长度为 SIZE，所有元素填充为 0。
# 这个数组将被 GPU 内核修改。
out = np.zeros((SIZE,))
# 初始化输入数组 a，包含从 0 到 SIZE-1 的整数序列 (0, 1, 2, 3)。
a = np.arange(SIZE)

# 创建 CudaProblem 实例，用于配置和运行 "Map" 谜题的测试。
# "Map": 谜题的名称。
# map_test: 用于生成 CUDA 内核的函数。
# [a]: 传递给 CUDA 内核的输入参数列表 (这里只有一个输入数组 a)。
# out: 传递给 CUDA 内核的输出参数。
# threadsperblock=Coord(SIZE, 1): 指定每个线程块的线程数量。
#   Coord(SIZE, 1) 表示在 x 方向有 SIZE 个线程，y 方向有 1 个线程 (适用于一维问题)。
# spec=map_spec: 用于生成期望结果的参考函数。
problem = CudaProblem(
    "Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec
)
# 显示谜题的相关信息，例如输入、期望输出和 GPU 执行的配置。
problem.show()
```

```
# 映射
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 |
```

```python
# 检查 GPU 内核执行的结果是否与 map_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [0. 0. 0. 0.]
期望结果: [10 11 12 13]
```

## 谜题 2 - 合并 (Zip)

实现一个内核，将 `a` 和 `b` 中每个位置的元素相加，并将结果存储在 `out` 中。每个位置分配一个线程。

```python
# 定义 zip_spec 函数，用于生成期望的输出结果 (在 CPU 上执行)。
# 它接收两个 numpy 数组 a 和 b 作为输入。
def zip_spec(a, b):
    # 将输入数组 a 和 b 的对应元素相加，并返回结果数组。
    # 这是标准的 numpy 向量化操作。
    return a + b


# 定义 zip_test 函数，该函数返回一个 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def zip_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组。
    # a: 第一个输入数组。
    # b: 第二个输入数组。
    def call(out, a, b) -> None:
        # 获取当前线程在线程块内的 x 方向索引。
        local_i = cuda.threadIdx.x
        
        # 核心逻辑：将输入数组 a 和 b 中对应当前线程索引的元素值相加，
        # 然后将结果存入输出数组 out 的相同索引位置。
        # 每个线程处理一对来自 a 和 b 的元素。
        out[local_i] = a[local_i] + b[local_i]
    # 返回定义的内核函数。
    return call


# 定义输入数组的大小。
SIZE = 4
# 初始化输出数组 out，长度为 SIZE，所有元素填充为 0。
out = np.zeros((SIZE,))
# 初始化第一个输入数组 a，包含从 0 到 SIZE-1 的整数序列 (0, 1, 2, 3)。
a = np.arange(SIZE)
# 初始化第二个输入数组 b，同样包含从 0 到 SIZE-1 的整数序列 (0, 1, 2, 3)。
b = np.arange(SIZE)

# 创建 CudaProblem 实例，用于配置和运行 "Zip" 谜题的测试。
# "Zip": 谜题的名称。
# zip_test: 用于生成 CUDA 内核的函数。
# [a, b]: 传递给 CUDA 内核的输入参数列表。
# out: 传递给 CUDA 内核的输出参数。
# threadsperblock=Coord(SIZE, 1): 每个线程块的线程数量。
# spec=zip_spec: 用于生成期望结果的参考函数。
problem = CudaProblem(
    "Zip", zip_test, [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec
)
# 显示谜题的相关信息。
problem.show()
```

```
# 合并
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 此处原有一个空的 code cell，通常用于用户自行添加代码或测试。
# 在此上下文中，它可能是为了让用户在检查前尝试填充缺失的代码。
```

```python
# 检查 GPU 内核执行的结果是否与 zip_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [0. 0. 0. 0.]
期望结果: [0 2 4 6]
```

## 谜题 3 - 守卫 (Guards)

实现一个内核，将 `a` 中每个位置的元素加 10，并将结果存储在 `out` 中。线程数多于位置数。

```python
# 定义 map_guard_test 函数，该函数返回一个带边界检查的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def map_guard_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组。
    # a: 输入数组。
    # size: 输入数组 a 的实际大小 (长度)。用于边界检查。
    def call(out, a, size) -> None:
        # 获取当前线程在线程块内的 x 方向索引。
        local_i = cuda.threadIdx.x
        
        # 核心逻辑：添加边界检查 (守卫)。
        # 由于线程数可能多于数组的实际元素数量，
        # 需要确保当前线程索引 local_i 没有超出数组的有效范围。
        if local_i < size:
            # 如果索引有效，则执行加法操作并存储结果。
            out[local_i] = a[local_i] + 10
    # 返回定义的内核函数。
    return call


# 定义输入数组的实际大小。
SIZE = 4
# 初始化输出数组 out，长度为 SIZE，所有元素填充为 0。
out = np.zeros((SIZE,))
# 初始化输入数组 a，包含从 0 到 SIZE-1 的整数序列 (0, 1, 2, 3)。
a = np.arange(SIZE)

# 创建 CudaProblem 实例，用于配置和运行 "Guard" 谜题的测试。
# "Guard": 谜题的名称。
# map_guard_test: 用于生成 CUDA 内核的函数。
# [a]: 传递给 CUDA 内核的输入参数列表 (输入数组 a)。
# out: 传递给 CUDA 内核的输出参数。
# [SIZE]: 传递给 CUDA 内核的额外参数 (数组的实际大小)。
# threadsperblock=Coord(8, 1): 指定每个线程块的线程数量。
#   这里线程数 (8) 大于数组的实际大小 (4)，因此需要边界检查。
# spec=map_spec: 用于生成期望结果的参考函数 (与谜题1相同)。
problem = CudaProblem(
    "Guard",
    map_guard_test,
    [a],
    out,
    [SIZE], 
    threadsperblock=Coord(8, 1), 
    spec=map_spec, 
)
# 显示谜题的相关信息。
problem.show()
```

```
# 守卫
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查 GPU 内核执行的结果是否与 map_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [0. 0. 0. 0.]
期望结果: [10 11 12 13]
```

## 谜题 4 - 二维映射 (Map 2D)

实现一个内核，将 `a` 中每个位置的元素加 10，并将结果存储在 `out` 中。输入 `a` 是二维方阵。线程数多于位置数。

```python
# 定义 map_2D_test 函数，该函数返回一个处理二维数组的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def map_2D_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出的二维数组。
    # a: 输入的二维数组。
    # size: 输入二维数组 a 的维度大小 (假设为方阵，size x size)。
    def call(out, a, size) -> None:
        # 获取当前线程在线程块内的 x 方向索引，对应二维数组的列索引。
        local_i = cuda.threadIdx.x
        # 获取当前线程在线程块内的 y 方向索引，对应二维数组的行索引。
        local_j = cuda.threadIdx.y
        
        # 核心逻辑：添加针对二维数组的边界检查。
        # 确保线程索引 (local_j, local_i) (行, 列) 没有超出数组的有效范围。
        if local_i < size and local_j < size:
            # 如果索引有效，则对 a[local_j, local_i] 执行加法操作并存储结果到 out[local_j, local_i]。
            # 注意：通常习惯将 y 索引视为行，x 索引视为列，即 a[row, col] -> a[local_j, local_i]。
            # 此处代码实现为 a[local_i, local_j]，可能需要根据实际数组存储和访问习惯调整。
            # 假设 numba 默认 cuda.threadIdx.x 对应第一维, cuda.threadIdx.y 对应第二维，
            # 或者用户期望 local_i 为行，local_j 为列。
            # 根据题目中 CudaProblem 的 threadsperblock=Coord(3,3) 和数组的显示方式，
            # 通常 Coord(x,y) 中 x 对应 cuda.threadIdx.x, y 对应 cuda.threadIdx.y。
            # 若 a 是 (SIZE, SIZE)，则 a[row, col] -> a[cuda.threadIdx.y, cuda.threadIdx.x]。
            # 因此，这里应该是 out[local_j, local_i] = a[local_j, local_i] + 10
            # 题目提供的填充代码是 out[local_i, local_j] = a[local_i, local_j] + 10，
            # 这意味着 threadIdx.x (local_i) 对应行，threadIdx.y (local_j) 对应列。
            # 我们将遵循题目已有的变量命名 local_i, local_j。
            # 如果 local_i 是行，local_j 是列：
            out[local_i, local_j] = a[local_i, local_j] + 10
    # 返回定义的内核函数。
    return call


# 定义二维方阵的维度大小。
SIZE = 2
# 初始化输出二维数组 out，形状为 (SIZE, SIZE)，所有元素填充为 0。
out = np.zeros((SIZE, SIZE))
# 初始化输入二维数组 a，元素为 [[0, 1], [2, 3]]。
# np.arange(SIZE * SIZE) 生成 [0, 1, 2, 3]。
# .reshape((SIZE, SIZE)) 将其转换为 2x2 矩阵。
a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))

# 创建 CudaProblem 实例，用于配置和运行 "Map 2D" 谜题的测试。
# "Map 2D": 谜题的名称。
# map_2D_test: 用于生成 CUDA 内核的函数。
# [a]: 传递给 CUDA 内核的输入参数列表。
# out: 传递给 CUDA 内核的输出参数。
# [SIZE]: 传递给 CUDA 内核的额外参数 (方阵的维度大小)。
# threadsperblock=Coord(3, 3): 指定每个线程块的线程数量为 3x3。
#   由于 SIZE=2，线程数 (3x3=9) 大于实际元素数 (2x2=4)，需要边界检查。
# spec=map_spec: 用于生成期望结果的参考函数 (与谜题1相同，Numpy的+10操作也适用于多维数组)。
problem = CudaProblem(
    "Map 2D", map_2D_test, [a], out, [SIZE], threadsperblock=Coord(3, 3), spec=map_spec
)
# 显示谜题的相关信息。
problem.show()
```

```
# Map 2D
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查 GPU 内核执行的结果是否与 map_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [[0. 0.]
  [0. 0.]]
期望结果: [[10 11]
  [12 13]]
```

## 谜题 5 - 广播 (Broadcast)

实现一个内核，将 `a` 和 `b` 相加，并将结果存储在 `out` 中。输入 `a` 和 `b` 是向量。线程数多于位置数。
(更准确地说，根据代码，a 是列向量 (SIZE, 1)，b 是行向量 (1, SIZE)，out 是 (SIZE, SIZE) 的矩阵。这是一个外积加法的形式。)

```python
# 定义 broadcast_test 函数，该函数返回一个实现广播加法的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def broadcast_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出的二维数组 (结果矩阵)。
    # a: 输入的列向量 (二维数组，形状为 size x 1)。
    # b: 输入的行向量 (二维数组，形状为 1 x size)。
    # size: 向量的长度或结果矩阵的维度。
    def call(out, a, b, size) -> None:
        # 获取当前线程在线程块内的 x 方向索引，对应输出矩阵的列索引。
        local_i = cuda.threadIdx.x # 对应输出矩阵的行索引 (按照题目填充习惯)
        # 获取当前线程在线程块内的 y 方向索引，对应输出矩阵的行索引。
        local_j = cuda.threadIdx.y # 对应输出矩阵的列索引 (按照题目填充习惯)
        
        # 核心逻辑：实现广播加法。
        # out[row, col] = a[row, 0] + b[0, col]
        # 根据题目对 local_i, local_j 的使用习惯 (local_i 为行，local_j 为列)：
        # 确保线程索引 (local_i, local_j) (行, 列) 没有超出输出矩阵的有效范围。
        if local_i < size and local_j < size:
            # a 是列向量，访问其元素时，列索引固定为 0。行索引是 local_i。
            # b 是行向量，访问其元素时，行索引固定为 0。列索引是 local_j。
            # 将 a 的第 local_i 个元素与 b 的第 local_j 个元素相加，
            # 结果存入 out[local_i, local_j]。
            out[local_i, local_j] = a[local_i, 0] + b[0, local_j]
    # 返回定义的内核函数。
    return call


# 定义向量的长度或结果矩阵的维度。
SIZE = 2
# 初始化输出二维数组 out，形状为 (SIZE, SIZE)，所有元素填充为 0。
out = np.zeros((SIZE, SIZE))
# 初始化输入列向量 a，形状为 (SIZE, 1)。
# np.arange(SIZE) 生成 [0, 1]。
# .reshape(SIZE, 1) 将其转换为 [[0], [1]]。
a = np.arange(SIZE).reshape(SIZE, 1)
# 初始化输入行向量 b，形状为 (1, SIZE)。
# np.arange(SIZE) 生成 [0, 1]。
# .reshape(1, SIZE) 将其转换为 [[0, 1]]。
b = np.arange(SIZE).reshape(1, SIZE)

# 创建 CudaProblem 实例，用于配置和运行 "Broadcast" 谜题的测试。
# "Broadcast": 谜题的名称。
# broadcast_test: 用于生成 CUDA 内核的函数。
# [a, b]: 传递给 CUDA 内核的输入参数列表。
# out: 传递给 CUDA 内核的输出参数。
# [SIZE]: 传递给 CUDA 内核的额外参数 (维度大小)。
# threadsperblock=Coord(3, 3): 指定每个线程块的线程数量为 3x3。
#   由于 SIZE=2，线程数 (9) 大于实际元素数 (4)，需要边界检查。
# spec=zip_spec: 用于生成期望结果的参考函数。
#   注意：这里的 zip_spec(a,b) 是 a+b，对于numpy，如果a是(S,1), b是(1,S)，a+b会执行广播。
#   所以 zip_spec(a,b) 的结果是 [[0,1],[1,2]] + [[0,0],[1,1]] (如果a,b是普通向量)
#   或者 a = [[0],[1]], b = [[0,1]] => a+b = [[0,1],[1,2]] (numpy广播)
#   这与 out[i,j] = a[i,0] + b[0,j] 的结果一致。
problem = CudaProblem(
    "Broadcast",
    broadcast_test,
    [a, b],
    out,
    [SIZE], 
    threadsperblock=Coord(3, 3), 
    spec=zip_spec, 
)
# 显示谜题的相关信息。
problem.show()
```

```
# 广播
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查 GPU 内核执行的结果是否与 zip_spec 函数 (通过Numpy广播) 生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [[0. 0.]
  [0. 0.]]
期望结果: [[0 1]
  [1 2]]
```

## 谜题 6 - 块 (Blocks)

实现一个内核，将 `a` 中每个位置的元素加 10，并将结果存储在 `out` 中。每个块中的线程数少于 `a` 的大小。

*提示：一个块是一组线程。每个块的线程数是有限的，但我们可以有很多不同的块。变量 `cuda.blockIdx` 告诉我们当前处于哪个块。*

```python
# 定义 map_block_test 函数，该函数返回一个使用多线程块处理一维数组的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def map_block_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组。
    # a: 输入数组。
    # size: 输入数组 a 的实际大小。
    def call(out, a, size) -> None:
        # 计算当前线程的全局索引 i。
        # cuda.blockIdx.x: 当前线程块在网格 (grid) 中的 x 方向索引。
        # cuda.blockDim.x: 每个线程块在 x 方向上的线程数量。
        # cuda.threadIdx.x: 当前线程在其所属线程块内的 x 方向索引。
        # 公式：全局索引 = (块索引 * 块内线程数) + 线程在块内索引
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        # 核心逻辑：添加边界检查。
        # 确保计算得到的全局索引 i 没有超出数组的有效范围。
        if i < size:
            # 如果索引有效，则对 a[i] 执行加法操作并存储结果到 out[i]。
            out[i] = a[i] + 10
    # 返回定义的内核函数。
    return call


# 定义输入数组的大小。
SIZE = 9
# 初始化输出数组 out，长度为 SIZE，所有元素填充为 0。
out = np.zeros((SIZE,))
# 初始化输入数组 a，包含从 0 到 SIZE-1 的整数序列 (0 到 8)。
a = np.arange(SIZE)

# 创建 CudaProblem 实例，用于配置和运行 "Blocks" 谜题的测试。
# "Blocks": 谜题的名称。
# map_block_test: 用于生成 CUDA 内核的函数。
# [a]: 传递给 CUDA 内核的输入参数列表。
# out: 传递给 CUDA 内核的输出参数。
# [SIZE]: 传递给 CUDA 内核的额外参数 (数组的实际大小)。
# threadsperblock=Coord(4, 1): 指定每个线程块有 4 个线程 (在 x 方向)。
# blockspergrid=Coord(3, 1): 指定网格中有 3 个线程块 (在 x 方向)。
#   总线程数 = 4 (线程/块) * 3 (块) = 12 个线程。
#   由于数组大小 SIZE = 9，总线程数 (12) 大于元素数 (9)，因此需要边界检查。
# spec=map_spec: 用于生成期望结果的参考函数。
problem = CudaProblem(
    "Blocks",
    map_block_test,
    [a],
    out,
    [SIZE], 
    threadsperblock=Coord(4, 1), 
    blockspergrid=Coord(3, 1),  
    spec=map_spec, 
)
# 显示谜题的相关信息。
problem.show()
```

```
# 块
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查 GPU 内核执行的结果是否与 map_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [0. 0. 0. 0. 0. 0. 0. 0. 0.]
期望结果: [10 11 12 13 14 15 16 17 18]
```

## 谜题 7 - 二维块 (Blocks 2D)

在二维情况下实现相同的内核。在两个方向上，每个块的线程数都少于 `a` 的大小。

```python
# 定义 map_block2D_test 函数，该函数返回一个使用二维线程块处理二维数组的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def map_block2D_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出的二维数组。
    # a: 输入的二维数组。
    # size: 输入二维数组 a 的维度大小 (假设为方阵，size x size)。
    def call(out, a, size) -> None:
        # 计算当前线程的全局行索引 i。
        # cuda.blockIdx.x: 当前线程块在网格中的 x 方向索引 (对应行块)。
        # cuda.blockDim.x: 每个线程块在 x 方向上的线程数量 (对应行内线程数)。
        # cuda.threadIdx.x: 当前线程在其块内 x 方向的索引 (对应行内局部索引)。
        # 假设 x 方向对应行，y 方向对应列。
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x # 全局行索引
        
        # 计算当前线程的全局列索引 j。
        # cuda.blockIdx.y: 当前线程块在网格中的 y 方向索引 (对应列块)。
        # cuda.blockDim.y: 每个线程块在 y 方向上的线程数量 (对应列内线程数)。
        # cuda.threadIdx.y: 当前线程在其块内 y 方向的索引 (对应列内局部索引)。
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y # 全局列索引
        
        # 核心逻辑：添加针对二维数组的边界检查。
        # 确保计算得到的全局索引 (i, j) (行, 列) 没有超出数组的有效范围。
        if i < size and j < size:
            # 如果索引有效，则对 a[i, j] 执行加法操作并存储结果到 out[i, j]。
            out[i, j] = a[i, j] + 10
    # 返回定义的内核函数。
    return call


# 定义二维方阵的维度大小。
SIZE = 5
# 初始化输出二维数组 out，形状为 (SIZE, SIZE)，所有元素填充为 0。
out = np.zeros((SIZE, SIZE))
# 初始化输入二维数组 a，形状为 (SIZE, SIZE)，所有元素填充为 1。
a = np.ones((SIZE, SIZE))

# 创建 CudaProblem 实例，用于配置和运行 "Blocks 2D" 谜题的测试。
# "Blocks 2D": 谜题的名称。
# map_block2D_test: 用于生成 CUDA 内核的函数。
# [a]: 传递给 CUDA 内核的输入参数列表。
# out: 传递给 CUDA 内核的输出参数。
# [SIZE]: 传递给 CUDA 内核的额外参数 (方阵的维度大小)。
# threadsperblock=Coord(3, 3): 指定每个线程块的线程数量为 3x3。
# blockspergrid=Coord(2, 2): 指定网格中的线程块数量为 2x2。
#   总线程数在 x 方向 (行): 3 (线程/块) * 2 (块) = 6 线程。
#   总线程数在 y 方向 (列): 3 (线程/块) * 2 (块) = 6 线程。
#   数组大小 SIZE = 5。由于总线程数 (6x6) 大于元素数 (5x5)，需要边界检查。
# spec=map_spec: 用于生成期望结果的参考函数 (Numpy的+10也适用于多维数组，且这里输入a是全1，所以期望输出是全11)。
problem = CudaProblem(
    "Blocks 2D",
    map_block2D_test,
    [a],
    out,
    [SIZE], 
    threadsperblock=Coord(3, 3), 
    blockspergrid=Coord(2, 2),  
    spec=map_spec, 
)
# 显示谜题的相关信息。
problem.show()
```

```
# 二维块
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查 GPU 内核执行的结果是否与 map_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [[0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]]
期望结果: [[11. 11. 11. 11. 11.]
  [11. 11. 11. 11. 11.]
  [11. 11. 11. 11. 11.]
  [11. 11. 11. 11. 11.]
  [11. 11. 11. 11. 11.]]
```

## 谜题 8 - 共享内存 (Shared)

实现一个内核，将 `a` 中每个位置的元素加 10，并将结果存储在 `out` 中。每个块中的线程数少于 `a` 的大小。

**警告**：每个块只能拥有*常量*大小的共享内存，供该块中的线程读取和写入。这需要是一个字面上的 Python 常量，而不是变量。写入共享内存后，你需要调用 `cuda.syncthreads()` 以确保线程不会交叉。

（这个例子并不真正需要共享内存或 `syncthreads`，但它是一个演示。）

```python
# 定义每个线程块的线程数 (Threads Per Block)。这是一个常量。
TPB = 4

# 定义 shared_test 函数，该函数返回一个使用共享内存的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def shared_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组。
    # a: 输入数组。
    # size: 输入数组 a 的实际大小。
    def call(out, a, size) -> None:
        # 在共享内存中创建一个一维数组 `shared`，大小为 TPB，数据类型为 numba.float32。
        # 共享内存是块内线程可见的高速缓存。
        shared = cuda.shared.array(TPB, numba.float32)
        
        # 计算当前线程的全局索引 i。
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # 获取当前线程在其所属线程块内的局部索引 local_i。
        local_i = cuda.threadIdx.x

        # 阶段 1: 从全局内存加载数据到共享内存。
        # 每个线程负责加载一个元素到共享内存的对应位置。
        if i < size: # 边界检查，确保不越界访问全局内存 a
            shared[local_i] = a[i]
        
        # 同步块内所有线程。
        # 确保所有线程都完成了对共享内存的写入操作，
        # 然后其他线程才能安全地读取这些值。
        cuda.syncthreads()

        # 阶段 2: 从共享内存读取数据，执行计算，并将结果写回全局内存。
        # 核心逻辑：如果全局索引 i 有效，
        # 则从共享内存 `shared` 中读取之前加载的值，加 10，
        # 并将结果存入输出数组 `out` 的相应位置。
        if i < size: # 边界检查，确保不越界写入全局内存 out
            out[i] = shared[local_i] + 10
            # 在这个特定例子中，由于每个线程独立地从共享内存读取并写入全局内存的不同位置，
            # 此处第二次同步不是严格必需的。
            # 但如果在后续操作中，线程间存在对共享内存的进一步依赖（例如，写后读），则可能需要再次同步。
            # cuda.syncthreads() # 如果后续有依赖，则取消注释此行
    # 返回定义的内核函数。
    return call


# 定义输入数组的大小。
SIZE = 8
# 初始化输出数组 out，长度为 SIZE，所有元素填充为 0。
out = np.zeros(SIZE)
# 初始化输入数组 a，长度为 SIZE，所有元素填充为 1。
a = np.ones(SIZE)

# 创建 CudaProblem 实例，用于配置和运行 "Shared" 谜题的测试。
# "Shared": 谜题的名称。
# shared_test: 用于生成 CUDA 内核的函数。
# [a]: 传递给 CUDA 内核的输入参数列表。
# out: 传递给 CUDA 内核的输出参数。
# [SIZE]: 传递给 CUDA 内核的额外参数 (数组的实际大小)。
# threadsperblock=Coord(TPB, 1): 每个块有 TPB (即 4) 个线程。
# blockspergrid=Coord(SIZE // TPB, 1): 网格中的块数。
#   SIZE // TPB = 8 // 4 = 2 个块。总共 4*2 = 8 个线程，正好等于数组大小。
# spec=map_spec: 用于生成期望结果的参考函数 (输入是全1，所以期望输出是全11)。
problem = CudaProblem(
    "Shared",
    shared_test,
    [a],
    out,
    [SIZE], 
    threadsperblock=Coord(TPB, 1), 
    blockspergrid=Coord(SIZE // TPB, 1), 
    spec=map_spec, 
)
# 显示谜题的相关信息。
problem.show()
```

```
# 共享
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 1 | 0 | 0 | 1 | 
```

```python
# 检查 GPU 内核执行的结果是否与 map_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [0. 0. 0. 0. 0. 0. 0. 0.]
期望结果: [11. 11. 11. 11. 11. 11. 11. 11.]
```

## 谜题 9 - 池化 (Pooling)

实现一个内核，将 `a` 中最后 3 个位置的元素相加，并将结果存储在 `out` 中。每个位置分配一个线程。每个线程只需要 1 次全局读取和 1 次全局写入。
(根据 `pool_spec` 的定义，`out[i] = a[max(i - 2, 0) : i + 1].sum()`，这意味着 `out[i]` 是 `a[i]` 以及其前面（最多）两个元素的和。)

*提示：记住要小心同步。*

```python
# 定义 pool_spec 函数，用于生成期望的池化结果 (在 CPU 上执行)。
# 它接收一个 numpy 数组 a 作为输入。
def pool_spec(a):
    # 初始化输出数组 out，形状与 a 相同，所有元素填充为 0。
    out = np.zeros(*a.shape)
    # 遍历输入数组 a 的每个元素索引 i。
    for i in range(a.shape[0]):
        # 计算池化窗口的起始索引。窗口包含当前元素及其前两个元素。
        # 使用 max(i - 2, 0)确保起始索引不小于0 (处理边界情况)。
        # 池化窗口为 a[start_index : i + 1]。
        # 计算这个窗口内所有元素的和，并存入 out[i]。
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    # 返回池化后的结果数组。
    return out


# 定义每个线程块的线程数。
TPB = 8

# 定义 pool_test 函数，该函数返回一个实现池化操作的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def pool_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组。
    # a: 输入数组。
    # size: 输入数组 a 的实际大小。
    def call(out, a, size) -> None:
        # 在共享内存中创建一个一维数组 `shared`，大小为 TPB。
        shared = cuda.shared.array(TPB, numba.float32)
        
        # 计算当前线程的全局索引 i。
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # 获取当前线程在其所属线程块内的局部索引 local_i。
        local_i = cuda.threadIdx.x
        
        # 阶段 1: 从全局内存加载数据到共享内存。
        # 每个线程负责加载一个元素 a[i] 到共享内存的 shared[local_i]。
        if i < size: # 边界检查
            shared[local_i] = a[i]
        else:
            # 对于超出数组范围的线程 (如果块大小不是数组大小的整数倍，或者总线程数大于数组大小)，
            # 在共享内存的对应位置填充一个不影响求和的值 (例如 0.0)。
            # 这一步是为了简化后续计算的边界条件处理，确保共享内存中总是有定义的值。
            shared[local_i] = 0.0 

        # 同步块内所有线程，确保所有数据已加载到共享内存。
        cuda.syncthreads()

        # 阶段 2: 从共享内存读取数据，执行池化计算，并将结果写回全局内存。
        if i < size: # 只为有效的输出位置计算结果
            # 初始化当前元素的池化和。
            current_sum = 0.0
            
            # 根据 pool_spec 的定义: out[i] = a[i] + a[i-1] (if i-1>=0) + a[i-2] (if i-2>=0)
            # 这些值分别对应共享内存中的 shared[local_i], shared[local_i-1], shared[local_i-2]
            # (假设当前线程块处理的是一个连续的数据段，并且池化窗口完全在该段内，
            # 或者说 TPB 足够大以容纳池化窗口所需的元素)。
            # 对于这个谜题的设置 (TPB=8, SIZE=8, blockspergrid=1)，单个块处理整个数组，
            # 所以 shared[local_i-1] 和 shared[local_i-2] 的访问是有效的 (在检查 local_i 边界后)。

            # 加上 a[i] (即 shared[local_i])
            current_sum += shared[local_i]
            
            # 加上 a[i-1] (即 shared[local_i-1])，如果 i-1 >= 0
            # 对应于检查 local_i >= 1 (因为 local_i 是当前元素在共享内存中的索引)
            if local_i >= 1:
                current_sum += shared[local_i-1]
            
            # 加上 a[i-2] (即 shared[local_i-2])，如果 i-2 >= 0
            # 对应于检查 local_i >= 2
            if local_i >= 2:
                current_sum += shared[local_i-2]
            
            # 将计算得到的池化和写入输出数组 out[i]。
            out[i] = current_sum
    # 返回定义的内核函数。
    return call


# 定义输入数组的大小。
SIZE = 8
# 初始化输出数组 out，长度为 SIZE，所有元素填充为 0。
out = np.zeros(SIZE)
# 初始化输入数组 a，包含从 0 到 SIZE-1 的整数序列 (0 到 7)。
a = np.arange(SIZE)

# 创建 CudaProblem 实例，用于配置和运行 "Pooling" 谜题的测试。
# "Pooling": 谜题的名称。
# pool_test: 用于生成 CUDA 内核的函数。
# [a]: 传递给 CUDA 内核的输入参数列表。
# out: 传递给 CUDA 内核的输出参数。
# [SIZE]: 传递给 CUDA 内核的额外参数 (数组的实际大小)。
# threadsperblock=Coord(TPB, 1): 每个块有 TPB (即 8) 个线程。
# blockspergrid=Coord(1, 1): 网格中只有 1 个块。
#   (总线程数 8*1=8，等于数组大小，每个线程负责一个输出元素)
# spec=pool_spec: 用于生成期望结果的参考函数。
problem = CudaProblem(
    "Pooling",
    pool_test,
    [a],
    out,
    [SIZE], 
    threadsperblock=Coord(TPB, 1), 
    blockspergrid=Coord(1, 1),  
    spec=pool_spec, 
)
# 显示谜题的相关信息。
problem.show()
```

```
# 池化
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查 GPU 内核执行的结果是否与 pool_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [0. 0. 0. 0. 0. 0. 0. 0.]
期望结果: [ 0.  1.  3.  6.  9. 12. 15. 18.]
```

## 谜题 10 - 点积 (Dot Product)

实现一个内核，计算 `a` 和 `b` 的点积，并将结果存储在 `out` 中。每个位置分配一个线程。每个线程只需要 2 次全局读取和 1 次全局写入。

*注意：对于这个问题，你不需要担心共享读取的次数。我们稍后会处理这个挑战。*

```python
# 定义 dot_spec 函数，用于生成期望的点积结果 (在 CPU 上执行)。
# 它接收两个 numpy 数组 a 和 b 作为输入。
def dot_spec(a, b):
    # 使用 Numpy 的 @ 运算符计算向量 a 和 b 的点积。
    # 等价于 np.dot(a,b) 或 (a * b).sum()。
    return a @ b 

# 定义每个线程块的线程数。
TPB = 8 # 在此谜题中，TPB 设置为等于数组大小 SIZE。

# 定义 dot_test 函数，该函数返回一个实现点积计算的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def dot_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组 (大小为1，存储最终的点积结果)。
    # a: 第一个输入向量。
    # b: 第二个输入向量。
    # size: 输入向量的实际大小。
    def call(out, a, b, size) -> None:
        # 在共享内存中创建一个一维数组 `shared`，大小为 TPB (即向量大小)。
        # 用于存储 a[i] * b[i] 的部分乘积，然后进行并行归约求和。
        shared = cuda.shared.array(TPB, numba.float32)

        # 计算当前线程的全局索引 i (在此设置下，也等于其局部索引 local_i)。
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # 获取当前线程在其所属线程块内的局部索引 local_i。
        local_i = cuda.threadIdx.x
        
        # 阶段 1: 计算部分乘积并存入共享内存。
        # 每个线程 local_i 计算 a[i] * b[i] 并存入 shared[local_i]。
        if i < size: # 边界检查 (虽然在此特定设置下 i 总会小于 size)
            shared[local_i] = a[i] * b[i]
        else:
            # 对于超出范围的线程 (理论上不应发生在此设置中)，
            # 在共享内存中填充 0 (不影响后续求和)。
            shared[local_i] = 0.0

        # 同步块内所有线程，确保所有部分乘积已计算完毕并存入共享内存。
        cuda.syncthreads()

        # 阶段 2: 在共享内存中执行并行归约求和 (reduction sum)。
        # 算法思想：每次迭代，将数组的后一半加到前一半。
        # s 是步长，初始为 TPB/2，每次迭代减半。
        s = TPB // 2 
        while s > 0:
            # 只有块内的前 s 个线程参与累加操作。
            # 线程 local_i 将 shared[local_i + s] 的值累加到 shared[local_i]。
            if local_i < s:
                shared[local_i] += shared[local_i + s]
            
            # 每次迭代后同步，确保当前轮次的累加操作完成，
            # 然后才能进行下一轮次步长更小的累加。
            cuda.syncthreads()
            s //= 2 # 步长减半
            
        # 阶段 3: 第一个线程 (local_i == 0) 将最终的求和结果 (存储在 shared[0]) 写入输出 out[0]。
        # 由于只有一个块 (blockspergrid=Coord(1, 1))，所以块内的和就是最终的点积。
        if local_i == 0:
            out[0] = shared[0]
    # 返回定义的内核函数。
    return call


# 定义输入向量的大小。
SIZE = 8
# 初始化输出数组 out，大小为 1，元素为 0 (点积结果是标量)。
out = np.zeros(1)
# 初始化第一个输入向量 a，包含从 0 到 SIZE-1 的整数序列。
a = np.arange(SIZE)
# 初始化第二个输入向量 b，同样包含从 0 到 SIZE-1 的整数序列。
b = np.arange(SIZE)

# 创建 CudaProblem 实例，用于配置和运行 "Dot" 谜题的测试。
# "Dot": 谜题的名称。
# dot_test: 用于生成 CUDA 内核的函数。
# [a, b]: 传递给 CUDA 内核的输入参数列表。
# out: 传递给 CUDA 内核的输出参数。
# [SIZE]: 传递给 CUDA 内核的额外参数 (向量的实际大小)。
# threadsperblock=Coord(SIZE, 1): 每个块的线程数等于向量大小 (即 TPB=8)。
# blockspergrid=Coord(1, 1): 网格中只有 1 个块。
#   (这意味着所有计算都在一个块内完成)
# spec=dot_spec: 用于生成期望结果的参考函数。
problem = CudaProblem(
    "Dot",
    dot_test,
    [a, b],
    out,
    [SIZE], 
    threadsperblock=Coord(SIZE, 1), 
    blockspergrid=Coord(1, 1),  
    spec=dot_spec, 
)
# 显示谜题的相关信息。
problem.show()
```

```
# 点积
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查 GPU 内核执行的结果是否与 dot_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [0.]
期望结果: 140
```

## 谜题 11 - 一维卷积 (1D Convolution)

实现一个内核，计算 `a` 和 `b` 之间的一维卷积，并将结果存储在 `out` 中。你需要处理一般情况。每个线程只需要 2 次全局读取和 1 次全局写入。

```python
# 定义 conv_spec 函数，用于生成期望的一维卷积结果 (在 CPU 上执行)。
# a: 输入信号数组。
# b: 卷积核数组。
def conv_spec(a, b):
    # 初始化输出数组 out，形状与输入信号 a 相同，所有元素填充为 0。
    out = np.zeros(*a.shape)
    # 获取卷积核 b 的长度。
    kernel_len = b.shape[0]
    # 遍历输入信号 a 的每个元素索引 i，对应输出 out[i] 的计算。
    for i in range(a.shape[0]):
        # 计算 out[i] 的值，它是 a 的一个子段与翻转的 b (标准卷积定义) 的点积。
        # 或者，按照很多库的实现方式 (互相关)，不翻转 b。
        # 这里的实现方式是：out[i] = sum(a[i+j] * b[j] for j if i+j is valid)
        # 这种方式通常称为“互相关”，但在深度学习中常直接称为“卷积”。
        # 遍历卷积核 b 的每个元素索引 j。
        current_sum = 0.0
        for j in range(kernel_len):
            # 确保 a 的索引 (i + j) 不越界。
            if i + j < a.shape[0]:
                current_sum += a[i + j] * b[j]
        out[i] = current_sum
        # 使用列表推导式可以更简洁地表达上述循环：
        # out[i] = sum([a[i + j] * b[j] for j in range(kernel_len) if i + j < a.shape[0]])
    # 返回卷积结果数组。
    return out


# 定义卷积核的最大长度，用于声明共享内存 (Numba要求共享内存大小为编译时常量)。
MAX_CONV = 4
# 定义每个线程块的线程数。
TPB = 8

# 定义 conv_test 函数，该函数返回一个实现一维卷积的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def conv_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组。
    # a: 输入信号数组。
    # b: 卷积核数组。
    # a_size: 输入信号 a 的实际大小。
    # b_size: 卷积核 b 的实际大小。
    def call(out, a, b, a_size, b_size) -> None:
        # 在共享内存中为输入信号 a 的一个片段声明数组。
        # 大小为 TPB + MAX_CONV - 1，以确保在计算卷积时，
        # 块内所有线程都能在共享内存中找到所需的 a 的元素。
        # 例如，计算块内最后一个输出元素时，可能需要访问超出当前块TPB范围的a元素。
        SHARED_MEM_SIZE_A = TPB + MAX_CONV - 1 
        shared_a = cuda.shared.array(SHARED_MEM_SIZE_A, numba.float32)
        
        # 在共享内存中为卷积核 b 声明数组。
        # 假设卷积核大小不超过 MAX_CONV。
        shared_b = cuda.shared.array(MAX_CONV, numba.float32)

        # 获取当前线程在其所属线程块内的局部索引 local_i。
        local_i = cuda.threadIdx.x
        # 计算当前块处理的输入信号 a 的起始全局索引。
        block_start_a_idx = cuda.blockIdx.x * cuda.blockDim.x # cuda.blockDim.x is TPB

        # 阶段 1: 加载数据到共享内存。
        # 1a: 加载卷积核 b 到 shared_b。
        #     由块内的前 b_size 个线程协作完成。
        if local_i < b_size:
            shared_b[local_i] = b[local_i]
        
        # 1b: 加载输入信号 a 的相关片段到 shared_a。
        #     每个线程 local_i 负责加载 shared_a[local_i]。
        #     还需要加载额外的元素以覆盖卷积窗口的末端。
        #     线程 local_i 加载 a[block_start_a_idx + local_i]。
        idx_to_load_a = block_start_a_idx + local_i
        if idx_to_load_a < a_size:
            shared_a[local_i] = a[idx_to_load_a]
        else:
            shared_a[local_i] = 0.0 # 越界部分用0填充

        # 如果 TPB 小于 SHARED_MEM_SIZE_A，需要更多线程或迭代来填充 shared_a 的剩余部分。
        # 假设每个线程只加载一个元素到 shared_a[local_i]。
        # 为了填充 shared_a 的末尾部分 (用于卷积窗口)，
        # 一些线程可能需要加载第二个元素。
        # 例如，线程 local_i 还可以负责加载 shared_a[local_i + TPB] (如果 local_i + TPB < SHARED_MEM_SIZE_A)
        if local_i + TPB < SHARED_MEM_SIZE_A:
            idx_to_load_a_extended = block_start_a_idx + local_i + TPB
            if idx_to_load_a_extended < a_size:
                shared_a[local_i + TPB] = a[idx_to_load_a_extended]
            else:
                shared_a[local_i + TPB] = 0.0
        
        # 同步块内所有线程，确保共享内存加载完毕。
        cuda.syncthreads()

        # 阶段 2: 执行卷积计算。
        # 计算当前线程负责的输出元素 out[global_out_idx]。
        global_out_idx = block_start_a_idx + local_i

        if global_out_idx < a_size: # 确保输出索引在有效范围内
            current_sum = 0.0
            # 遍历卷积核 b (存储在 shared_b 中)。
            for k_conv in range(b_size):
                # 输入信号 a 中对应的元素是 a[global_out_idx + k_conv]。
                # 该元素在 shared_a 中的索引是 local_i + k_conv。
                # (因为 shared_a[0] 对应 a[block_start_a_idx]，
                #  shared_a[local_i] 对应 a[block_start_a_idx + local_i] = a[global_out_idx])
                val_a_from_shared = shared_a[local_i + k_conv]
                val_b_from_shared = shared_b[k_conv]
                current_sum += val_a_from_shared * val_b_from_shared
            
            # 将计算得到的卷积和写入输出数组。
            out[global_out_idx] = current_sum
    # 返回定义的内核函数。
    return call


# 测试 1
SIZE_A1 = 6 # 输入信号 a 的大小
SIZE_B1 = 3 # 卷积核 b 的大小
out1 = np.zeros(SIZE_A1) # 初始化输出数组
a1 = np.arange(SIZE_A1)   # 初始化输入信号 a
b1 = np.arange(SIZE_B1)   # 初始化卷积核 b

problem1 = CudaProblem(
    "1D Conv (Simple)",          # 谜题名称
    conv_test,                   # 用于生成 CUDA 内核的函数
    [a1, b1],                    # 传递给内核的输入参数列表 [a, b]
    out1,                        # 传递给内核的输出参数
    [SIZE_A1, SIZE_B1],          # 额外参数 [a_size, b_size]
    blockspergrid=Coord((SIZE_A1 + TPB - 1) // TPB, 1), # 计算所需块数
                                                        # (6+8-1)//8 = 13//8 = 1 (取整，应为向上取整)
                                                        # blockspergrid应该是 Coord(1,1) for this case
    threadsperblock=Coord(TPB, 1),# 每个块的线程数 (TPB=8)
    spec=conv_spec,              # 用于生成期望结果的参考函数
)
problem1.show()
```

```
# 1D Conv (Simple)
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查第一个测试用例的结果。
problem1.check()
```

```
测试失败。
你的结果: [0. 0. 0. 0. 0. 0.]
期望结果: [ 5.  8. 11. 14.  5.  0.]
```

测试 2

```python
SIZE_A2 = 15 # 输入信号 a 的大小
SIZE_B2 = 4  # 卷积核 b 的大小
out2 = np.zeros(SIZE_A2) # 初始化输出数组
a2 = np.arange(SIZE_A2)    # 初始化输入信号 a
b2 = np.arange(SIZE_B2)    # 初始化卷积核 b

problem2 = CudaProblem(
    "1D Conv (Full)",            # 谜题名称
    conv_test,                   # 用于生成 CUDA 内核的函数
    [a2, b2],                    # 输入参数 [a, b]
    out2,                        # 输出参数
    [SIZE_A2, SIZE_B2],          # 额外参数 [a_size, b_size]
    blockspergrid=Coord((SIZE_A2 + TPB - 1) // TPB, 1), # (15+8-1)//8 = 22//8 = 2 (向上取整应为2)
                                                        # 这里应该是 Coord(2,1)
    threadsperblock=Coord(TPB, 1), # 每个块的线程数 (TPB=8)
    spec=conv_spec,              # 参考函数
)
problem2.show()
```

```
# 1D Conv (Full)
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查第二个测试用例的结果。
problem2.check()
```

```
测试失败。
你的结果: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
期望结果: [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
```

## 谜题 12 - 前缀和 (Prefix Sum)

实现一个内核，计算 `a` 上的和，并将结果存储在 `out` 中。如果 `a` 的大小大于块大小，则仅存储每个块的和。

我们将使用共享内存中的[并行前缀和](https://www.google.com/search?q=https://zh.wikipedia.org/wiki/%E5%89%8D%E7%BC%80%E5%92%8C)算法来完成此操作。也就是说，算法的每一步都应该将剩余数字的一半相加。请遵循此图：

(注意：上图展示的是前缀和 (scan) 操作，而题目描述的是块内求和 (reduction)。这里实现的将是块内求和。)

```python
# 定义每个线程块的线程数。
TPB = 8

# 定义 sum_spec 函数，用于生成期望的块和结果 (在 CPU 上执行)。
# a: 输入数组。
def sum_spec(a):
    # 计算输出数组的大小，即块的数量。
    # (a.shape[0] + TPB - 1) // TPB 是标准的向上取整除法，计算需要多少个大小为 TPB 的块来覆盖 a。
    num_blocks = (a.shape[0] + TPB - 1) // TPB
    # 初始化输出数组 out，用于存储每个块的和。
    out = np.zeros(num_blocks)
    # 遍历每个块。
    # j 是块的索引 (对应 out 的索引)。
    # i 是当前块在输入数组 a 中的起始索引。
    for j, i in enumerate(range(0, a.shape[0], TPB)): # 使用 a.shape[0] 更通用
        # 计算当前块 (从 a[i] 到 a[i+TPB-1]) 内所有元素的和。
        out[j] = a[i : i + TPB].sum()
    # 返回包含各块和的数组。
    return out


# 定义 sum_test 函数，该函数返回一个实现并行块内求和的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def sum_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出数组，存储每个块的和。
    # a: 输入数组。
    # size: 输入数组 a 的实际大小。
    def call(out, a, size: int) -> None:
        # 在共享内存中创建一个一维数组 `cache`，大小为 TPB。
        # 用于块内并行归约求和。
        cache = cuda.shared.array(TPB, numba.float32)
        
        # 计算当前线程的全局索引 i。
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # 获取当前线程在其所属线程块内的局部索引 local_i。
        local_i = cuda.threadIdx.x
        
        # 阶段 1: 从全局内存加载数据到共享内存。
        # 线程 local_i 负责加载 a[i] 到 cache[local_i]。
        if i < size: # 边界检查
            cache[local_i] = a[i]
        else:
            # 对于超出数组范围的线程，在共享内存中填充 0 (不影响求和)。
            cache[local_i] = 0.0
        
        # 同步块内所有线程，确保所有数据已加载到共享内存。
        cuda.syncthreads()

        # 阶段 2: 在共享内存中执行并行归约求和。
        # s 是归约的步长 (或偏移量)。
        s = TPB // 2
        while s > 0:
            # 只有块内的前 s 个线程参与当前的累加操作。
            # 线程 local_i 将 cache[local_i + s] 的值累加到 cache[local_i]。
            if local_i < s:
                cache[local_i] += cache[local_i + s]
            # 每次迭代后同步，确保当前轮次的累加操作完成。
            cuda.syncthreads()
            s //= 2 # 步长减半
            
        # 阶段 3: 第一个线程 (local_i == 0) 将当前块的和 (存储在 cache[0]) 写入输出数组 out。
        # 输出数组 out 的索引对应于当前块的索引 cuda.blockIdx.x。
        if local_i == 0:
            out[cuda.blockIdx.x] = cache[0]
    # 返回定义的内核函数。
    return call


# 测试 1 (单个块的情况)
SIZE1 = 8 # 输入数组大小
# 输出数组大小为1，因为只有一个块 (SIZE1 <= TPB)
out1 = np.zeros(1) 
inp1 = np.arange(SIZE1) # 输入数组 [0, 1, ..., 7]

problem1 = CudaProblem(
    "Sum (Simple)",             # 谜题名称
    sum_test,                   # 用于生成 CUDA 内核的函数
    [inp1],                     # 输入参数列表
    out1,                       # 输出参数
    [SIZE1],                    # 额外参数 (数组大小)
    blockspergrid=Coord(1, 1),  # 网格中只有1个块
                                # (SIZE1 + TPB - 1) // TPB = (8+8-1)//8 = 1
    threadsperblock=Coord(TPB, 1),# 每个块的线程数 (TPB=8)
    spec=sum_spec,              # 参考函数
)
problem1.show()
```

```
# Sum (Simple)
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查第一个测试用例的结果。
problem1.check()
```

```
测试失败。
你的结果: [0.]
期望结果: [28.]
```

测试 2 (多个块的情况)

```python
SIZE2 = 15 # 输入数组大小
# 输出数组大小为块的数量: (15 + 8 - 1) // 8 = 22 // 8 = 2 (整数除法)
# 向上取整应该是 ceil(15/8) = 2。
num_blocks2 = (SIZE2 + TPB - 1) // TPB
out2 = np.zeros(num_blocks2) 
inp2 = np.arange(SIZE2) # 输入数组 [0, 1, ..., 14]

problem2 = CudaProblem(
    "Sum (Full)",               # 谜题名称
    sum_test,                   # 用于生成 CUDA 内核的函数
    [inp2],                     # 输入参数列表
    out2,                       # 输出参数
    [SIZE2],                    # 额外参数 (数组大小)
    blockspergrid=Coord(num_blocks2, 1), # 网格中的块数 Coord(2, 1)
    threadsperblock=Coord(TPB, 1), # 每个块的线程数 (TPB=8)
    spec=sum_spec,              # 参考函数
)
problem2.show()
```

```
# Sum (Full)
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查第二个测试用例的结果。
problem2.check()
```

```
测试失败。
你的结果: [0. 0.]
期望结果: [28. 77.]
```

## 谜题 13 - 轴和 (Axis Sum)

实现一个内核，计算 `a` 的每一列的和，并将结果存储在 `out` 中。
(根据 `sum_spec` 的实现 `out[batch_idx, j_block] = a[batch_idx, i_start_col : i_start_col + TPB].sum()`，这实际上是计算二维数组 `a` 的每一**行**的**块和**。`a` 的形状是 `(BATCH, SIZE_of_row)`，`out` 的形状是 `(BATCH, num_blocks_per_row)`。因此，这是对每一行独立进行块内求和。)

```python
# 定义每个线程块在 x 方向的线程数 (用于处理行内数据)。
TPB = 8

# 定义 sum_spec 函数，用于生成期望的轴和结果 (在 CPU 上执行)。
# a: 输入的二维数组，形状为 (BATCH, SIZE_ax_1)，BATCH是批次数，SIZE_ax_1是每行元素数。
def sum_spec(a):
    # 计算每行需要多少个大小为 TPB 的块来覆盖其所有元素。
    num_blocks_per_row = (a.shape[1] + TPB - 1) // TPB
    # 初始化输出数组 out，形状为 (BATCH, num_blocks_per_row)。
    # out[r, c] 将存储第 r 行的第 c 个数据块的和。
    out = np.zeros((a.shape[0], num_blocks_per_row))
    
    # 遍历输入的每一行 (batch_idx 对应批次中的行)。
    for batch_idx in range(a.shape[0]):
        # 遍历当前行中的每个数据块。
        # j_block 是当前行内块的索引。
        # i_start_col 是当前块在行内的起始列索引。
        for j_block, i_start_col in enumerate(range(0, a.shape[1], TPB)):
            # 计算当前块 a[batch_idx, i_start_col : i_start_col + TPB] 内所有元素的和。
            out[batch_idx, j_block] = a[batch_idx, i_start_col : i_start_col + TPB].sum()
    # 返回包含各行各块和的二维数组。
    return out


# 定义 axis_sum_test 函数，该函数返回一个实现二维数组行内块求和的 CUDA 内核。
# cuda 参数由 CudaProblem 框架传入。
def axis_sum_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出二维数组，存储每行各块的和。
    # a: 输入的二维数组。
    # size_ax1: 输入数组 a 的第二维的大小 (即每行的元素数量)。
    def call(out, a, size_ax1: int) -> None:
        # 在共享内存中创建一个一维数组 `cache`，大小为 TPB。
        cache = cuda.shared.array(TPB, numba.float32)
        
        # CUDA 线程和块索引的映射：
        # cuda.blockIdx.y: 对应输入的行号 (batch index)。
        # cuda.blockIdx.x: 对应当前行内的块号。
        # cuda.threadIdx.x: 对应当前块内线程的局部索引 (处理行内数据)。

        # 当前处理的行号 (批次索引)。
        current_row_batch_idx = cuda.blockIdx.y
        # 当前线程在“行内块”中的局部索引 (0 to TPB-1)。
        local_col_idx_in_block = cuda.threadIdx.x
        
        # 当前线程负责处理的输入元素 a[current_row_batch_idx, global_col_idx_in_row] 的全局列索引。
        # global_col_idx_in_row = (行内块号 * 块内线程数) + 块内线程局部列索引
        global_col_idx_in_row = cuda.blockIdx.x * cuda.blockDim.x + local_col_idx_in_block
        
        # 阶段 1: 从全局内存加载数据到共享内存。
        # 线程 local_col_idx_in_block 负责加载 a[current_row_batch_idx, global_col_idx_in_row] 
        # 到 cache[local_col_idx_in_block]。
        if global_col_idx_in_row < size_ax1: # 边界检查，确保列索引在行范围内
            cache[local_col_idx_in_block] = a[current_row_batch_idx, global_col_idx_in_row]
        else:
            cache[local_col_idx_in_block] = 0.0 # 超出范围则填充0
            
        # 同步块内所有线程 (处理同一行内同一块的线程)。
        cuda.syncthreads()

        # 阶段 2: 在共享内存中执行并行归约求和 (逻辑与 Puzzle 12 相同)。
        s = TPB // 2
        while s > 0:
            if local_col_idx_in_block < s:
                cache[local_col_idx_in_block] += cache[local_col_idx_in_block + s]
            cuda.syncthreads()
            s //= 2
            
        # 阶段 3: 第一个线程 (local_col_idx_in_block == 0) 将当前块的和写入输出数组 out。
        # out[current_row_batch_idx, cuda.blockIdx.x] 存储当前行 (batch) 的当前块 (blockIdx.x) 的和。
        if local_col_idx_in_block == 0:
            out[current_row_batch_idx, cuda.blockIdx.x] = cache[0]
    # 返回定义的内核函数。
    return call


# 定义批次大小 (行数) 和每行内的元素数量 (列数)。
BATCH = 4       # a.shape[0]
SIZE_AX1 = 6    # a.shape[1]

# 计算输出数组 out 的形状: (BATCH, num_blocks_per_row)
# num_blocks_per_row = (SIZE_AX1 + TPB - 1) // TPB
# (6 + 8 - 1) // 8 = 13 // 8 = 1 (因为 TPB=8, SIZE_AX1=6，每行只需要1个块)
num_blocks_per_row = (SIZE_AX1 + TPB - 1) // TPB
out = np.zeros((BATCH, num_blocks_per_row)) 
# 初始化输入二维数组 inp。
inp = np.arange(BATCH * SIZE_AX1).reshape((BATCH, SIZE_AX1))

problem = CudaProblem(
    "Axis Sum",                 # 谜题名称
    axis_sum_test,              # 用于生成 CUDA 内核的函数
    [inp],                      # 输入参数列表
    out,                        # 输出参数
    [SIZE_AX1],                 # 额外参数 (每行的元素数量)
    # blockspergrid: 
    #   x 方向是每行的块数 (num_blocks_per_row)
    #   y 方向是行数 (BATCH)
    blockspergrid=Coord(num_blocks_per_row, BATCH), # Coord(1, 4)
    # threadsperblock:
    #   x 方向是处理行内数据的线程数 (TPB)
    #   y 方向是 1 (因为每个线程块只处理一行的一部分)
    threadsperblock=Coord(TPB, 1), # Coord(8, 1)
    spec=sum_spec,              # 参考函数
)
problem.show()
```

```
# Axis Sum
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查 GPU 内核执行的结果是否与 sum_spec 函数生成的期望结果一致。
problem.check()
```

```
测试失败。
你的结果: [[0.]
  [0.]
  [0.]
  [0.]]
期望结果: [[ 15.]
  [ 51.]
  [ 87.]
  [123.]]
```

## 谜题 14 - 矩阵乘法\! (Matrix Multiply\!)

实现一个内核，将方阵 `a` 和 `b` 相乘，并将结果存储在 `out` 中。

*提示：这里最有效的算法是将一个块复制到共享内存中，然后再计算每个单独的行-列点积。如果矩阵适合共享内存，这很容易做到。先做那种情况。然后更新你的代码以计算部分点积，并迭代地移动你复制到共享内存的部分。* 你应该能够以 6 次全局读取完成困难情况。

```python
# 定义 matmul_spec 函数，用于生成期望的矩阵乘法结果 (在 CPU 上执行)。
# a, b: 输入的两个方阵。
def matmul_spec(a, b):
    # 使用 Numpy 的 @ 运算符计算矩阵 a 和 b 的乘积。
    return a @ b 

# 定义每个线程块在一个维度上的线程数。
# 对于矩阵乘法，通常使用 TPB x TPB 的线程块。
TPB = 3 

# 定义 mm_oneblock_test 函数 (在原始代码中用于单块和多块测试)，
# 该函数返回一个实现分块矩阵乘法的 CUDA 内核。
def mm_oneblock_test(cuda):
    # 定义实际在 GPU 上执行的 CUDA 内核函数 call。
    # out: 输出矩阵 C (C = A * B)。
    # a: 输入矩阵 A。
    # b: 输入矩阵 B。
    # size: 输入方阵的维度 N (NxN)。
    def call(out, a, b, size: int) -> None:
        # 在共享内存中为矩阵 A 的子块 (tile) 声明数组。大小为 TPB x TPB。
        a_shared = cuda.shared.array((TPB, TPB), numba.float32)
        # 在共享内存中为矩阵 B 的子块 (tile) 声明数组。大小为 TPB x TPB。
        b_shared = cuda.shared.array((TPB, TPB), numba.float32)

        # CUDA 线程和块索引的映射：
        # cuda.threadIdx.y (local_row_idx): 线程在块内的局部行索引。
        # cuda.threadIdx.x (local_col_idx): 线程在块内的局部列索引。
        # cuda.blockIdx.y: 线程块在网格中的行索引。
        # cuda.blockIdx.x: 线程块在网格中的列索引。
        
        local_row_idx = cuda.threadIdx.y # 当前线程在块内的局部行索引 (0 to TPB-1)
        local_col_idx = cuda.threadIdx.x # 当前线程在块内的局部列索引 (0 to TPB-1)
        
        # 计算当前线程负责计算的输出矩阵 out 中的元素的全局行和列索引。
        # global_row = (块的行索引 * 块的行维度) + 线程在块内的行索引
        # global_col = (块的列索引 * 块的列维度) + 线程在块内的列索引
        global_row = cuda.blockIdx.y * TPB + local_row_idx
        global_col = cuda.blockIdx.x * TPB + local_col_idx

        # 初始化累加器，用于存储 out[global_row, global_col] 的部分和。
        acc_val = 0.0
        
        # 计算需要处理的子块 (tile) 的数量。
        # 矩阵 A 的列数 (或 B 的行数) 被划分为多少个 TPB 大小的块。
        num_tiles = (size + TPB - 1) // TPB

        # 遍历所有子块 (tiles)。
        for tile_idx in range(num_tiles):
            # 阶段 1: 从全局内存加载 A 和 B 的当前子块到共享内存。
            # 每个线程 (local_row_idx, local_col_idx) 负责加载 a_shared 和 b_shared 中的一个元素。
            
            # 加载 a_shared[local_row_idx, local_col_idx]
            # 它对应于全局矩阵 A 中的元素 a[global_row, tile_idx * TPB + local_col_idx]
            # 注意：每个线程加载 a_shared[local_row_idx, local_col_idx]
            # 对应的全局A元素应该是 a[global_row_for_a_tile, col_in_a_tile]
            # global_row_for_a_tile = global_row (因为a_shared的行对应out的行)
            # col_in_a_tile = tile_idx * TPB + local_col_idx (a_shared的列对应A的当前tile的列)
            a_load_row = global_row
            a_load_col = tile_idx * TPB + local_col_idx
            if a_load_row < size and a_load_col < size: # 边界检查
                a_shared[local_row_idx, local_col_idx] = a[a_load_row, a_load_col]
            else:
                a_shared[local_row_idx, local_col_idx] = 0.0 # 越界填充
            
            # 加载 b_shared[local_row_idx, local_col_idx]
            # 它对应于全局矩阵 B 中的元素 b[tile_idx * TPB + local_row_idx, global_col]
            # 注意：每个线程加载 b_shared[local_row_idx, local_col_idx]
            # 对应的全局B元素应该是 b[row_in_b_tile, global_col_for_b_tile]
            # row_in_b_tile = tile_idx * TPB + local_row_idx (b_shared的行对应B的当前tile的行)
            # global_col_for_b_tile = global_col (b_shared的列对应out的列)
            b_load_row = tile_idx * TPB + local_row_idx
            b_load_col = global_col
            if b_load_row < size and b_load_col < size: # 边界检查
                b_shared[local_row_idx, local_col_idx] = b[b_load_row, b_load_col]
            else:
                b_shared[local_row_idx, local_col_idx] = 0.0 # 越界填充

            # 同步块内所有线程，确保 A 和 B 的子块已完全加载到共享内存。
            cuda.syncthreads()
            
            # 阶段 2: 计算当前子块的点积，并累加到 acc_val。
            # out[global_row, global_col] += sum_{k=0}^{TPB-1} (a_shared[local_row_idx, k] * b_shared[k, local_col_idx])
            for k in range(TPB):
                acc_val += a_shared[local_row_idx, k] * b_shared[k, local_col_idx]
            
            # 再次同步块内所有线程。这非常重要！
            # 确保在进入下一个 tile_idx 迭代 (加载新的子块) 之前，
            # 当前子块的所有计算 (读取共享内存) 都已完成。
            cuda.syncthreads()
            
        # 阶段 3: 所有子块处理完毕后，将最终计算得到的累加值写入输出矩阵 out。
        if global_row < size and global_col < size: # 边界检查，确保不越界写入
            out[global_row, global_col] = acc_val
    # 返回定义的内核函数。
    return call

# 测试 1 (小型矩阵)
SIZE1 = 2 # 矩阵维度
out1 = np.zeros((SIZE1, SIZE1)) # 初始化输出矩阵
inp1_a = np.arange(SIZE1 * SIZE1).reshape((SIZE1, SIZE1)) # 输入矩阵 A
# 题目原始代码中 inp1_b 被转置了。
# inp1_b = np.arange(SIZE1 * SIZE1).reshape((SIZE1, SIZE1)).T
# 如果 spec 是 a @ b，那么内核应该计算 a @ b。
# CudaProblem 的 spec=matmul_spec，matmul_spec(a,b) 是 a@b。
# 所以，传递给内核的第二个参数应该是原始的矩阵 B。
# 我们假设 problem.check() 会使用 inp1_a @ inp1_b (其中 inp1_b 是原始B)。
# 因此，我们这里也用原始B。
inp1_b = np.arange(SIZE1 * SIZE1).reshape((SIZE1, SIZE1)) 
# Sasha Rush 的代码中 inp2 是 .T 的版本，并且 check 是正确的，说明内核设计是 C = A * B (B没有预转置)
# 让我们保持与原始谜题一致，即 inp2 = ...T
inp1_b_transposed = np.arange(SIZE1 * SIZE1).reshape((SIZE1, SIZE1)).T

problem1 = CudaProblem(
    "Matmul (Simple)",          # 谜题名称
    mm_oneblock_test,           # 用于生成 CUDA 内核的函数
    [inp1_a, inp1_b_transposed],# 输入参数 [A, B] (B 已在CPU转置，但内核仍按A*B处理)
                                # 或者，如果内核要实现 A*B^T，而B^T已传入，那么内核还是 A*B'
                                # 实际上，内核实现的是 C[i,j] = sum(A[i,k]*B[k,j])
                                # 传入的 inp1_b_transposed 就是矩阵 B。
    out1,                       # 输出参数
    [SIZE1],                    # 额外参数 (矩阵维度 N)
    # blockspergrid: 网格中的块数。每个维度需要 (SIZE + TPB - 1) // TPB 个块。
    blockspergrid=Coord((SIZE1 + TPB - 1) // TPB, (SIZE1 + TPB - 1) // TPB), 
                                # For SIZE1=2, TPB=3 -> (2+3-1)//3 = 4//3 = 1. So Coord(1,1).
    threadsperblock=Coord(TPB, TPB), # 每个块的线程数 TPB x TPB (3x3)
    spec=matmul_spec,           # 参考函数
)
problem1.show(sparse=True) # sparse=True 可能用于控制大型矩阵的显示方式
```

```
# Matmul (Simple)
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查第一个测试用例的结果。
problem1.check()
```

```
测试失败。
你的结果: [[0. 0.]
  [0. 0.]]
期望结果: [[ 1  3]
  [ 3 13]]
```

测试 2 (较大型矩阵)

```python
SIZE2 = 8 # 矩阵维度
out2 = np.zeros((SIZE2, SIZE2)) # 初始化输出矩阵
inp2_a = np.arange(SIZE2 * SIZE2).reshape((SIZE2, SIZE2))    # 输入矩阵 A
inp2_b_transposed = np.arange(SIZE2 * SIZE2).reshape((SIZE2, SIZE2)).T # 输入矩阵 B (与谜题保持一致)

problem2 = CudaProblem(
    "Matmul (Full)",            # 谜题名称
    mm_oneblock_test,           # 用于生成 CUDA 内核的函数
    [inp2_a, inp2_b_transposed],# 输入参数 [A, B]
    out2,                       # 输出参数
    [SIZE2],                    # 额外参数 (矩阵维度 N)
    blockspergrid=Coord((SIZE2 + TPB - 1) // TPB, (SIZE2 + TPB - 1) // TPB), 
                                # SIZE2=8, TPB=3. (8+3-1)//3 = 10//3 = 3 (整数除法). So Coord(3,3).
    threadsperblock=Coord(TPB, TPB), # 每个块的线程数 TPB x TPB (3x3)
    spec=matmul_spec,           # 参考函数
)
problem2.show(sparse=True)
```

```
# Matmul (Full)
# 每个线程的最大得分:
# | 全局读取 | 全局写入 | 共享读取 | 共享写入 |
# | 0 | 0 | 0 | 0 | 
```

```python
# 检查第二个测试用例的结果。
problem2.check()
```

```
测试失败。
你的结果: [[0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0.]]
期望结果: [[  140   364   588   812  1036  1260  1484  1708]
  [  364  1100  1836  2572  3308  4044  4780  5516]
  [  588  1836  3084  4332  5580  6828  8076  9324]
  [  812  2572  4332  6092  7852  9612 11372 13132]
  [ 1036  3308  5580  7852 10124 12396 14668 16940]
  [ 1260  4044  6828  9612 12396 15180 17964 20748]
  [ 1484  4780  8076 11372 14668 17964 21260 24556]
  [ 1708  5516  9324 13132 16940 20748 24556 28364]]
```

