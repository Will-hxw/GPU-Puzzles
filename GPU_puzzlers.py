# # GPU 习题集
# - 作者：[Sasha Rush](http://rush-nlp.com) - [srush_nlp](https://twitter.com/srush_nlp)

# ![](https://github.com/srush/GPU-Puzzles/raw/main/cuda.png)

# GPU 架构对于机器学习至关重要，并且似乎变得越来越重要。然而，即使你是一位机器学习专家，也可能从未接触过 GPU 编程。通过抽象层工作很难获得直觉。

# 这个笔记本尝试以完全交互式的方式教授初学者 GPU 编程。与其提供带有概念的文本，不如直接让你进入编码和构建 GPU 内核的过程。
# 这些练习使用 NUMBA，它直接将 Python 代码映射到 CUDA 内核。看起来像 Python，但实际上与编写低级 CUDA 代码基本相同。
# 只需几个小时，我相信你可以从基础到理解当今 99% 深度学习算法的真正实现。

# 如果你想阅读手册，可以参考以下链接：

# [NUMBA CUDA 指南](https://numba.readthedocs.io/en/stable/cuda/index.html)

# 我建议在 Colab 中完成这些练习，因为它易于启动。确保创建自己的副本，在设置中启用 GPU 模式（`运行时 / 更改运行时类型`，然后将 `硬件加速器` 设置为 `GPU`），然后开始编码。

# [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github/srush/GPU-Puzzles/blob/main/GPU_puzzlers.ipynb)

# （如果你喜欢这种谜题风格，还可以查看我的 [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) 用于 PyTorch。）

# !pip install -qqq git+https://github.com/danoneata/chalk@srush-patch-1
# !wget -q https://github.com/srush/GPU-Puzzles/raw/main/robot.png https://github.com/srush/GPU-Puzzles/raw/main/lib.py

import numba
import numpy as np
import warnings
from lib import CudaProblem, Coord

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)

# ## 练习 1: 映射
#
# 实现一个“内核”（GPU 函数），将向量 `a` 的每个位置加 10，并将其存储在向量 `out` 中。每个位置有一个线程。

# **警告** 这段代码看起来像 Python，但实际上它是 CUDA！你不能使用标准的 Python 工具（如列表推导式）或询问 Numpy 属性（如形状或大小）。如果需要大小，则作为参数给出。
# 这些练习只需要执行简单的操作，例如 +、*、简单的数组索引、for 循环和 if 语句。
# 你可以使用局部变量。
# 如果出现错误，那可能是因为你做了某些复杂的事情 :).

# *提示：可以将函数 `call` 视为被每个线程运行 1 次。唯一不同的是 `cuda.threadIdx.x` 每次都会改变。*

# +
def map_spec(a):
    return a + 10


def map_test(cuda):
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        # 在这里填写代码（大约 1 行）

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "映射", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec
)
problem.show()

# +
problem.check()
# -

# ## 练习 2 - 合并
#
# 实现一个内核，将 `a` 和 `b` 的每个位置相加，并将其存储在 `out` 中。每个位置有一个线程。

# +
def zip_spec(a, b):
    return a + b


def zip_test(cuda):
    def call(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        # 在这里填写代码（大约 1 行）

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem(
    "合并", zip_test, [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec
)
problem.show()
# +

# +
problem.check()
# -

# ## 练习 3 - 守卫
#
# 实现一个内核，将 `a` 的每个位置加 10，并将其存储在 `out` 中。线程数多于位置数。

# +
def map_guard_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        # 在这里填写代码（大约 2 行）

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "守卫",
    map_guard_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(8, 1),
    spec=map_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 4 - 二维映射
#
# 实现一个内核，将 `a` 的每个位置加 10，并将其存储在 `out` 中。输入 `a` 是二维且正方形的。线程数多于位置数。

# +
def map_2D_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # 在这里填写代码（大约 2 行）

    return call


SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
problem = CudaProblem(
    "二维映射", map_2D_test, [a], out, [SIZE], threadsperblock=Coord(3, 3), spec=map_spec
)
problem.show()

# +
problem.check()
# -

# ## 练习 5 - 广播
#
# 实现一个内核，将 `a` 和 `b` 相加，并将其存储在 `out` 中。输入 `a` 和 `b` 是向量。线程数多于位置数。

# +
def broadcast_test(cuda):
    def call(out, a, b, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # 在这里填写代码（大约 2 行）

    return call


SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE).reshape(SIZE, 1)
b = np.arange(SIZE).reshape(1, SIZE)
problem = CudaProblem(
    "广播",
    broadcast_test,
    [a, b],
    out,
    [SIZE],
    threadsperblock=Coord(3, 3),
    spec=zip_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 6 - 块
#
# 实现一个内核，将 `a` 的每个位置加 10，并将其存储在 `out` 中。每个块的线程数少于 `a` 的大小。

# *提示：块是一组线程。每个块的线程数有限，但我们可以有多个不同的块。变量 `cuda.blockIdx` 告诉我们当前所在的块。*

# +
def map_block_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # 在这里填写代码（大约 2 行）

    return call


SIZE = 9
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "块",
    map_block_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(4, 1),
    blockspergrid=Coord(3, 1),
    spec=map_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 7 - 二维块
#
# 在二维中实现相同的内核。每个块的线程数少于 `a` 的大小。

# +
def map_block2D_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # 在这里填写代码（大约 4 行）

    return call


SIZE = 5
out = np.zeros((SIZE, SIZE))
a = np.ones((SIZE, SIZE))

problem = CudaProblem(
    "二维块",
    map_block2D_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(3, 3),
    blockspergrid=Coord(2, 2),
    spec=map_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 8 - 共享内存
#
# 实现一个内核，将 `a` 的每个位置加 10，并将其存储在 `out` 中。每个块的线程数少于 `a` 的大小。

# **警告**：每个块只能拥有一个 *常量* 大小的共享内存，该内存可以被该块中的线程读写。这需要是一个字面的 Python 常量，而不是变量。
# 写入共享内存后，需要调用 `cuda.syncthreads` 以确保线程不会跨越。

# （这个例子实际上不需要共享内存或 syncthreads，但它是一个演示。）

# +
TPB = 4
def shared_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        if i < size:
            shared[local_i] = a[i]
            cuda.syncthreads()

        # 在这里填写代码（大约 2 行）

    return call


SIZE = 8
out = np.zeros(SIZE)
a = np.ones(SIZE)
problem = CudaProblem(
    "共享内存",
    shared_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(TPB, 1),
    blockspergrid=Coord(2, 1),
    spec=map_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 9 - 池化
#
# 实现一个内核，将 `a` 的最后 3 个位置相加，并将其存储在 `out` 中。每个位置有一个线程。每个线程只需要 1 次全局读取和 1 次全局写入。

# *提示：记得小心同步。*

# +
def pool_spec(a):
    out = np.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out


TPB = 8
def pool_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # 在这里填写代码（大约 8 行）

    return call


SIZE = 8
out = np.zeros(SIZE)
a = np.arange(SIZE)
problem = CudaProblem(
    "池化",
    pool_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(TPB, 1),
    blockspergrid=Coord(1, 1),
    spec=pool_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 10 - 点积
#
# 实现一个内核，计算 `a` 和 `b` 的点积，并将其存储在 `out` 中。每个位置有一个线程。每个线程只需要 2 次全局读取和 1 次全局写入。

# *注意：对于这个问题，你不需要担心共享内存读取的数量。我们稍后会处理这个挑战。*

# +
def dot_spec(a, b):
    return a @ b

TPB = 8
def dot_test(cuda):
    def call(out, a, b, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # 在这里填写代码（大约 9 行）
    return call


SIZE = 8
out = np.zeros(1)
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem(
    "点积",
    dot_test,
    [a, b],
    out,
    [SIZE],
    threadsperblock=Coord(SIZE, 1),
    blockspergrid=Coord(1, 1),
    spec=dot_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 11 - 一维卷积
#
# 实现一个内核，计算 `a` 和 `b` 之间的一维卷积，并将其存储在 `out` 中。需要处理通用情况。每个线程只需要 2 次全局读取和 1 次全局写入。

# +
def conv_spec(a, b):
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


MAX_CONV = 4
TPB = 8
TPB_MAX_CONV = TPB + MAX_CONV
def conv_test(cuda):
    def call(out, a, b, a_size, b_size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        # 在这里填写代码（大约 17 行）

    return call


# 测试 1

SIZE = 6
CONV = 3
out = np.zeros(SIZE)
a = np.arange(SIZE)
b = np.arange(CONV)
problem = CudaProblem(
    "一维卷积（简单）",
    conv_test,
    [a, b],
    out,
    [SIZE, CONV],
    Coord(1, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)
problem.show()

# +
problem.check()
# -

# 测试 2

# +
out = np.zeros(15)
a = np.arange(15)
b = np.arange(4)
problem = CudaProblem(
    "一维卷积（完整）",
    conv_test,
    [a, b],
    out,
    [15, 4],
    Coord(2, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)
problem.show()
# -

# +
problem.check()
# -

# ## 练习 12 - 前缀和
#
# 实现一个内核，计算 `a` 的总和并将其存储在 `out` 中。如果 `a` 的大小大于块大小，则仅存储每个块的总和。

# 我们将使用 [并行前缀和](https://en.wikipedia.org/wiki/Prefix_sum) 算法在共享内存中完成此操作。也就是说，算法的每一步应该将剩余数字的一半相加。遵循此图：

# ![](https://user-images.githubusercontent.com/35882/178757889-1c269623-93af-4a2e-a7e9-22cd55a42e38.png)

# +
TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0] + TPB - 1) // TPB)
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[j] = a[i : i + TPB].sum()
    return out


def sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # 在这里填写代码（大约 12 行）

    return call


# 测试 1

SIZE = 8
out = np.zeros(1)
inp = np.arange(SIZE)
problem = CudaProblem(
    "求和（简单）",
    sum_test,
    [inp],
    out,
    [SIZE],
    Coord(1, 1),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()

# +
problem.check()
# -

# 测试 2

# +
SIZE = 15
out = np.zeros(2)
inp = np.arange(SIZE)
problem = CudaProblem(
    "求和（完整）",
    sum_test,
    [inp],
    out,
    [SIZE],
    Coord(2, 1),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 13 - 轴求和
#
# 实现一个内核，计算 `a` 的每一列的总和并将其存储在 `out` 中。

# +
TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0], (a.shape[1] + TPB - 1) // TPB))
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[..., j] = a[..., i : i + TPB].sum(-1)
    return out


def axis_sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        batch = cuda.blockIdx.y
        # 在这里填写代码（大约 12 行）

    return call


BATCH = 4
SIZE = 6
out = np.zeros((BATCH, 1))
inp = np.arange(BATCH * SIZE).reshape((BATCH, SIZE))
problem = CudaProblem(
    "轴求和",
    axis_sum_test,
    [inp],
    out,
    [SIZE],
    Coord(1, BATCH),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()

# +
problem.check()
# -

# ## 练习 14 - 矩阵乘法！
#
# 实现一个内核，将方阵 `a` 和 `b` 相乘，并将结果存储在 `out` 中。

# *提示：这里最有效的算法是先将一个块复制到共享内存中，然后再计算每个单独的行-列点积。如果矩阵适合共享内存，这样做很容易。先完成这种情况。
# 然后更新你的代码以计算部分点积，并迭代地移动你复制到共享内存的部分。* 你应该能够在困难的情况下完成 6 次全局读取。

# +
def matmul_spec(a, b):
    return a @ b


TPB = 3
def mm_oneblock_test(cuda):
    def call(out, a, b, size: int) -> None:
        a_shared = cuda.shared.array((TPB, TPB), numba.float32)
        b_shared = cuda.shared.array((TPB, TPB), numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # 在这里填写代码（大约 14 行）

    return call

# 测试 1

SIZE = 2
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

problem = CudaProblem(
    "矩阵乘法（简单）",
    mm_oneblock_test,
    [inp1, inp2],
    out,
    [SIZE],
    Coord(1, 1),
    Coord(TPB, TPB),
    spec=matmul_spec,
)
problem.show(sparse=True)

# +
problem.check()
# -

# 测试 2

# +
SIZE = 8
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

problem = CudaProblem(
    "矩阵乘法（完整）",
    mm_oneblock_test,
    [inp1, inp2],
    out,
    [SIZE],
    Coord(3, 3),
    Coord(TPB, TPB),
    spec=matmul_spec,
)
problem.show(sparse=True)
# -

# +
problem.check()
# -