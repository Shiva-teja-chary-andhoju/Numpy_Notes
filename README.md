# NumPy Notes

- [Numpy Official Documentation & User Guide](https://numpy.org/)


## Table of Contents

1. [What is NumPy?](#1-what-is-numpy)
2. [Create Arrays](#2-create-arrays)
3. [Basic Array Understanding](#3-basic-array-understanding)
4. [Array Data Types](#4-array-data-types)
5. [Indexing & Slicing](#5-indexing--slicing)
6. [Boolean & Fancy Indexing](#6-boolean--fancy-indexing)
7. [Reshaping Arrays](#7-reshaping-arrays)
8. [Joining & Splitting Arrays](#8-joining--splitting-arrays)
9. [Arithmetic & Math Operations](#9-arithmetic--math-operations)
10. [Broadcasting](#10-broadcasting)
11. [Aggregation Functions](#11-aggregation-functions)
12. [Sorting & Searching](#12-sorting--searching)
13. [Random Numbers](#13-random-numbers)
14. [Linear Algebra](#14-linear-algebra)
15. [Copy vs View](#15-copy-vs-view)
16. [Iterating Arrays](#16-iterating-arrays)
17. [Working with Missing / Special Values (NaN, Inf)](#17-working-with-missing--special-values-nan-inf)
18. [Set Operations](#18-set-operations)
19. [Save and Load Arrays](#19-save-and-load-arrays)
20. [Vectorization & Performance](#20-vectorization--performance)

---

## 1. What is NumPy?

A Python library for fast numerical computing with multi-dimensional arrays. It's the foundation for pandas, scikit-learn, and most of the data science stack — much faster than plain Python lists because it stores data in contiguous memory and operates in compiled C code.

| Term | Description |
|---|---|
| `ndarray` | The core N-dimensional array object |
| `axis` | A dimension of the array (`axis=0` rows, `axis=1` columns) |
| `shape` | Tuple describing the size of each dimension |
| `dtype` | The data type of the array's elements (all elements share one dtype) |

```python
import numpy as np
np.__version__
```

---

## 2. Create Arrays

| Function | Description |
|---|---|
| `np.array([...])` | Create an array from a list/tuple |
| `np.zeros(shape)` | Array filled with zeros |
| `np.ones(shape)` | Array filled with ones |
| `np.full(shape, value)` | Array filled with a constant value |
| `np.arange(start, stop, step)` | Array of evenly spaced values (like `range`) |
| `np.linspace(start, stop, n)` | `n` evenly spaced values between start and stop (inclusive) |
| `np.eye(n)` | Identity matrix of size `n x n` |
| `np.empty(shape)` | Uninitialized array (garbage values, fastest to create) |

```python
arr = np.array([1, 2, 3])
mat = np.array([[1, 2], [3, 4]])

np.zeros((2, 3))
np.ones((2, 3))
np.full((2, 3), 7)

np.arange(0, 10, 2)          # 0,2,4,6,8
np.linspace(0, 1, 5)         # 5 values between 0 and 1

np.eye(3)
np.empty((2, 2))
```

---

## 3. Basic Array Understanding

| Function | Description |
|---|---|
| `.shape` | Tuple of dimension sizes |
| `.ndim` | Number of dimensions |
| `.size` | Total number of elements |
| `.dtype` | Data type of elements |
| `.itemsize` | Bytes per element |
| `.nbytes` | Total bytes used by the array |
| `len(arr)` | Size of the first axis |

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape     # (2, 3)
arr.ndim      # 2
arr.size      # 6
arr.dtype     # int64
arr.itemsize  # 8
arr.nbytes    # 48
len(arr)      # 2
```

---

## 4. Array Data Types

| Function | Description |
|---|---|
| `np.array(data, dtype=...)` | Set the dtype at creation |
| `.astype('type')` | Convert to a different dtype (returns new array) |
| Common dtypes | `int32`, `int64`, `float32`, `float64`, `bool`, `str_`, `complex128` |

```python
arr = np.array([1, 2, 3], dtype='float32')
arr2 = arr.astype('int32')

np.array([1, 0, 1], dtype='bool')
```

---

## 5. Indexing & Slicing

| Function | Description |
|---|---|
| `arr[i]` | Element at position `i` |
| `arr[i, j]` | Element at row `i`, column `j` (2-D) |
| `arr[start:stop:step]` | Slice (stop exclusive) |
| `arr[:, j]` | Entire column `j` |
| `arr[i, :]` | Entire row `i` |
| `arr[::-1]` | Reverse the array |

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr[0, 1]        # 2
arr[1]            # [4, 5, 6]
arr[:, 1]         # column -> [2, 5, 8]
arr[0:2, 0:2]     # sub-matrix
arr[::-1]         # reversed rows
arr[:, ::-1]      # reversed columns
```

---

## 6. Boolean & Fancy Indexing

| Function | Description |
|---|---|
| `arr[condition]` | Select elements where condition is True |
| `arr[arr > n]` | Boolean mask filtering |
| `arr[[i1, i2, i3]]` | Fancy indexing — select by list of positions |
| `np.where(condition, a, b)` | Element-wise if/else |
| `np.select(conditions, choices)` | Multiple conditions, like SQL `CASE WHEN` |

```python
arr = np.array([10, 20, 30, 40, 50])

arr[arr > 20]                     # [30, 40, 50]
arr[(arr > 10) & (arr < 50)]      # multiple conditions

arr[[0, 2, 4]]                    # fancy indexing -> [10, 30, 50]

np.where(arr > 25, 'High', 'Low')
np.select([arr < 20, arr < 40], ['Low', 'Mid'], default='High')
```

---

## 7. Reshaping Arrays

| Function | Description |
|---|---|
| `.reshape(shape)` | Change shape without changing data (returns a view when possible) |
| `.flatten()` | Collapse to 1-D (always returns a copy) |
| `.ravel()` | Collapse to 1-D (returns a view when possible, faster) |
| `.T` / `np.transpose()` | Transpose (swap axes) |
| `np.expand_dims(arr, axis)` | Add a new axis |
| `np.squeeze(arr)` | Remove axes of length 1 |

```python
arr = np.arange(12)

arr.reshape(3, 4)
arr.reshape(3, -1)     # -1 = infer this dimension

mat = arr.reshape(3, 4)
mat.flatten()
mat.ravel()
mat.T

np.expand_dims(arr, axis=0)   # shape (1, 12)
np.squeeze(np.array([[1, 2, 3]]))
```

---

## 8. Joining & Splitting Arrays

| Function | Description |
|---|---|
| `np.concatenate([a1, a2], axis=)` | Join arrays along an existing axis |
| `np.vstack([a1, a2])` | Stack vertically (row-wise) |
| `np.hstack([a1, a2])` | Stack horizontally (column-wise) |
| `np.stack([a1, a2], axis=)` | Join arrays along a new axis |
| `np.split(arr, n)` | Split into `n` equal parts |
| `np.array_split(arr, n)` | Split into `n` parts (allows uneven sizes) |

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.concatenate([a, b])
np.vstack([a, b])
np.hstack([a, b])
np.stack([a, b], axis=0)

np.split(np.arange(9), 3)
np.array_split(np.arange(10), 3)
```

---

## 9. Arithmetic & Math Operations

| Function | Description |
|---|---|
| `+ - * /` | Element-wise arithmetic |
| `arr ** 2` | Element-wise power |
| `np.sqrt(arr)` | Square root |
| `np.exp(arr)` | Exponential (e^x) |
| `np.log(arr)` / `np.log10(arr)` | Natural log / log base 10 |
| `np.round(arr, n)` | Round to `n` decimals |
| `np.abs(arr)` | Absolute value |
| `np.dot(a, b)` / `a @ b` | Dot product / matrix multiplication |

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b
a * b            # element-wise multiply
a ** 2
np.sqrt(a)
np.exp(a)
np.log(a)
np.round(3.14159, 2)
np.abs(np.array([-1, -2, 3]))

np.dot(a, b)     # scalar for 1-D arrays
mat_a = np.array([[1, 2], [3, 4]])
mat_b = np.array([[5, 6], [7, 8]])
mat_a @ mat_b    # matrix multiplication
```

---

## 10. Broadcasting

NumPy automatically expands smaller arrays to match the shape of larger ones during arithmetic, without copying data — as long as their trailing dimensions are compatible (equal, or one of them is 1).

| Rule | Description |
|---|---|
| Compare shapes from the right | Dimensions must be equal, or one of them must be 1 |
| Missing dimensions | Treated as 1 and stretched to match |

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
row = np.array([10, 20, 30])              # shape (3,)

arr + row     # row is broadcast across both rows

col = np.array([[1], [2]])                # shape (2, 1)
arr + col     # col is broadcast across both columns

arr * 2       # scalar broadcast to every element
```

---

## 11. Aggregation Functions

| Function | Description |
|---|---|
| `.sum()` / `.mean()` | Sum / average of all elements |
| `.min()` / `.max()` | Minimum / maximum |
| `.std()` / `.var()` | Standard deviation / variance |
| `.argmin()` / `.argmax()` | Index of min/max value |
| `axis=0` | Aggregate down columns |
| `axis=1` | Aggregate across rows |
| `.cumsum()` / `.cumprod()` | Running cumulative sum / product |

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.sum()          # total of all elements
arr.sum(axis=0)    # column-wise sums -> [5, 7, 9]
arr.sum(axis=1)    # row-wise sums -> [6, 15]

arr.mean()
arr.std()
arr.min()
arr.argmax()        # index of the max in flattened array

arr.cumsum()
```

---

## 12. Sorting & Searching

| Function | Description |
|---|---|
| `np.sort(arr)` | Return a sorted copy |
| `arr.sort()` | Sort in place |
| `np.argsort(arr)` | Indices that would sort the array |
| `np.unique(arr)` | Sorted unique values |
| `np.searchsorted(arr, val)` | Index where `val` should be inserted to keep order |
| `np.nonzero(arr)` | Indices of non-zero elements |

```python
arr = np.array([3, 1, 2])

np.sort(arr)          # [1, 2, 3]
np.argsort(arr)       # [1, 2, 0]
np.unique([1, 1, 2, 3, 3])

np.searchsorted([1, 3, 5], 4)   # 2
np.nonzero([0, 3, 0, 5])         # (array([1, 3]),)
```

---

## 13. Random Numbers

| Function | Description |
|---|---|
| `np.random.seed(n)` | Set seed for reproducibility |
| `np.random.rand(shape)` | Uniform random values in [0, 1) |
| `np.random.randn(shape)` | Random values from standard normal distribution |
| `np.random.randint(low, high, size)` | Random integers |
| `np.random.choice(arr, size)` | Random sample from an array |
| `np.random.shuffle(arr)` | Shuffle an array in place |

```python
np.random.seed(42)

np.random.rand(3)              # 3 uniform values
np.random.randn(2, 2)          # 2x2 normal distribution
np.random.randint(1, 100, 5)   # 5 random ints between 1 and 99
np.random.choice([10, 20, 30], size=2, replace=False)

arr = np.array([1, 2, 3, 4])
np.random.shuffle(arr)
```

---

## 14. Linear Algebra

| Function | Description |
|---|---|
| `np.dot(a, b)` / `a @ b` | Matrix multiplication |
| `np.linalg.inv(mat)` | Matrix inverse |
| `np.linalg.det(mat)` | Determinant |
| `np.linalg.eig(mat)` | Eigenvalues and eigenvectors |
| `np.linalg.solve(A, b)` | Solve linear system `Ax = b` |
| `np.linalg.norm(arr)` | Vector/matrix norm (magnitude) |

```python
mat = np.array([[1, 2], [3, 4]])

np.linalg.inv(mat)
np.linalg.det(mat)
np.linalg.eig(mat)

A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
np.linalg.solve(A, b)

np.linalg.norm(np.array([3, 4]))   # 5.0
```

---

## 15. Copy vs View

| Concept | Description |
|---|---|
| View | Shares memory with the original — slicing returns a view by default |
| Copy | Independent array — `.copy()` guarantees no link back |
| `.base` | Check if an array owns its data (`None`) or is a view (points to original) |

```python
arr = np.array([1, 2, 3, 4])

view = arr[1:3]        # view - shares memory
view[0] = 99            # modifies arr too!

copy = arr[1:3].copy()  # independent copy
copy[0] = -1             # does NOT affect arr

view.base is arr        # True
copy.base is None        # True
```

---

## 16. Iterating Arrays

| Function | Description |
|---|---|
| `for x in arr` | Iterate over the first axis |
| `for x in np.nditer(arr)` | Iterate over every element, any dimension |
| `np.ndenumerate(arr)` | Iterate with index + value pairs |

```python
arr = np.array([[1, 2], [3, 4]])

for row in arr:
    print(row)

for val in np.nditer(arr):
    print(val)

for idx, val in np.ndenumerate(arr):
    print(idx, val)
```

---

## 17. Working with Missing / Special Values (NaN, Inf)

| Function | Description |
|---|---|
| `np.nan` | Represents a missing/undefined float value |
| `np.isnan(arr)` | Boolean mask — is each value NaN |
| `np.nansum()` / `np.nanmean()` | Aggregations that ignore NaN |
| `np.isinf(arr)` | Boolean mask — is each value infinite |
| `np.isfinite(arr)` | Boolean mask — is each value finite (not NaN/Inf) |

```python
arr = np.array([1, np.nan, 3, np.inf])

np.isnan(arr)
np.isinf(arr)
np.isfinite(arr)

np.nansum(arr)     # ignores NaN
np.nanmean(np.array([1, np.nan, 3]))
```

---

## 18. Set Operations

| Function | Description |
|---|---|
| `np.union1d(a, b)` | Union of two arrays |
| `np.intersect1d(a, b)` | Common elements |
| `np.setdiff1d(a, b)` | Elements in `a` but not in `b` |
| `np.in1d(a, b)` / `np.isin(a, b)` | Boolean mask — is each element of `a` in `b` |

```python
a = np.array([1, 2, 3, 4])
b = np.array([3, 4, 5, 6])

np.union1d(a, b)         # [1,2,3,4,5,6]
np.intersect1d(a, b)     # [3,4]
np.setdiff1d(a, b)       # [1,2]
np.isin(a, b)             # [False, False, True, True]
```

---

## 19. Save and Load Arrays

| Function | Description |
|---|---|
| `np.save(path, arr)` | Save a single array to a binary `.npy` file |
| `np.load(path)` | Load a `.npy` file |
| `np.savez(path, a=arr1, b=arr2)` | Save multiple arrays to one `.npz` file |
| `np.savetxt(path, arr)` | Save as a plain text/CSV file |
| `np.loadtxt(path)` | Load from a plain text file |

```python
np.save('arr.npy', arr)
loaded = np.load('arr.npy')

np.savez('arrays.npz', a=a, b=b)
data = np.load('arrays.npz')
data['a']

np.savetxt('arr.csv', arr, delimiter=',')
np.loadtxt('arr.csv', delimiter=',')
```

---

## 20. Vectorization & Performance

| Tip | Description |
|---|---|
| Prefer vectorized ops | `a + b` instead of looping element by element |
| Avoid Python loops on arrays | NumPy's C-level loops are far faster |
| Use `np.vectorize(func)` sparingly | Convenience only — not truly faster than a loop |
| Preallocate arrays | `np.zeros(shape)` then fill, instead of growing with `append` |
| Use in-place ops when possible | `arr += 1` instead of `arr = arr + 1` (avoids new allocation) |
| Check `.flags` for contiguity | Contiguous memory layout speeds up operations |

```python
# Slow - Python loop
result = [x * 2 for x in range(1000000)]

# Fast - vectorized
arr = np.arange(1000000)
result = arr * 2

# Preallocate instead of growing
out = np.zeros(5)
for i in range(5):
    out[i] = i ** 2

arr.flags['C_CONTIGUOUS']
```

