# foograd cheat sheet

Insights picked up while building the engine, one section per idea.
Updated as each phase adds something new.

---

## 1. The one rule behind every backward pass

- `x.grad` always has the **shape of `x`**.
- `out.grad` always has the **shape of `out`**. It is fixed by the forward pass; it never changes depending on which input you are computing.
- A backward pass is: take `out.grad`, combine it with the local derivative, then **get it back to the input's shape**.
- Always accumulate: `x.grad += ...`, never `=`. A node used twice must collect gradient from both uses.

---

## 2. Broadcasting (forward)

Line shapes up **from the right**. Per dimension:

| left | right | result |
|---|---|---|
| equal | equal | that size |
| 1 | n | n (the 1 is copied) |
| missing | n | n (padded on the left as 1, then copied) |
| m | n, m != n, neither 1 | **error** |

```
(3, 2)    + (2,)   -> (3, 2)
(3, 4)    + (3, 1) -> (3, 4)
(2, 3, 4) + (3, 1) -> (2, 3, 4)
(3, 1)    + (1, 4) -> (3, 4)
(3, 2)    + (3,)   -> error   (2 vs 3 on the right)
(3, 2)    + ()     -> (3, 2)
```

---

## 3. Copy forward, sum backward (and the reverse)

| forward | backward |
|---|---|
| **broadcast** (a value is copied to many places) | **sum** the gradients from those places back in |
| **sum** (many values collapse into one) | **broadcast** the gradient back out to all of them |

These are mirror images. Whenever a value is reused, its gradients add up.

### `unbroadcast(grad, shape)`, in this order

1. While `grad.ndim > len(shape)`: `grad = grad.sum(axis=0)` (dims padded on the left never existed in the input).
2. For each axis `i` where `shape[i] == 1` and `grad.shape[i] != 1`: `grad = grad.sum(axis=i, keepdims=True)` (the input had size 1 and was copied).

Example: grad `[[1,0],[2,0],[3,1]]` back to `(2,)` gives `[6, 1]`.

---

## 4. Elementwise ops

| op | local grads | note |
|---|---|---|
| `a + b` | 1, 1 | `unbroadcast(out.grad, a.shape)` |
| `a * b` | `b`, `a` | **multiply first, then unbroadcast**: `unbroadcast(b.data * out.grad, a.shape)` |
| `a ** n` (scalar n) | `n * a**(n-1)` | don't forget `* out.grad` |
| `-a`, `a - b`, `a / b` | built from `+`, `*`, `**` | no new backward code needed |

---

## 5. Matmul (2-D)

`out = A @ B`, with `A (m,k)`, `B (k,n)`, `out (m,n)`.

```
A.grad += out.grad @ B.T     (m,n) @ (n,k) -> (m,k)
B.grad += A.T @ out.grad     (k,m) @ (m,n) -> (k,n)
```

**Don't memorise it, derive it from shapes:** take `out.grad` and the *other* input (transposed if needed), and order them so the shapes produce the input's shape.

- `(a,b) @ (c,d)` needs `b == c` and gives `(a,d)`.
- Transpose swaps rows and columns: `(m,n) -> (n,m)`, `T[i][j] = X[j][i]`. The diagonal stays put.

Intuition with an all-ones `out.grad`:

```
A = [[1,2],[3,4]]   B = [[5,6],[7,8]]
A.grad = [[11,15],[11,15]]   each row = row sums of B    (5+6, 7+8)
B.grad = [[4,4],[6,6]]       each row = column sums of A (1+3, 2+4)
```

Matmul is **not commutative**: `__rmatmul__(self, other)` means `other @ self`.

---

## 6. Reductions: sum and mean

Backward = broadcast `out.grad` back to `x.shape`. The catch is **axis alignment**.

With `x = [[1,2,3],[4,5,6]]` `(2,3)`:

| forward | `s.shape` | upstream `s.grad` | `x.grad` |
|---|---|---|---|
| `x.sum()` | `()` | `1` | all ones |
| `x.sum(axis=0)` | `(3,)` | `[10,20,30]` | `[[10,20,30],[10,20,30]]` (columns constant) |
| `x.sum(axis=1)` | `(2,)` | `[10,20]` | `[[10,10,10],[20,20,20]]` (rows constant) |
| `x.sum(axis=1, keepdims=True)` | `(2,1)` | `[[10],[20]]` | `[[10,10,10],[20,20,20]]` |
| `x.mean()` | `()` | `1` | all `1/6` |

The gradient goes back **along the axis that was summed**.

**The gotcha:** `np.broadcast_to(s.grad, x.shape)` aligns from the right.

- `axis=1`: `(2,)` vs `(2,3)` gives 2 vs 3, an **error**. You need `(2,1)`.
- `axis=0`: `(3,)` vs `(2,3)` happens to work, because the summed axis wasn't the last one. **Testing only `axis=0` hides the bug.**
- Fix: `np.expand_dims(s.grad, axis)` puts the removed size-1 dim back in place.
- Not needed when `keepdims=True` (the dim is still there) or `axis=None` (a `()` scalar broadcasts to any shape).

`mean = sum / N`, where `N = x.size / out.size` (the number of elements averaged into each output). Each element's grad is divided by `N`.

---

## 7. Gradcheck

- Central difference: `(f(x + eps) - f(x - eps)) / (2 * eps)`, one element at a time.
- `backward()` seeds `out.grad` with ones, which is the same as the gradient of `out.sum()`. So compare against the numerical grad of `f(x).data.sum()`.
- Compare with `np.allclose(..., atol, rtol)`, never `==`.
- Prove the checker works: a deliberately broken backward must make gradcheck **return False**.
- Use **non-square** shapes: a wrong transpose then fails with a shape error instead of hiding.
- Use a **seeded** RNG (`np.random.default_rng(0)`) and small float inputs (e.g. `uniform(0.5, 2)`) so `/` and `**` stay well behaved and failures are reproducible.
- `np.random.default_rng(0)` **returns** a generator; it does not seed the global `np.random`. Keep it (`rng = ...`) and draw from `rng`.
- **Don't end on the op under test.** `backward()` seeds the last node with ones, so a backward that forgets `* out.grad` still passes. Chain something after it, e.g. `x.sum() ** 2`, so `out.grad != 1`.

---

## 8. Testing lessons

- **Expected values must come from somewhere independent**: hand-worked numbers or numerical gradients. Checking `a.grad == ones @ b.T` just repeats the code under test and passes even when it's wrong.
- Make sure a test fails **for the right reason**. A 2-D `(3,2) @ (3,2)` "unsupported shape" test passed because NumPy raised its own error, not ours.
- Check error messages with `pytest.raises(ValueError, match="...")` (it is a regex search).
- Hand-check one small case before trusting a big one.

---

## 9. NumPy / Python gotchas

- `ndarray <op> Tensor`: NumPy's operator runs first and either errors (`@`) or silently loops elementwise, returning an object array of Tensors (`+`). Fix: `__array_ufunc__ = None` on the class makes NumPy return `NotImplemented`, so Python calls our reflected method.
- `np.array(data)` copies; `np.asarray(data)` may alias the caller's array.
- `list.copy()` is shallow: the arrays inside are still shared.
- `ndarray.data` is a memoryview (it even has `.ndim`), so a missed `_coerce` can half-work instead of failing.
- `raise ValueError(...)`, not `return ValueError`.
- A 1-tuple is `(3,)`, not `(,3)`.
