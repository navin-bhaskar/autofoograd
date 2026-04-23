def unbroadcast(grad, shape):
    # Reduce extra dimensions
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # Reduce dimensions where original shape is 1
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad