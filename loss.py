import mindspore.ops as ops

def custom_loss_function(output, target, valid_mask=None):
    valid_mask = valid_mask.astype(output.dtype) / 255
    di = (target - output) * valid_mask
    n = valid_mask.sum((1,2,3))
    di2 = ops.pow(di, 2)
    fisrt_term = ops.sum(di2,(1,2,3))
    second_term = 0.5*ops.pow(ops.sum(di,(1,2,3)), 2)
    loss = fisrt_term / n - second_term/ n ** 2
    return loss.mean()

def scale_invariant(output, target, valid_mask=None):
    # di = output - target
    valid_mask = valid_mask.astype(output.dtype) / 255
    di = (target - output) * valid_mask
    n = valid_mask.sum((1,2,3))
    di2 = ops.pow(di, 2)
    fisrt_term = ops.sum(di2,(1,2,3))/n
    second_term = ops.pow(ops.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()