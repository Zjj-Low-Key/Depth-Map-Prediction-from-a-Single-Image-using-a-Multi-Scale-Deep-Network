import mindspore.ops as ops

def threeshold_percentage(output, target, valid_mask, threeshold_val):
    valid_mask = valid_mask.astype(output.dtype) / 255
    n = ops.sum(valid_mask, (1,2,3))
    scale = ops.sum((target - output) * valid_mask, (1,2,3)) / n
    d1 = ops.exp(output) * ops.exp(scale) /ops.exp(target)
    d2 = ops.exp(target) / ops.exp(scale) /ops.exp(output)
    max_d1_d2 = ops.maximum(d1,d2)
    zero = ops.zeros((output.shape[0], output.shape[1], output.shape[2], output.shape[3]))
    one = ops.ones((output.shape[0], output.shape[1], output.shape[2], output.shape[3]))
    bit_mat = ops.where(max_d1_d2 < threeshold_val, one, zero)
    bit_mat[~valid_mask] = 0
    count_mat = ops.sum(bit_mat, (1,2,3))
    
    threeshold_mat = count_mat/ n
    return threeshold_mat.mean()

def rmse_linear(output, target, valid_mask):
    valid_mask = valid_mask.astype(output.dtype) / 255
    n = ops.sum(valid_mask, (1,2,3))
    scale = ops.sum((target - output) * valid_mask, (1,2,3)) / n
    actual_output = ops.exp(output) * ops.exp(scale)
    actual_target = ops.exp(target)
    # actual_output = output
    # actual_target = target
    diff = actual_output - actual_target
    diff[~valid_mask] = 0
    diff2 = ops.pow(diff, 2)
    mse = ops.sum(diff2, (1,2,3)) / n 
    rmse = ops.sqrt(mse)
    return rmse.mean()

def rmse_log(output, target, valid_mask):
    valid_mask = valid_mask.astype(output.dtype) / 255
    n = ops.sum(valid_mask, (1,2,3))
    scale = ops.sum((target - output) * valid_mask, (1,2,3)) / n
    diff = output + scale - target
    diff[~valid_mask] = 0
    # diff = ops.log(output) - ops.log(target)
    diff2 = ops.pow(diff, 2)
    mse = ops.sum(diff2, (1,2,3)) / n 
    rmse = ops.sqrt(mse)
    return rmse.mean()

def abs_relative_difference(output, target, valid_mask):
    valid_mask = valid_mask.astype(output.dtype) / 255
    n = ops.sum(valid_mask, (1,2,3))
    scale = ops.sum((target - output) * valid_mask, (1,2,3)) / n
    actual_output = ops.exp(output) * ops.exp(scale)
    actual_target = ops.exp(target)
    # actual_output = output
    # actual_target = target
    abs_relative_diff = ops.abs(actual_output - actual_target)/actual_target
    abs_relative_diff[~valid_mask] = 0
    abs_relative_diff = ops.sum(abs_relative_diff, (1,2,3)) / n
    return abs_relative_diff.mean()

def squared_relative_difference(output, target, valid_mask):
    valid_mask = valid_mask.astype(output.dtype) / 255
    n = ops.sum(valid_mask, (1,2,3))
    scale = ops.sum((target - output) * valid_mask, (1,2,3)) / n
    actual_output = ops.exp(output) * ops.exp(scale)
    actual_target = ops.exp(target)
    # actual_output = output
    # actual_target = target
    square_relative_diff = ops.pow(ops.abs(actual_output - actual_target), 2)/actual_target
    square_relative_diff[~valid_mask] = 0
    square_relative_diff = ops.sum(square_relative_diff, (1,2,3)) / n 
    return square_relative_diff.mean()