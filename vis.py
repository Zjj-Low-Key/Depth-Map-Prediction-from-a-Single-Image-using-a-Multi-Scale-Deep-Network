from json import load
import mindspore
import mindspore.ops as ops
from PIL import Image
import matplotlib.pyplot as plt
from model import coarseNet, fineNet
from data import NYUDataset
import mindspore.dataset as ds

test_dataset = ds.GeneratorDataset(NYUDataset('test'), column_names=['t_image', 't_depth', 't_valid_mask'], shuffle=False, num_parallel_workers=1)
test_dataset = test_dataset.batch(1, drop_remainder=False)
test_dataloader = test_dataset.create_dict_iterator()

coarse_model = coarseNet()
fine_model = fineNet()
coarse_param = mindspore.load_checkpoint('/data/local_userdata/zhujiajun/mindspore/checkpoint/coarse_model_34.ckpt')
mindspore.load_param_into_net(coarse_model, coarse_param)
fine_param = mindspore.load_checkpoint('/data/local_userdata/zhujiajun/mindspore/checkpoint/fine_model_21.ckpt')
mindspore.load_param_into_net(fine_model, fine_param)

coarse_model.set_train(False)
fine_model.set_train(False)
for i, batch in enumerate(test_dataloader):
    image_tensor = batch['t_image']
    depth = batch['t_depth']
    valid_mask = batch['t_valid_mask']
    depth = depth.squeeze()
    depth = ops.exp(depth)
    valid_mask = valid_mask.squeeze()
    coarse_output = coarse_model(image_tensor)
    fine_output = fine_model(image_tensor, coarse_output)
    coarse_output = coarse_output.squeeze()
    coarse_output = ops.exp(coarse_output)
    coarse_output = ops.clip(coarse_output, 0.1, 10)
    plt.imsave(f'/data/local_userdata/zhujiajun/mindspore/vis/coarse_output_{i}.png', coarse_output,cmap='gray')
    fine_output = fine_output.squeeze()
    fine_output = ops.exp(fine_output)
    fine_output = ops.clip(fine_output, 0.1, 10)
    plt.imsave(f'/data/local_userdata/zhujiajun/mindspore/vis/fine_output_{i}.png', fine_output,cmap='gray')
