from networks.net_factory import net_factory
import torch
from torchstat import stat
from thop import profile



def create_model(model_name):
    model = net_factory(net_type=model_name, in_chns=3,
                        class_num=num_classes)
    return model


if __name__ == '__main__':
    # Hyperparameters
    img_size = 224
    batch_size = 1
    learning_rate = 1e-3
    num_epochs = 10
    num_classes = 7
    channels = 3
    model_name = "ShiftMLP_b"

    # Initialize model
    model = create_model(model_name).cpu()
    # print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    # 计算参数量
    params_num = sum(p.numel() for p in model.parameters())
    print("\nModle's Params: %.3fM" % (params_num / 1e6))
    x = torch.randn(batch_size, 3, img_size, img_size).cpu()

    flops, params = profile(model, inputs=(x,))
    print('flops:{} G'.format(flops / 1e9))
    print('params:{}'.format(params))

    output = model(x)

    print(output.size())