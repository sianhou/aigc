import copy
import os

import torch
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score import get_inception_and_fid_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

attn = [1]
batch_size = 128
beta_1 = 0.0001
beta_T = 0.02
ch = 16
ch_mult = [1, 2, 2]
ema_decay = 0.9999
eval_step = 100000
dropout = 0.1
fid_cache = "./stats/mnist2.train.npz"
fid_use_torch = False
grad_clip = 1.0
img_size = 28
logdir = "./logs/DDPM_MNIST2_EPS"
lr = 0.0002
num_images = 5000
num_res_blocks = 2
num_workers = 4
mean_type = "epsilon"
sample_size = 64
sample_step = 1000
save_step = 5000
T = 1000
total_steps = 100000
var_type = "fixedlarge"
warmup = 5000


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def evaluate(sampler, model, batch_size=32):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, num_images, batch_size, desc=desc):
            batch_size = min(batch_size, num_images - i)
            x_T = torch.randn((batch_size, 3, img_size, img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()

    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, fid_cache, num_images=num_images,
        use_torch=fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


# This function warmup_lr(step) implements a linear learning rate warm-up schedule -
# a common technique in training neural networks,
# especially in deep learning setups like transformers or diffusion models.
# When step <=  warmup, the function returns a value between 0 and 1, increasing linearly.
# When step >  warmup, the function returns 1.0 (i.e., full learning rate).
def warmup_lr(step):
    return min(step, warmup) / warmup


def main():
    # data
    dataset = MNIST(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]))

    # # 筛选标签为 1 和 2 的索引
    # target_indices = np.where((np.array(dataset.targets) == 1) |
    #                           (np.array(dataset.targets) == 2))[0]
    #
    # # 创建子数据集
    # filtered_dataset = Subset(dataset, target_indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model
    net_model = UNet(T=T, in_ch=1, out_ch=ch, ch_mult=ch_mult, attn=attn, num_res_blocks=num_res_blocks,
                     dropout=dropout)
    ema_model = copy.deepcopy(net_model).to(device)
    optim = torch.optim.Adam(net_model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(net_model, beta_1, beta_T, T).to(device)
    net_sampler = GaussianDiffusionSampler(net_model, beta_1, beta_T, T, img_size,
                                           mean_type, var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(ema_model, beta_1, beta_T, T, img_size,
                                           mean_type, var_type).to(device)

    # log setup
    os.makedirs(os.path.join(logdir, 'sample'), exist_ok=True)
    x_T = torch.randn(sample_size, 1, img_size, img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:sample_size]) + 1) / 2
    writer = SummaryWriter(logdir)
    writer.add_image('real_sample', grid)
    writer.flush()

    # backup all arguments
    with open(os.path.join(logdir, "flagfile.txt"), 'w') as f:
        f.write("Beginning:")

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if sample_step > 0 and step % sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if save_step > 0 and step % save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(logdir, 'ckpt.pt'))

            # # evaluate
            # if eval_step > 0 and step % eval_step == 0:
            #     net_IS, net_FID, _ = evaluate(net_sampler, net_model)
            #     ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
            #     metrics = {
            #         'IS': net_IS[0],
            #         'IS_std': net_IS[1],
            #         'FID': net_FID,
            #         'IS_EMA': ema_IS[0],
            #         'IS_std_EMA': ema_IS[1],
            #         'FID_EMA': ema_FID
            #     }
            #     pbar.write(
            #         "%d/%d " % (step, total_steps) +
            #         ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
            #     for name, value in metrics.items():
            #         writer.add_scalar(name, value, step)
            #     writer.flush()
            #     with open(os.path.join(logdir, 'eval.txt'), 'a') as f:
            #         metrics['step'] = step
            #         f.write(json.dumps(metrics) + "\n")
    writer.close()


if __name__ == '__main__':
    main()
