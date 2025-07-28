import torch
from absl import flags, app
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import trange

from model import UNet
from score import get_inception_and_fid_score

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


# This function warmup_lr(step) implements a linear learning rate warm-up schedule -
# a common technique in training neural networks,
# especially in deep learning setups like transformers or diffusion models.
# When step <= FLAGS.warmup, the function returns a value between 0 and 1, increasing linearly.
# When step > FLAGS.warmup, the function returns 1.0 (i.e., full learning rate).
def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()

    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def main(argv):
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    # dataset = MNIST(
    #     root='./data', train=True, download=True,
    #     transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(T=FLAGS.T, out_ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
                     num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    torch.onnx.export(net_model, (torch.randn(128, 3, 32, 32), torch.randint(0, 1000, (128, 1))),
                      FLAGS.logdir + 'model.onnx')
    # ema_model = copy.deepcopy(net_model).to(device)
    # optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    # sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    # trainer = GaussianDiffusionTrainer(
    #     net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    # net_sampler = GaussianDiffusionSampler(
    #     net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
    #     FLAGS.mean_type, FLAGS.var_type).to(device)
    # ema_sampler = GaussianDiffusionSampler(
    #     ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
    #     FLAGS.mean_type, FLAGS.var_type).to(device)
    # if FLAGS.parallel:
    #     trainer = torch.nn.DataParallel(trainer)
    #     net_sampler = torch.nn.DataParallel(net_sampler)
    #     ema_sampler = torch.nn.DataParallel(ema_sampler)
    #
    # # log setup
    # os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    # x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    # x_T = x_T.to(device)
    # grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    # writer = SummaryWriter(FLAGS.logdir)
    # writer.add_image('real_sample', grid)
    # writer.flush()
    #
    # # backup all arguments
    # with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
    #     f.write(FLAGS.flags_into_string())
    #
    # # show model size
    # model_size = 0
    # for param in net_model.parameters():
    #     model_size += param.data.nelement()
    # print('Model params: %.2f M' % (model_size / 1024 / 1024))
    #
    # # start training
    # with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
    #     for step in pbar:
    #         # train
    #         optim.zero_grad()
    #         x_0 = next(datalooper).to(device)
    #         loss = trainer(x_0).mean()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
    #         optim.step()
    #         sched.step()
    #         ema(net_model, ema_model, FLAGS.ema_decay)
    #
    #         # log
    #         writer.add_scalar('loss', loss, step)
    #         pbar.set_postfix(loss='%.3f' % loss)
    #
    #         # sample
    #         if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
    #             net_model.eval()
    #             with torch.no_grad():
    #                 x_0 = ema_sampler(x_T)
    #                 grid = (make_grid(x_0) + 1) / 2
    #                 path = os.path.join(
    #                     FLAGS.logdir, 'sample', '%d.png' % step)
    #                 save_image(grid, path)
    #                 writer.add_image('sample', grid, step)
    #             net_model.train()
    #
    #         # save
    #         if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
    #             ckpt = {
    #                 'net_model': net_model.state_dict(),
    #                 'ema_model': ema_model.state_dict(),
    #                 'sched': sched.state_dict(),
    #                 'optim': optim.state_dict(),
    #                 'step': step,
    #                 'x_T': x_T,
    #             }
    #             torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))
    #
    #         # evaluate
    #         if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
    #             net_IS, net_FID, _ = evaluate(net_sampler, net_model)
    #             ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
    #             metrics = {
    #                 'IS': net_IS[0],
    #                 'IS_std': net_IS[1],
    #                 'FID': net_FID,
    #                 'IS_EMA': ema_IS[0],
    #                 'IS_std_EMA': ema_IS[1],
    #                 'FID_EMA': ema_FID
    #             }
    #             pbar.write(
    #                 "%d/%d " % (step, FLAGS.total_steps) +
    #                 ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
    #             for name, value in metrics.items():
    #                 writer.add_scalar(name, value, step)
    #             writer.flush()
    #             with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
    #                 metrics['step'] = step
    #                 f.write(json.dumps(metrics) + "\n")
    # writer.close()


if __name__ == '__main__':
    app.run(main)
