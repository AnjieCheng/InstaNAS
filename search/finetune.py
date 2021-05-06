# build-in libraries
import os
import argparse
import numpy as np
# installed libraries
import tqdm
# pytorch
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
# local files
import utils
# profile settings
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.set_num_threads(1)
# ignore warnings
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='InstaNas Finetune Stage')
parser.add_argument('--eval_freq', type=float, default=1, help='evaluate freqency')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=4e-5, help='weight decay')
parser.add_argument('--model', default='InstaMobile_ImgNet', help='<Net>_<Dataset>')
parser.add_argument('--data_dir', default='../data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load instanet+agent from')
parser.add_argument('--pretrained', default=None, help='load instanet pretrain only')
parser.add_argument('--cv_dir', default='./cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--max_epochs', type=int, default=300, help='total epochs to run')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--lr_type', default='cosine', type=str, metavar='T', help='learning rate strategy (default: cosine)', choices=['cosine', 'multistep'])
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)


def train(epoch):
    agent.eval()
    instanet.train()
    matches, rewards, policies, dur = [], [], [], []
    matches_, rewards_, policies_, dur_ = [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        cur_lr = utils.adjust_learning_rate_cos(optimizer, epoch, args.max_epochs, args.lr, batch=batch_idx, nBatch=len(trainloader), method=args.lr_type)
        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)

        #---------------------------------------------------------------------#
        probs, value = agent(inputs)

        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        with torch.no_grad():
            v_inputs = Variable(inputs.data)

        preds_map, lat_map  = instanet.forward(v_inputs, policy_map)
        reward_map, match_map, avg_elasped_map = utils.get_reward(preds_map, targets, policy_map.data, lat_map, instanet.module.baseline)

        loss = F.cross_entropy(preds_map, targets)
        #---------------------------------------------------------------------#

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        matches_.append(match_map.cpu())
        rewards_.append(reward_map.cpu())
        policies_.append(policy_map.data.cpu())
        dur_.append(np.mean(avg_elasped_map.data.cpu().numpy()))


    accuracy, reward, sparsity, policy_set = utils.performance_stats(policies_, rewards_, matches_)
    log_str = ' [*] FT - E: %d | A: %.3f | R: %.2E | S: %.3f | #: %d | D: %.3f |'%(epoch, accuracy, reward, sparsity, len(policy_set), np.mean(dur_))
    print (log_str)


def test(epoch):

    agent.eval()
    instanet.eval()

    matches, rewards, policies, dur = [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)

        probs, _ = agent(inputs)

        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0

        policy = Variable(policy)

        preds, lat = instanet.forward(inputs, policy)

        reward, match, avg_elasped = utils.get_reward(preds, targets, policy.data, lat, instanet.module.baseline)

        matches.append(match)
        rewards.append(reward)
        policies.append(policy.data)
        dur.append(np.mean(avg_elasped.data.cpu().numpy()))

    accuracy, reward, sparsity, policy_set = utils.performance_stats(policies, rewards, matches)

    log_str = ' [*] TS - A: %.3f | R: %.2E | S: %.3f | #: %d | D: %.3f |'%(accuracy, reward, sparsity, len(policy_set), np.mean(dur))
    print (log_str)

    # save the model
    agent_state_dict = agent.state_dict()
    instanet_state_dict = instanet.state_dict()

    state = {
      'agent': agent_state_dict,
      'instanet': instanet_state_dict,
      'epoch': epoch,
      'reward': reward,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_R_%.2E_S_%.2f_#_%d.t7'%(epoch, accuracy, reward, sparsity, len(policy_set)))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=32)
instanet, agent = utils.get_model(args.model)

# Load agent model
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    new_state = instanet.state_dict()
    new_state.update(checkpoint['instanet'])
    instanet.load_state_dict(new_state)
    start_epoch = 0
    # start_epoch = checkpoint['epoch'] + 1
    print (' [*] Loaded agent+instanet from', args.load)


if args.pretrained is not None:
    checkpoint = torch.load(args.pretrained)
    agent.load_state_dict(checkpoint['instanet'])
    print (' [*] Loaded instanet pretrained from', args.pretrained)


instanet.train().cuda()
agent.cuda()

print(" [*] Basline Latency:", instanet.module.baseline)

optimizer = optim.SGD(instanet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)

cur_lr = args.lr
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    print(" [*] Current Learning Rate:", cur_lr)
    ub = 0.4 - ((0.4-instanet.module.baseline)/args.max_epochs*epoch)
    lb = 0.2 - ((0.2-(instanet.module.baseline/2))/args.max_epochs*epoch)
    ub = ub.cuda()
    lb = lb.cuda()

    train(epoch)
    if (epoch+1)%args.eval_freq == 0:
        test(epoch)
