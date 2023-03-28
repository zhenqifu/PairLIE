import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import argparse
from thop import profile
from net.net import net
from data import get_eval_set
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *


parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
# parser.add_argument('--data_test', type=str, default='../dataset/LIE/LOL-test/raw')
# parser.add_argument('--data_test', type=str, default='../dataset/LIE/SICE-test/image')
parser.add_argument('--data_test', type=str, default='../dataset/LIE/MEF')
parser.add_argument('--model', default='weights/PairLIE.pth', help='Pretrained base model')  
parser.add_argument('--output_folder', type=str, default='results/MEF/')
opt = parser.parse_args()


print('===> Loading datasets')
test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')
model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]
        input = input.cuda()
        print(name)

        with torch.no_grad():
            L, R, X = model(input)
            D = input- X        
            I = torch.pow(L,0.2) * R  # default=0.2, LOL=0.14.
            # flops, params = profile(model, (input,))
            # print('flops: ', flops, 'params: ', params)

        if not os.path.exists(opt.output_folder):
            os.mkdir(opt.output_folder)
            os.mkdir(opt.output_folder + 'L/')
            os.mkdir(opt.output_folder + 'R/')
            os.mkdir(opt.output_folder + 'I/')  
            os.mkdir(opt.output_folder + 'D/')                       

        L = L.cpu()
        R = R.cpu()
        I = I.cpu()
        D = D.cpu()        

        L_img = transforms.ToPILImage()(L.squeeze(0))
        R_img = transforms.ToPILImage()(R.squeeze(0))
        I_img = transforms.ToPILImage()(I.squeeze(0))                
        D_img = transforms.ToPILImage()(D.squeeze(0))  

        L_img.save(opt.output_folder + '/L/' + name[0])
        R_img.save(opt.output_folder + '/R/' + name[0])
        I_img.save(opt.output_folder + '/I/' + name[0])  
        D_img.save(opt.output_folder + '/D/' + name[0])                       

    torch.set_grad_enabled(True)

eval()


