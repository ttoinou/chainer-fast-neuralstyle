import numpy as np
import argparse
from PIL import Image
import time
import os

import chainer
from chainer import cuda, Variable, serializers
from net import *

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('input')
parser.add_argument('--input', '-i', type=str)
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
args = parser.parse_args()


def restartTime():
    return time.time()

def showTime(msg,start):
    print msg,' => ',time.time() - start, ' sec'

def generate(input):
    start = restartTime()
    x = Variable(input)
    
    y = model(x)
    result = cuda.to_cpu(y.data)
    
    result = result.transpose(0, 2, 3, 1)
    result = result.reshape((result.shape[1:]))
    result = np.uint8(result)
    
    showTime('generate \t',start)
    return result

def generateFromImageUrl(inputUrl,outputUrl):
    start = restartTime()
    
    image = xp.asarray(Image.open(inputUrl).convert('RGB'), dtype=xp.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    result = Image.fromarray(generate(image)).save(outputUrl)
    
    showTime('generateImage \t',start)

# loading
start = restartTime()

model = FastStyleNet()
serializers.load_npz(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

showTime('loading\t',start)


def main(args):
    if os.path.isdir(args.input):
        fs = os.listdir(args.input)
        imagesPaths = []
        for fn in fs:
            base, ext = os.path.splitext(fn)
            if ext == '.jpg' or ext == '.png':
                imagepath = os.path.join(args.input,fn)
                imagesPaths.append(imagepath)
        
        print 'folder ',args.input,' has ',len(imagesPaths),'images'
        
        for imagePath in imagesPaths:
            #print imagePath
            #print os.path.join(args.out,os.path.basename(imagePath))
            generateFromImageUrl(imagePath,os.path.join(args.out,os.path.basename(imagePath)))
    else:
        generateFromImageUrl(args.input,args.out)
        

if __name__ == '__main__':
    main(args)