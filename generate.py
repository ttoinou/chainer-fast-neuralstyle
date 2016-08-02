import numpy as np
import argparse
from PIL import Image, ImageOps
import time
import os
import math
import random

import chainer
from chainer import cuda, Variable, serializers
from net import *

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('mode', default='', type=str)
parser.add_argument('--input', '-i', type=str)
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
parser.add_argument('--scale', '-s', default=1.0, type=float)
parser.add_argument('--border', '-b', default=0, type=int)
args = parser.parse_args()

def divisionNumberInequality(Mpx,w,h,o,n):
    return (w/float(n)+2*o)*(h/float(n)+2*o) - Mpx*1000*1000
'''
# not working
# wrong formula
def findDivisionNumber0(pixelmax,width,height,overlap):
    r = width/height + height/width - 2.0
    S = height*width
    o = overlap
    Max = math.sqrt( (o*o*r + 4*pixelmax*1000*1000)/S ) -o*(1.0/width + 1.0/height)
    
    return int(math.ceil( 1.0 / Max ) )
'''
def findDivisionNumber(Mpx,w,h,o):
    n = 1
    while divisionNumberInequality(Mpx,w,h,o,n) > 0:
        n += 1
    return n
    
def restartTime():
    return time.time()

def showTime(msg,start):
    print msg,' => ', int((time.time() - start)*1000.0), 'ms'

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

def symmetry(x,s):
    return s + s - x

def expandImage(args,image):
    w = image.size[0]
    h = image.size[1]
    
    image = ImageOps.expand(image,border=args.border,fill='white')
    # fill with real pixels the white border
    # assuming o < w and o < h
    pixels = image.load() # create the pixel map
    o = args.border
    X = 0
    Y = 0
    do = False
    
    for x in range(image.size[0]):    # for every pixel:
        for y in range(image.size[1]):
            X = x
            Y = y
            do = False
            
            
            if X < o:
                X = symmetry(X,o)
                do = True
            elif X > w+o-1:
                X = symmetry(X,w+o-1)
                do = True
                
            if Y < o:
                Y = symmetry(Y,o)
                do = True
            elif Y > h+o-1:
                Y = symmetry(Y,h+o-1)
                do = True
            
            #print X,Y,w,h
            if do:
                pixels[x,y] = pixels[X,Y]
    
    return image

def reduceImage(args,image):
    #return image
    return ImageOps.expand(image,border=-args.border,fill='white')

def imageUrlToArray(args,inputUrl):
    image = Image.open(inputUrl).convert('RGB')
    
    if args.scale != 1.0:
        image = ImageOps.fit(image, ( int(image.size[0]*args.scale) , int(image.size[1]*args.scale) ), 2)
    
    if args.border > 0:
        image = expandImage(args,image)
    
    print float(image.size[0]*image.size[1])/(1000000)," MPx"
    
    image = xp.asarray(image, dtype=xp.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    return image
    
def generateFromImageUrl(args,inputUrl,outputUrl):
    start = restartTime()
    image = imageUrlToArray(args,inputUrl)
    
    image = generate(image)
    image = Image.fromarray(image)
    
    if args.border > 0:
        image = reduceImage(args,image)
    
    image.save(outputUrl)
    showTime('with image Ops \t',start)

# loading
def loading(args):
    start = restartTime()
    
    model = FastStyleNet()
    serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
    
    showTime('loading\t\t',start)


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
            generateFromImageUrl(args,imagePath,os.path.join(args.out,os.path.basename(imagePath)))
    else:
        generateFromImageUrl(args,args.input,args.out)
        

if args.mode == 'divisionTest':
    j = 0
    while True:
        o = random.uniform(1,30)
        w = random.uniform(31,4000)
        h = random.uniform(31,4000)
        M = random.uniform(0.1,2)
        
        n = findDivisionNumber(M,w,h,o)
        if n > 1 and ( divisionNumberInequality(M,w,h,o,n) > 0 or divisionNumberInequality(M,w,h,o,n-1) < 0 ) :
            print "division test error",o,w,h,M,n
        else:
            j += 1
            
        if j % 1000000 == 0:
            print "division test no error",j
        
        
    
elif args.mode == '' or __name__ == '__main__':
    loading(args)
    main(args)
    