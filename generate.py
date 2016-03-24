import numpy as np
import scipy as sp
from scipy import misc

def Generate(background, bkg_size, picture, pic_size, N_samples, sigma):
    bkg = sp.misc.imread(background)
    bkg = 0.299*bkg[:, :, 0] + 0.587*bkg[:, :, 1] + 0.114*bkg[:, :, 2]
    bkg = sp.misc.imresize(bkg, size=bkg_size)
    pic = sp.misc.imread(picture)
    pic = 0.299*pic[:, :, 0] + 0.587*pic[:, :, 1] + 0.114*pic[:, :, 2]
    pic = sp.misc.imresize(pic, size=pic_size)
    
    W = np.random.random((bkg.shape[0] - pic.shape[0] + 1, bkg.shape[1] - pic.shape[1] + 1))
    W /= W.sum()
    prob = np.hstack((0.0, W.reshape(W.shape[0]*W.shape[1])))
    for i in xrange(1,prob.shape[0]):
        prob[i] += prob[i-1]
    
    seeds = np.random.random(N_samples)
    pos = np.empty((N_samples, 2), dtype=np.int16)
    for i in xrange(1, prob.shape[0]):
        seeds_in_range = np.logical_and(prob[i-1] <= seeds, seeds < prob[i]).nonzero()[0]
        if seeds_in_range.shape[0] != 0:
            pos[seeds_in_range, 0] = (i-1)/W.shape[1]
            pos[seeds_in_range, 1] = (i-1)%W.shape[1]
            
    pics = np.zeros((bkg.shape[0], bkg.shape[1], N_samples), dtype=np.float32)
    for i in xrange(N_samples):
        pics[:, :, i] += bkg
        pics[pos[i, 0]:pos[i, 0]+pic.shape[0], pos[i, 1]:pos[i, 1]+pic.shape[1], i] = pic
    pics += np.random.normal(0, sigma, pics.shape)
    pics[pics < 0] = 0
    pics[pics > 255] = 255
    
    return np.uint8(pics)