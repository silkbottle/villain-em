import numpy as np
from scipy.stats import norm
from scipy.misc import logsumexp

class villainEM():
    def __init__(self, h = 96, w = 75, max_iter=5, tol=1e-1):
        self.h = h
        self.w = w
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, pics):
        N = pics.shape[2]
        H = pics.shape[0]
        W = pics.shape[1]
        dh = H - self.h + 1
        dw = W - self.w + 1
        
        self.weights = np.ones((dh, dw), dtype=np.float64)
        self.weights /= self.weights.sum()
        
        self.B = np.uint8(pics.mean(axis=2))
        
        self.F = np.zeros((self.h, self.w))
        for n in xrange(N):
            pic = pics[:, :, n] - self.B
            dd = 0
            max_s = 0
            for d in xrange(dh*dw):
                pos_i = d/dw
                pos_j = d%dw
                s = np.fabs(pic[pos_i:pos_i+self.h, pos_j:pos_j+self.w].sum())
                if s > max_s:
                    dd, max_s = d, s
            self.F += pics[dd/dw:dd/dw+self.h, dd%dw:dd%dw+self.w, n]
        self.F = np.uint8(self.F/N)
        
        self.sigma = 0.0
        for p in xrange(pics.shape[2]):
            self.sigma += ((pics[:, :, p] - self.B)**2).sum()
        self.sigma = np.sqrt(self.sigma/H/W/N)
        self.log_likelihood = [float("-inf")]
        
        for n_iter in xrange(self.max_iter):
            # E-step #
            log_bkg_probs = norm.logpdf(np.kron(np.ones((H, W, 1), dtype=np.uint8), np.arange(256).reshape((1, 1, 256))),
                                        np.kron(self.B.reshape(H, W, 1), np.ones((1, 1, 256), dtype=np.uint8)), self.sigma)
            pixW, pixH = np.meshgrid(np.arange(W), np.arange(H))
            pixH = np.kron(pixH.reshape(H, W, 1), np.ones(N, dtype=np.int))
            pixW = np.kron(pixW.reshape(H, W, 1), np.ones(N, dtype=np.int))
            log_bkg = log_bkg_probs[pixH, pixW, pics]
            
            log_face_probs = norm.logpdf(np.kron(np.ones((self.h, self.w, 1), dtype=np.uint8), np.arange(256).reshape((1, 1, 256))),
                                         np.kron(self.F.reshape(self.h, self.w, 1), np.ones((1, 1, 256), dtype=np.uint8)), self.sigma)
            pixw, pixh = np.meshgrid(np.arange(self.w), np.arange(self.h))
            pixh = np.kron(pixh.reshape(self.h, self.w, 1), np.ones(N, dtype=np.int))
            pixw = np.kron(pixw.reshape(self.h, self.w, 1), np.ones(N, dtype=np.int))
            
            log_gamma = np.zeros((dh, dw, N), dtype=np.float64)
            for d in xrange(dh*dw):
                pos_i = d/dw
                pos_j = d%dw
                log_face = log_face_probs[pixh, pixw, pics[pos_i:pos_i+self.h, pos_j:pos_j+self.w]]
                log_pics = log_bkg.copy()
                log_pics[pos_i:pos_i+self.h, pos_j:pos_j+self.w] = log_face
                log_gamma[pos_i, pos_j] = np.log(self.weights[pos_i, pos_j]) + log_pics.sum(axis=(0,1))
            gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=(0,1)).reshape((1,1,N)))
            self.log_likelihood.append(logsumexp(log_gamma))
            # M-step #
            self.weights = gamma.sum(axis=2)/N
            Fnew = np.zeros(self.F.shape, dtype=np.float64)
            Bnew = np.zeros(self.B.shape, dtype=np.float64)
            for d in xrange(dh*dw):
                pos_i = d/dw
                pos_j = d%dw
                Fnew += (gamma[pos_i, pos_j].reshape((1, 1, N))*pics[pos_i:pos_i+self.h, pos_j:pos_j+self.w]).sum(axis=2)
                Bnew += (gamma[pos_i, pos_j].reshape((1, 1, N))*pics).sum(axis=2)
            self.F = np.uint8(Fnew/N)
            self.B = np.uint8(Bnew/N)
            
#            Snew = np.zeros(1, dtype=np.float64)
#            for d in xrange(dh*dw):
#                pos_i = d/dw
#                pos_j = d%dw
#                pic = self.B.copy()
#                pic[pos_i:pos_i+self.h, pos_j:pos_j+self.w] = self.F.copy()
#                Snew += (gamma[pos_i, pos_j]*(((pics-pic.reshape((H, W, 1)))**2).sum(axis=(0, 1)))).sum()
#            self.sigma = np.sqrt(Snew/N)
            # Convergency criterion #
            print self.log_likelihood[-1]
            if np.fabs(self.log_likelihood[-1] - self.log_likelihood[-2]) < self.tol:
                break