import numpy as np
from scipy.stats import norm
from scipy.misc import logsumexp

class VillainEM():
    def __init__(self, h = 96, w = 75, max_iter=5, tol=1e-1, hard=False, random_init=False):
        self.h = h
        self.w = w
        self.max_iter = max_iter
        self.tol = tol
        self.hard = hard
        self.random_init = random_init
        
    def fit(self, pics):
        N = pics.shape[2]
        H = pics.shape[0]
        W = pics.shape[1]
        dh = H - self.h + 1
        dw = W - self.w + 1
        pics = pics.astype(np.uint8)
        
        self.weights = np.ones((dh, dw), dtype=np.float64)
        self.weights /= self.weights.sum()
        if self.random_init:
            self.B = np.uint8(256*np.random.random((H, W)))
            self.F = np.uint8(256*np.random.random((self.h, self.w)))
            self.sigma = 256*np.random.random(1)
        else:
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
            for n in xrange(N):
                self.sigma += ((pics[:, :, n] - np.float32(self.B))**2).sum()
            self.sigma = np.sqrt(self.sigma/N/H/W)
        
        self.log_ll = [float("-inf")]
        
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
                
            if self.hard:
                positions = log_gamma.reshape((dh*dw, N)).argmax(axis=0)
                pos_i = positions/dw
                pos_j = positions%dw
                log_ll = 0.0
                for n in xrange(N):
                    log_ll += log_gamma[pos_i[n], pos_j[n], n]
                self.log_ll.append(log_ll)
            else:
                gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=(0,1)).reshape((1,1,N)))
                self.log_ll.append(logsumexp(log_gamma, axis=(0,1)).sum())
            
            # M-step #
            if self.hard:
                self.weights *= 0.0
                pos, n = np.unique(positions, return_counts=True)
                self.weights[pos/dw, pos%dw] += np.float32(n)/N
                Fnew = np.zeros((self.h, self.w), dtype=np.float32)
                Bnew = np.zeros((H, W), dtype=np.float32)
                Bnorm = N*np.ones((H, W), dtype=np.int32)
                for n in xrange(N):
                    picbuf = pics[:, :, n].copy()
                    picbuf[pos_i[n]:pos_i[n]+self.h, pos_j[n]:pos_j[n]+self.w] = 0
                    Bnew += picbuf
                    Bnorm[pos_i[n]:pos_i[n]+self.h, pos_j[n]:pos_j[n]+self.w] -= 1
                    Fnew += pics[pos_i[n]:pos_i[n]+self.h, pos_j[n]:pos_j[n]+self.w, n]
                self.F = np.uint8(Fnew/N)
                valid = Bnorm > 0
                self.B[valid] = np.uint8(Bnew[valid]/Bnorm[valid])

                Snew = np.zeros(1, dtype=np.float64)
                for n in xrange(N):
                    picbuf = self.B.copy().astype(np.float32)
                    picbuf[pos_i[n]:pos_i[n]+self.h, pos_j[n]:pos_j[n]+self.w] = self.F.copy()
                    Snew += ((pics[:, :, n]-picbuf)**2).sum()/H/W
                self.sigma = np.sqrt(Snew/N)
            else:
                self.weights = gamma.sum(axis=2)/N
                Fnew = np.zeros(self.F.shape, dtype=np.float64)
                Bnew = np.zeros(self.B.shape, dtype=np.float64)
                Bnorm = N*np.ones(self.B.shape, dtype=np.float64)
                for d in xrange(dh*dw):
                    pos_i = d/dw
                    pos_j = d%dw
                    Fnew += (gamma[pos_i, pos_j].reshape((1, 1, N))*pics[pos_i:pos_i+self.h, pos_j:pos_j+self.w]).sum(axis=2)
                    picsbuf = pics.copy()
                    picsbuf[pos_i:pos_i+self.h, pos_j:pos_j+self.w] = 0.0
                    Bnew += (gamma[pos_i, pos_j].reshape((1, 1, N))*picsbuf).sum(axis=2)
                    Bnorm[pos_i:pos_i+self.h, pos_j:pos_j+self.w] -= gamma[pos_i, pos_j].sum()
                self.F = np.uint8(Fnew/N)
                valid = Bnorm > 0
                self.B[valid] = np.uint8(Bnew[valid]/Bnorm[valid])

                Snew = np.zeros(1, dtype=np.float64)
                for d in xrange(dh*dw):
                    pos_i = d/dw
                    pos_j = d%dw
                    pic = self.B.copy().astype(np.float32)
                    pic[pos_i:pos_i+self.h, pos_j:pos_j+self.w] = self.F.copy()
                    Snew += (gamma[pos_i, pos_j]*(((pics-pic.reshape((H, W, 1)))**2).sum(axis=(0, 1))/H/W)).sum()
                self.sigma = np.sqrt(Snew/N)
            # Convergency criterion #
            if np.fabs(self.log_ll[-1] - self.log_ll[-2]) < self.tol:
                break