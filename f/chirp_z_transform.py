import math
import numpy as np

#chirpz from pychirpz adapted to use without numba speedup and commented by me

def nextpow2(n):
    
    
    """
    Return the smallest power of two greater than or equal to n.
    """
    return int(math.ceil(math.log(n)/math.log(2)))
               
def chirpz(x, M, A, W):
    """
    chirp z transform per Rabiner derivation pp1256
    x is our (complex) signal of length N
    
    If you want to evaluate finely spaced DFT, abs(A) = 1 as this evaluates on the unit circle on the z plane.
    A will give the frequency offset to start the evaluation at 
    M is number of points you want
    W is the step in frequency (complex exponential)
    
    To evaluate at e.g. 32 points between frequencies f1 to f2 - frequencies as if sample spacing = 1
    M = 32
    
    
    A = np.exp(-2j*np.pi*f1)
    
    And 
    
    W = np.exp(-2j*np.pi*(f2-f1)/(M))
    
    """
   
    N = len(x)
    #L is an efficient size for the convolutions required
    L = 2**(nextpow2(N + M -1))  # or nearest power of two
    #calculating first convolution vector
    yn = np.zeros(L, dtype=np.complex128)
    for n in range(N):
        yn_scale =  A**(-n) * W**((n**2.0)/2.0)
        yn[n] = x[n] * yn_scale
    Yr = np.fft.fft(yn)
    
    #calculate second convolution vector
    vn = np.zeros(L, dtype=np.complex128)
    for n in range(M):
        vn[n] = W**((-n**2.0)/2.0)
        
    for n in range(L-N+1, L):
        vn[n] = W**(-((L-n)**2.0)/2.0)
        
    Vr = np.fft.fft(vn)
    
    #do convolution
    Gr = Yr * Vr
    
    gk = np.fft.ifft(Gr)
    #gk = np.convolve(yn, vn)
    
    #add phase factor
    Xk = np.zeros(M, dtype=np.complex128)
    for k in range(M):
        g_scale = W**((k**2.0)/2.0) 
        Xk[k] = g_scale * gk[k]
        
    return Xk

def fchirpz2d(x, M, A, W):
    """
    chirp z transform per Rabiner derivation pp1256
    x is our (complex) signal of length N
    assume x is square, output M will be square, dims are the same on all sides
    
    
    """
    N = len(x)
    L = 2**(nextpow2(N + M -1))  # or nearest power of two
    yn = np.zeros((L, L), dtype=np.complex128)
    ns = np.arange(N)
    ms = np.arange(M)
    
    yn_scale =  A**(-ns) * W**((ns**2.0)/2.0)
    
    a = np.outer(yn_scale, yn_scale)

    yn[:N, :N] = x * np.outer(yn_scale, yn_scale)

    Yr = np.fft.fft2(yn)
    
    vn = np.zeros(L, dtype=np.complex128)
    for n in range(M):
        vn[n] = W**((-n**2.0)/2.0)
        
    for n in range(L-N+1, L):
        vn[n] = W**(-((L-n)**2.0)/2.0)
        
    Vr = np.fft.fft2(np.outer(vn, vn))
    
    Gr = Yr * Vr
    
    gk = np.fft.ifft2(Gr)
    
   
    Xk = W**((ms**2.0)/2.0) 
        
    return gk[:M, :M] * np.outer(Xk, Xk)

def zoom_fft(x, theta_start, step_size, M):
    """
    "zoomed" version of the fft, produces M step_sized samples
    around the unit circle starting at theta_start
    
    """
    A = np.exp(1j * theta_start)
    W = np.exp(-1j * step_size)
    
    return chirpz(x, M, A, W)

def zoom_fft2(x, theta_start, step_size, M):
    """
    "zoomed" version of the fft2, produces M step_sized samples
    around the unit circle starting at theta_start
    
    """
    A = np.exp(1j * theta_start)
    W = np.exp(-1j * step_size)
    
    return fchirpz2d(x, M, A, W)