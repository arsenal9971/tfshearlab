# Import libraries
import numpy as np
import demandimport

with demandimport.enabled():
    import matplotlib.pyplot as plt
    from itertools import compress
    import tensorflow as tf

__all__ = ('tfftshift','itfftshift', 'tfsheardec2D', 'tfshearrec2D', 'tfsheardecadjoint2D', 'tfshearrecadjoint2D')

# tfftshift and itfftshift

def tfftshift(xtf, axes=None):
    if len(xtf.shape)==3:
        ndim = len(xtf.shape)-1
    else:
        ndim = len(xtf.shape)-2
    if axes is None:
        axes = list(np.array(range(ndim))+1)
    elif isinstance(axes, integer_types):
        axes = (axes,)
    ytf = xtf
    for k in axes:
        n = int(ytf.shape[k])
        p2 = (n+1)//2
        mylist = np.concatenate((np.arange(p2, n), np.arange(p2)))
        ytf = tf.gather(ytf, indices = mylist, axis = k)
    return ytf

def itfftshift(ytf, axes=None):
    if len(ytf.shape)==3:
        ndim = len(ytf.shape)-1
    else:
        ndim = len(ytf.shape)-2
    if axes is None:
        axes = list(np.array(range(ndim))+1)
    elif isinstance(axes, integer_types):
        axes = (axes,)
    xtf = ytf
    for k in axes:
        n = int(xtf.shape[k])
        p2 = n-(n+1)//2
        mylist = np.concatenate((np.arange(p2, n), np.arange(p2)))
        xtf = tf.gather(xtf, indices = mylist, axis = k)
    return xtf

# shearlet decomposition
def tfsheardec2D(Xtf, tfshearlets):
    """Shearlet Decomposition function."""
    Xfreqtf = tfftshift(tf.fft2d(itfftshift(Xtf)))
    return tfftshift(tf.transpose(tf.ifft2d(tf.transpose(itfftshift(tf.multiply(tf.expand_dims(Xfreqtf,3),
                                                                                        tf.conj(tfshearlets))),[0,3,1,2])),[0,2,3,1]))

# shearlet reconstruction
def tfshearrec2D(coeffstf, tfshearlets,tfdualFrameWeights ):
    Xfreqtf = tf.reduce_sum(tf.multiply(tfftshift(tf.transpose(tf.fft2d(tf.transpose(itfftshift(coeffstf),[0,3,1,2])),[0,2,3,1])),
                                        tfshearlets),axis=3)
    return tfftshift(tf.ifft2d(itfftshift(tf.multiply(Xfreqtf,1/tfdualFrameWeights))))

# dual of decomposition
def tfsheardecadjoint2D(coeffstf, tfshearlets,tfdualFrameWeights):
    Xfreqtf = tf.reduce_sum(tf.multiply(tfftshift(tf.transpose(tf.fft2d(tf.transpose(itfftshift(coeffstf),[0,3,1,2])),[0,2,3,1])),
                                        tf.conj(tfshearlets)),axis=3)
    return tfftshift(tf.ifft2d(itfftshift(tf.multiply(Xfreqtf,1/tfdualFrameWeights))))

# dual of reconstruction
def shearrecadjoint2D(Xtf, tfshearlets):
    """Shearlet Decomposition function."""
    Xfreqtf = tfftshift(tf.fft2d(itfftshift(Xtf)))
    return tfftshift(tf.transpose(tf.ifft2d(tf.transpose(itfftshift(tf.multiply(tf.expand_dims(Xfreqtf,3),
                                                                                        (tfshearlets))),[0,3,1,2])),[0,2,3,1]))
