import os
import h5py
from glob import glob

import numpy as np
from PIL import Image

files = glob('reconstructions/*.h5')

for i in files:
    filename = i.split('.')[0]
    hf = h5py.File(i)
    # hf.keys() reconstructions
    recon = hf['reconstruction']
    for k, j in enumerate(recon):
        j = j - j.min()
        j = j / j.max() * 255
        im = Image.fromarray(np.squeeze(j)).convert("L")
        if not os.path.exists(filename):
            os.mkdir(filename)
        im.save(i.split('.')[0] + '/' + str(k) +'.png')
    