import os
from pathlib import Path

import numpy as np

from jarvis_gui.utils.hsi_save_load import load_hsi, save_hsi


class SpecimStitcher:
    def __init__(self, load_path, save_path, dark):
        self.all_lines = []
        self.load_path = Path(load_path)
        self.save_path = Path(save_path)
        self.dark = dark
        self.bad_lines = 0
    
    def load_img(self, fname, w=640, c=224):
        img = np.load(fname)
        img = load_hsi(img)
        img = img.reshape((-1, c, w))
        img = img.swapaxes(1, 2)
        img = img[:, ::-1]
        img = img.swapaxes(0, 1)
        self.img = img

    def load_line(self, line, w=640, c=224):
        """
        args:
            line: an array of shape (width, channels) and dtype np.uint8 containing the lines to be stitched
        """
        try:
            line = load_hsi(line)
            line = line.reshape([c, w]).swapaxes(0, 1)[::-1]
        except:
            line = np.zeros(shape=(w, c), dtype=np.uint16)
            self.bad_lines += 1
        self.all_lines.append(line)

    def load_lines(self):
        files = os.listdir(self.load_path)
        files = sorted(files)#, key=lambda name: os.path.getmtime(self.path / name)) # Is this sorting correct? Or should neighbouring pairs be swapped?
        if all(map(lambda f: f[-4:] == ".bin", files)):
            print("Loading binary files")
            arrs = [np.fromfile(f"{self.load_path}/{f}", dtype=np.uint8) for f in files]
        elif all(map(lambda f: f[-4:] == ".npy", files)):
            print("Loading NumPy arrays")
            arrs = [np.load(f"{self.load_path}/{f}") for f in files]
        else:
            raise ValueError(
                "Did not find exclusively .bin or .npy files in the directory."
            )
        print(f'Found {len(arrs)} lines.')
        for i, a in enumerate(arrs):
            # if a.shape[0] != int(107520 / 2):
            #     print(f'Skipped {i}')
            #     continue
            self.load_line(a)
        print(f'{self.bad_lines} bad lines.')
        # self.img = np.stack(self.all_lines, axis=1)
        self.stitch_lines()
        hsi_img = save_hsi(self.img)
        save_path = self.save_path / 'raw_hsi_img_mono12p.npy' if not self.dark else self.save_path / 'raw_hsi_dark_mono12p.npy'
        np.save(save_path, hsi_img)
        for f in files:
            os.remove(self.load_path / f)
    
    def stitch_lines(self):
        self.img = np.stack(self.all_lines, axis=1)