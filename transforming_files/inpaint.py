#!/usr/bin/env python

'''
Inpainting sample.
Inpainting repairs damage to images by floodfilling
the damage with surrounding image areas.
Usage:
  inpaint.py [<image>]
Keys:
  SPACE - inpaint
  r     - reset the inpainting mask
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from cv2 import cv2

from common import Sketcher

def main():
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'fruits.jpg'

    img = cv2.imread(cv2.samples.findFile(fn))
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    sketch = Sketcher('img', [img_mark, mark], lambda : ((255, 0, 0), 50))

    while True:
        ch = cv2.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            res = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_TELEA)
            cv2.imshow('inpaint', res)
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            sketch.show()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()