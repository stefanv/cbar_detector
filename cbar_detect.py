from skimage import measure, io, color, img_as_float, filters
import matplotlib.pyplot as plt
import numpy as np


img = img_as_float(io.imread('test_figs/ligo.jpg'))
#mask = color.deltaE_ciede2000(img, (1, 1, 1)) > 0.2
mask = filters.canny(color.rgb2gray(img))
labels = measure.label(mask)
regions = measure.regionprops(labels)

def bbox_area(bbox):
    r0, c0, r1, c1 = bbox
    return abs((r0 - r1) * (c0 - c1))

regions = [r for r in regions if r.convex_area / bbox_area(r.bbox) > 0.9]
regions = [r for r in regions if r.area > 200]

out = np.zeros_like(img)
for r in regions:
    out[r.coords[:, 0], r.coords[:, 1]] = r.convex_area / bbox_area(r.bbox)

#plt.imshow((labels * 1123) % 87, cmap='spectral')
plt.imshow(out, cmap='gray')
plt.colorbar()
plt.show()
