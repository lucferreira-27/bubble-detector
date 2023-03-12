from PIL import Image
import imageio
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from PIL import ImageFont
from PIL import ImageDraw


# Load image
im = imageio.imread('One Piece v1-044.jpg')

# Convert to grayscale
grayscale = rgb2gray(im)

# Save grayscale image
Image.fromarray((grayscale * 255).astype('uint8'), 'L').save('gray.jpg')

# Detect edges with Canny
edges = canny(grayscale)

# Save edges image
Image.fromarray(edges.astype('uint8')*255, 'L').save('edges.jpg')


# Thicken edges with dilation
#thick_edges = dilation(dilation(edges),  np.ones((32,32), dtype=bool))
#thick_edges = dilation(dilation(edges),  np.ones((5,5), dtype=bool))
#thick_edges = dilation(dilation(edges),  np.ones((8,8), dtype=bool))
thick_edges = dilation(dilation(edges),  np.ones((3,3), dtype=bool))

# Save thick edges image
Image.fromarray(thick_edges.astype('uint8')*255, 'L').save('thick_edges.jpg')


# Fill holes in the image
filled_edges = ndi.binary_fill_holes(thick_edges)

# Save filled edges image
Image.fromarray(filled_edges.astype('uint8')*255, 'L').save('filled_edges.jpg')

# Label each patch in the image
labels = label(filled_edges)
print(len(labels))

# Save labeled image
label_hues = np.uint8(179 * labels / np.max(labels))
label_rgb = label2rgb(labels, image=im, colors=[(0.8,0,0), (0,0.8,0), (0,0,0.8)])
Image.fromarray((label_rgb * 255).astype(np.uint8)).save('labeled.jpg')

# Group patches into panels
regions = regionprops(labels)
print(len(regions))

panels = []


def do_bboxes_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap.
    
    Parameters:
    bbox1 (tuple): First bounding box as a tuple of (min_row, min_col, max_row, max_col).
    bbox2 (tuple): Second bounding box as a tuple of (min_row, min_col, max_row, max_col).
    
    Returns:
    bool: True if the two bounding boxes overlap, False otherwise.
    """
    overlap_rows = (bbox1[2] > bbox2[0]) and (bbox2[2] > bbox1[0])
    overlap_cols = (bbox1[3] > bbox2[1]) and (bbox2[3] > bbox1[1])
    return overlap_rows and overlap_cols

def merge_bboxes(bbox1, bbox2):
    """Merge two bounding boxes into a single bounding box."""
    return (min(bbox1[0], bbox2[0]),    # ymin
            min(bbox1[1], bbox2[1]),    # xmin
            max(bbox1[2], bbox2[2]),    # ymax
            max(bbox1[3], bbox2[3]))    # xmax

for region in regions:
    for i, panel in enumerate(panels):
        if do_bboxes_overlap(region.bbox, panel):
            panels[i] = merge_bboxes(panel, region.bbox)
            break
    else:
        panels.append(region.bbox)

# Remove small panels
for i, bbox in reversed(list(enumerate(panels))):
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area < 0.01 * im.shape[0] * im.shape[1]:
        del panels[i]

# Create panel image
panel_img = np.zeros_like(labels)

for i, bbox in enumerate(panels, start=1):
    panel_img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = i

# Save panel image
Image.fromarray((label2rgb(panel_img, bg_label=0) * 255).astype('uint8')).save('panels.jpg')

# Order panels
def are_bboxes_aligned(a, b, axis):
    return (
        a[0 + axis] < b[2 + axis] and
        b[0 + axis] < a[2 + axis]
    )

def cluster_bboxes(bboxes, axis=0):
    clusters = []
    for bbox in bboxes:
        for cluster in clusters:
            if any(
                are_bboxes_aligned(b, bbox, axis=axis)
                for b in cluster
            ):
                cluster.append(bbox)
                break
        else:
            clusters.append([bbox])

    clusters.sort(key=lambda c: c[0][0 + axis])

    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            clusters[i] = cluster_bboxes(
                bboxes=cluster,
                axis=1 if axis == 0 else 0
            )

    return clusters

clusters = cluster_bboxes(panels)

# Order panels
def are_bboxes_aligned(a, b, axis):
    return (
        a[0 + axis] < b[2 + axis] and
        b[0 + axis] < a[2 + axis]
    )

def cluster_bboxes(bboxes, axis=0):
    clusters = []
    for bbox in bboxes:
        for cluster in clusters:
            if any(
                are_bboxes_aligned(b, bbox, axis=axis)
                for b in cluster
            ):
                cluster.append(bbox)
                break
        else:
            clusters.append([bbox])

    clusters.sort(key=lambda c: c[0][0 + axis])

    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            clusters[i] = cluster_bboxes(
                bboxes=cluster,
                axis=1 if axis == 0 else 0
            )

    return clusters

clusters = cluster_bboxes(panels)

# Order panels
ordered_panels = []

for cluster in clusters:
    if len(cluster) == 1:
        ordered_panels.extend(cluster)
    else:
        ordered_panels.extend(sorted(cluster, key=lambda x: x[1] if len(x) > 1 else 0))

# Create image with ordered panels
ordered_panels_img = np.zeros_like(panel_img)
for i, bbox in enumerate(ordered_panels, start=1):
    ordered_panels_img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = i

# Save image with ordered panels
Image.fromarray((label2rgb(ordered_panels_img, bg_label=0) * 255).astype('uint8')).save('ordered_panels.jpg')

# Add labels to ordered panel image
ordered_panels_img_rgb = label2rgb(ordered_panels_img, bg_label=0)
draw = ImageDraw.Draw(Image.fromarray((ordered_panels_img_rgb * 255).astype('uint8')))
font = ImageFont.truetype("arial.ttf", 20)

for i, bbox in enumerate(ordered_panels, start=1):
    x = bbox[1] + 5
    y = bbox[0] + 5
    draw.text((x, y), str(i), font=font, fill=(255, 255, 255, 255))

# Save image with ordered panels and labels
Image.fromarray((ordered_panels_img_rgb * 255).astype('uint8')).save('ordered_panels_labeled.jpg')