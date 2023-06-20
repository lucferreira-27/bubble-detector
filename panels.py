from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from scipy import ndimage as ndi
from skimage.color import label2rgb, rgb2gray
from skimage.feature import canny
from skimage.measure import label, regionprops, approximate_polygon
from skimage.morphology import dilation
from skimage import measure
import json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(console)


def process_image(image_path, output_dir = './outputs/panels/'):
    """Processes an image by detecting and labeling panels."""
    logging.info(f"Processing image: {image_path}")

    # Create directories for output files
    # image_dir = os.path.join(output_dir, os.path.basename(image_path))
    # process_dir = os.path.join(image_dir, 'process')
    os.makedirs(output_dir, exist_ok=True)

    # Load image and convert to grayscale
    im = load_image(image_path)
    grayscale = convert_to_grayscale(im)

    # Detect edges with Canny and thicken edges with dilation
    edges = detect_edges(grayscale)
    thick_edges = thicken_edges(edges)

    # Fill holes in the image and label each patch
    colors = get_colors(num_colors=99)
    labels = fill_holes(thick_edges)
    print(len(labels))
    label_rgb_image = label_patches(im, labels, colors)

    # Draw custom shapes and labels for each patch
    labeled_image = draw_patches(im, label_rgb_image, labels, colors)

    # Save labeled image with text
    # labeled_image.save(os.path.join(process_dir, 'labeled_with_text.jpg'))

    # Extract and order panels
    panels, vertices = extract_panels(labels, im)

    # Display ordered images in a grid
    ordered_images = order_panels(
        im, edges, thick_edges, label_rgb_image, labels, panels)

    # Create a list of dictionaries containing the panel bounding box and vertices
    panels_with_vertices = [{"bbox": list(map(
        int, panel)), "vertices": vertex_set} for panel, vertex_set in zip(panels, vertices)]

    # Save the panel information as a JSON file
    print(f'{output_dir}/panels-{os.path.basename(image_path)}.json')
    with open(f'{output_dir}/panels-{os.path.basename(image_path)}.json', 'w') as f:
        json.dump(panels_with_vertices, f)

    # save_images(ordered_images, process_dir)
    logging.info(f"Processing of {image_path} finished.")


def save_images(list_images, output_dir):
    """Saves each image in list_images to a file in the output_dir directory."""
    for i, img in enumerate(list_images):
        filename = f"image_{i}.jpg"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img).save(filepath)


def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=False, num_cols=2, figsize=(200, 100), title_fontsize=30):
    logging.info("Processing image: Displaying image grid")

    def img_is_color(img):
        if len(img.shape) == 3:
            # Check the color channels to see if they're all the same.
            c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            if (c1 == c2).all() and (c2 == c3).all():
                return True

        return False

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (
            len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (
            len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + \
        (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (
            None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    plt.show()


def show_image_grid(list_images, num_cols=3, figsize=(20, 20)):
    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + \
        (1 if num_images % num_cols != 0 else 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx >= num_images:
                break
            img = list_images[idx]
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

    fig.tight_layout()
    plt.show()


def load_image(image_path):
    """Loads an image using the ImageIO library."""
    logging.info("Processing image: Loading image")
    return imageio.v2.imread(image_path)


def convert_to_grayscale(image):
    """Converts an image to grayscale using the RGB2GRAY function from the skimage.color module."""
    logging.info("Processing image: Converting image to grayscale")
    if len(image.shape) == 3:
        # Convert to grayscale if image has three channels
        image = image[:, :, :3]
        return rgb2gray(image)
    else:
        # Use image as grayscale if it has only two channels
        return image


def detect_edges(image):
    """Detects edges in an image using the Canny function from the skimage.feature module."""
    logging.info("Processing image: Detecting edges")
    return canny(image)


def thicken_edges(edges):
    """Thickens edges in an image using dilation from the skimage.morphology module."""
    logging.info("Processing image: Thickening edges")
    return dilation(dilation(edges), np.ones((3, 3), dtype=bool))


def fill_holes(thick_edges):
    """Fills holes in an image using label and binary_fill_holes from the skimage.measure and scipy.ndimage modules."""
    logging.info("Processing image: Filling holes")
    labels = label(thick_edges)
    props = regionprops(labels)
    max_size = max([prop.area for prop in props])
    size_threshold = max_size * 0.05
    filled_edges = np.zeros_like(thick_edges)

    for prop in props:
        if prop.area >= size_threshold:
            filled_edges |= ndi.binary_fill_holes(labels == prop.label)

    return label(filled_edges)


def get_colors(num_colors=99):
    logging.info("Processing image: Getting colors")
    color_map = plt.get_cmap("Set1", num_colors)
    colors = [color_map(i)[:3] for i in range(num_colors)]
    return colors


def label_patches(image, labels, colors=None):
    """Labels patches in an image using label2rgb from the skimage.color module."""
    logging.info("Processing image: Labeling patches")
    if not colors:
        colors = get_colors()
    label_rgb = label2rgb(labels, image=image, colors=colors)
    return Image.fromarray((label_rgb * 255).astype(np.uint8))


def draw_patches(im, image, labels, colors):
    """Draws custom shapes and labels for each patch in an image."""
    logging.info("Processing image: Drawing patches")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./fonts/arial.ttf", 160)
    regions = regionprops(labels)
    size_limit = 1000
    masks = []
    label_number = 1

    for idx, region in enumerate(regions):
        # Create an empty binary mask for the region with the same shape as the original image
        area = region.area
        bbox = region.bbox
        if area < size_limit:
            continue
        mask = np.zeros_like(labels, dtype=bool)
        coords = region.coords

        # Set the corresponding pixels in the binary mask to True
        mask[coords[:, 0], coords[:, 1]] = True

        # Add the binary mask to the list of masks
        masks.append(mask)

        # Apply the mask to the original image

        masked_image = np.zeros_like(im)
        masked_image[mask] = im[mask]

        # Draw label text for the patch
        bbox = draw.textbbox((0, 0), str(label_number), font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        mask_indices = np.argwhere(mask)
        min_row, min_col = np.min(mask_indices, axis=0)
        max_row, max_col = np.max(mask_indices, axis=0)

        tx = (min_col + max_col - w) / 2
        ty = (min_row + max_row - h) / 2
        draw.text((tx, ty), str(label_number), font=font, fill=(255, 255, 255))

        # Find the contours of the mask and extract the outer contour
        contours = measure.find_contours(mask, 0.5)
        outer_contour = None
        max_size = 0

        for contour in contours:
            if len(contour) > max_size:
                outer_contour = contour
                max_size = len(contour)

        # Define custom shape vertices using the outer contour
        shape_vertices = [(col, row) for row, col in outer_contour.astype(int)]

        print(len(colors), label_number)
    
        
        # Draw custom shape
        draw.polygon(shape_vertices, outline=tuple(round(x * 255)
                     for x in colors[label_number - 1]), width=10)
        label_number = label_number + 1

    return image


def extract_panels(labels, image):
    """Extracts panels from an image using regionprops from the skimage.measure module."""
    logging.info("Extracting panels AI")
    panels = []
    regions = regionprops(labels)

    def create_masks(regions, image):
        masks = []
        for idx, region in enumerate(regions):
            mask = np.zeros_like(labels, dtype=bool)
            coords = region.coords

            # Set the corresponding pixels in the binary mask to True
            mask[coords[:, 0], coords[:, 1]] = True

            # Add the binary mask to the list of masks
            masks.append(mask)

            # Apply the mask to the original image
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]
        return masks

    def get_shape_vertices(masks):
        shape_vertices_list = []
        for mask in masks:
            # Find contours of mask
            contours = measure.find_contours(mask, 0.5)

            # Extract outer contour (assume only one contour per mask)
            contour = contours[0]

            # Convert contour coordinates to np.array
            contour_array = np.array([[int(x), int(y)] for y, x in contour])

            # Simplify the contour using skimage.measure.approximate_polygon
            simplified_contour = approximate_polygon(
                contour_array, tolerance=2.5)

            # Convert simplified contour coordinates to vertices
            vertices = [(int(x), int(y)) for x, y in simplified_contour]

            shape_vertices_list.append(vertices)

        return shape_vertices_list

    def is_panel_inside(bbox1, bbox2, threshold=0.7):
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2

        inter_area = max(0, min(y1_max, y2_max) - max(y1_min, y2_min)) * \
            max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        area1 = (y1_max - y1_min) * (x1_max - x1_min)
        area2 = (y2_max - y2_min) * (x2_max - x2_min)

        ratio1 = inter_area / area1
        ratio2 = inter_area / area2
        return ratio1 >= threshold or ratio2 >= threshold

    masks = create_masks(regions, image)
    bboxs = []
    for mask in masks:
        mask_indices = np.argwhere(mask)
        x1, y1 = np.min(mask_indices, axis=0)
        x2, y2 = np.max(mask_indices, axis=0)
        bboxs.append([x1, y1, x2, y2])

    vertices = get_shape_vertices(masks)
    panel_regions = []
    print(vertices)
    for region in regions:
        area = region.area
        coords = region.coords
        bbox = (np.min(coords[:, 0]), np.min(coords[:, 1]),
                np.max(coords[:, 0]), np.max(coords[:, 1]))
        # bbox = region.bbox
        # Check if the current panel is inside another panel
        merge_with = -1
        for i, other_bbox in enumerate(panels):
            if is_panel_inside(bbox, other_bbox):
                merge_with = i
                break

        # Merge the panels
        if merge_with != -1:
            y_min, x_min, y_max, x_max = panels[merge_with]
            y_min_new, x_min_new, y_max_new, x_max_new = bbox
            panels[merge_with] = (min(y_min, y_min_new), min(x_min, x_min_new),
                                  max(y_max, y_max_new), max(x_max, x_max_new))
            panel_regions[merge_with] = region
        else:
            panels.append(bbox)
            panel_regions.append(region)
    # Remove small panels
    for i, bbox in reversed(list(enumerate(panels))):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print(area)
        if area < 0.01 * image.shape[0] * image.shape[1] or area < 110000.0:
            del panel_regions[i]
            del panels[i]
    return panels, vertices


def order_panels(image, edges, thick_edges, label_rgb_image, labels, panels):
    """Orders panels in an image by creating a list of images showing each panel separately."""
    logging.info("Processing image: Ordering panels")
    panel_img = np.zeros_like(labels)

    for i, bbox in enumerate(panels, start=1):
        panel_img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = i

    panel_img = Image.fromarray(
        (label2rgb(panel_img, bg_label=0) * 255).astype('uint8'))

    # Create list of images to show
    list_images = [np.array(image), np.array(edges), np.array(thick_edges), np.array(label_rgb_image),
                   np.array(panel_img)]
    panel_images = []

    list_images.extend(panel_images)
    # show_image_list(list_images, figsize=(10, 10))
    return list_images


# process_image('E:/Projects/bubble-capture/bubble-detector/images/One Piece v1-118.jpg')


def get_panels_labels(image_dir,panel_dir):
    for filename in os.listdir(image_dir):
        if not filename.endswith('.png') and not filename.endswith('.jpg'):
            continue
        image_path = os.path.join(image_dir, filename)
        process_image(image_path,panel_dir)
