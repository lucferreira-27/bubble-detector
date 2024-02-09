import numpy as np
from skimage import io, color, morphology, feature, measure
from scipy import ndimage as ndi
import cv2
from shapely.geometry import Polygon,box,MultiPolygon


def is_bubble_inside_panel(bubble, vertices):
    panel_polygon = Polygon(vertices)
    bubble_polygon = box(bubble[0], bubble[1], bubble[2], bubble[3])

    intersection_area = panel_polygon.intersection(bubble_polygon).area
    bubble_area = bubble_polygon.area

    return intersection_area / bubble_area > 0.6

def join_panels_bubbles(bubbles, panels):
    joined = []
    for panel in panels:
        vertices = panel["vertices"]
        panel_bubbles = []
        for bubble in bubbles:
            if is_bubble_inside_panel(bubble['xyxy'], vertices):
                panel_bubbles.append(bubble)
        joined.append({'panel': panel, 'bubbles': panel_bubbles})
    return joined

def draw_bounding_boxes(image, vertices):
 print(f"[DRAW BLOCKS] -> Drawing bounding boxes on the image")
 # Create a copy of the image to draw on
 image_copy = image.copy()

 # Create a semi-transparent black overlay
 overlay = np.zeros_like(image_copy)
 alpha = 0.4
 cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)

 # Define colors for the bounding boxes
 colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

 # Create a mask with the same size as the image
 mask = np.zeros_like(image_copy)

 # Draw bounding boxes on the mask
 for i, vertex in enumerate(vertices):
    # Convert vertices to numpy array
    vertex = np.array(vertex).reshape((-1, 1, 2)).astype(np.int32)
    # Draw a colored polygon
    cv2.polylines(mask, [vertex], True, colors[i % len(colors)], 2)
    # Fill the interior of the polygon
    cv2.fillConvexPoly(mask, vertex, colors[i % len(colors)])

 # Blend the mask with the original image
 image_copy = cv2.addWeighted(image_copy, 1, mask, 0.5, 0)

 return image_copy


def read_image(image_path):
   print(f"[PANEL EXTRACTION] -> Reading image: {image_path}")
   return io.imread(image_path)

def convert_to_grayscale(image):
    print(f"[PANEL EXTRACTION] -> Converting image to grayscale")
    # If image is grayscale, return it as is
    if len(image.shape) == 2:
        return image
    # If image is RGBA, convert to RGB by discarding the alpha channel
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    # Convert RGB to grayscale
    return color.rgb2gray(image)

def detect_edges(gray_image):
   print(f"[PANEL EXTRACTION] -> Detecting edges in image")
   return feature.canny(gray_image, sigma=2)

def dilate_edges(edges):
   print(f"[PANEL EXTRACTION] -> Dilating edges in image")
   return morphology.dilation(edges, morphology.disk(3))

def fill_holes(dilated_edges):
   print(f"[PANEL EXTRACTION] -> Filling holes in image")
   return ndi.binary_fill_holes(dilated_edges)

def label_regions(filled_image):
   print(f"[PANEL EXTRACTION] -> Labeling regions in image")
   return measure.label(filled_image)

def filter_bboxes(label_image,image):
   print(f"[PANEL EXTRACTION] -> Filtering bounding boxes in image")
   regions = measure.regionprops(label_image)
   bboxes = [region.bbox for region in regions]
   min_area = 0.01 * np.prod(image.shape[:2])
   return [bbox for bbox in bboxes if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) >= min_area]

def sort_bboxes(bboxes):
   print(f"[PANEL EXTRACTION] -> Sorting bounding boxes in image")
   return sorted(bboxes, key=lambda x: (x[0], x[1]))


def remove_intersection_area(bboxes):
    polygons = [Polygon([(bbox[1], bbox[0]), (bbox[3], bbox[0]), (bbox[3], bbox[2]), (bbox[1], bbox[2])]) for bbox in bboxes]
    for i, poly1 in enumerate(polygons):
        for j, poly2 in enumerate(polygons):
            if i != j and poly1.intersects(poly2):
                intersection = poly1.intersection(poly2)
                if intersection.area > 0:
                    # Subtract the intersection from the larger polygon
                    if poly1.area > poly2.area:
                        new_poly1 = poly1.difference(intersection)
                        if isinstance(new_poly1, Polygon):
                            polygons[i] = new_poly1
                        elif isinstance(new_poly1, MultiPolygon):
                            # Access the geoms property for MultiPolygon
                            polygons[i] = max(new_poly1.geoms, key=lambda p: p.area)
                    else:
                        new_poly2 = poly2.difference(intersection)
                        if isinstance(new_poly2, Polygon):
                            polygons[j] = new_poly2
                        elif isinstance(new_poly2, MultiPolygon):
                            # Access the geoms property for MultiPolygon
                            polygons[j] = max(new_poly2.geoms, key=lambda p: p.area)

    # Extract the vertices of the resulting polygons
    vertices_list = []
    for poly in polygons:
        if isinstance(poly, Polygon):
            vertices_list.append(list(poly.exterior.coords))
        elif isinstance(poly, MultiPolygon):
            # In case of MultiPolygon, add the largest Polygon's vertices
            largest_poly = max(poly.geoms, key=lambda p: p.area)
            vertices_list.append(list(largest_poly.exterior.coords))
    return vertices_list


def distance(p1, p2):
   return abs(p1['panel']['bbox'][0] - p2['panel']['bbox'][0])

#RIGHT-LEFT, TOP TO BOTTOM
def sort_by_read(blocks, x_margin=60):
    # Helper function to determine if the X difference is significant
    def x_significant(p1, p2):
        return abs(p1['panel']['bbox'][2] - p2['panel']['bbox'][2]) > x_margin

    # Sort the blocks based on their y-coordinate primarily, and x-coordinate conditionally
    blocks = sorted(blocks, key=lambda p: (-p['panel']['bbox'][1], 
                                           p['panel']['bbox'][0] if x_significant(p, blocks[0]) else 0))

    # Sort the bubbles within each block
    for block in blocks:
        block['bubbles'] = sorted(block['bubbles'], key=lambda b: (-b['xyxy'][0], b['xyxy'][1]))

    return blocks

def extract_panels(image_path):
   image = read_image(image_path)
   gray_image = convert_to_grayscale(image)
   edges = detect_edges(gray_image)
   dilated_edges = dilate_edges(edges)
   filled_image = fill_holes(dilated_edges)
   label_image = label_regions(filled_image)
   bboxes = filter_bboxes(label_image,image)
   bboxes = sort_bboxes(bboxes)
   vertices = remove_intersection_area(bboxes)
   new_bboxes = []
   for vertex in vertices:
      x_coords = [point[0] for point in vertex]
      y_coords = [point[1] for point in vertex]
      bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
      new_bboxes.append(bbox)
   print(f"[PANEL EXTRACTION] -> Extracted {len(vertices)} panels from image: {image_path}")
   return  [{"vertices": v, "bbox": b} for v, b in zip(vertices, new_bboxes)]





if __name__ == "__main__":
    image_path = "demo.jpg"
    image = read_image(image_path)
    vertices = extract_panels(image)

    image_with_boxes = draw_bounding_boxes(image, vertices)
    cv2.imwrite('demo_image_with_boxes.png', image_with_boxes)
