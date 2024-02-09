import cv2
import numpy as np

def combine_images(img_dict):
    # Properties
    border_color = [200, 200, 200]  # Gray
    border_thickness = 10
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 2  # Increased font size
    title_font_thickness = 3  # Thicker font for better visibility
    title_color = [50, 50, 50]
    title_spacing = 30  # Spacing from the top border

    img_list = []

    # Convert each image in the dict to a 3-channel image and add borders and titles
    for title, img in img_dict.items():
        # Convert to 3-channel image if not
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Calculate space required for title
        text_size = cv2.getTextSize(title, title_font, title_font_scale, title_font_thickness)
        title_height = text_size[0][1] + title_spacing

        # Create space for title above the image
        title_space = np.zeros((title_height, img.shape[1], 3), dtype=np.uint8)
        title_space.fill(255)  # Fill with white

        # Position of title in the new space
        text_x = (title_space.shape[1] - text_size[0][0]) // 2
        text_y = text_size[0][1] + (title_spacing // 2)

        # Add title to the space
        cv2.putText(title_space, title, (text_x, text_y), title_font, title_font_scale, title_color, title_font_thickness, cv2.LINE_AA)

        # Concatenate title space and image vertically
        img_with_title = cv2.vconcat([title_space, img])

        # Add border
        bordered_img = cv2.copyMakeBorder(img_with_title, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
        
        img_list.append(bordered_img)

    # Ensure all images have the same height
    max_height = max(img.shape[0] for img in img_list)
    for idx, img in enumerate(img_list):
        if img.shape[0] != max_height:
            img_list[idx] = cv2.resize(img, (img.shape[1], max_height))

    # Concatenate all the images horizontally
    combined_img = cv2.hconcat(img_list)
    
    return combined_img