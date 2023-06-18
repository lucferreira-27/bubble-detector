# Assuming you have a json file named data.json with the xyxy coordinates
import json
from PIL import Image, ImageDraw

# Read the json file and parse it
with open('opepiece_t_1063_011.json', 'r') as f:
  data = f.read()
  result = json.loads(data)

# Load the image from a file or a url
image = Image.open('opepiece_t_1063_011.png')

# Create a draw object for the image
draw = ImageDraw.Draw(image)

# Set the stroke style for the rectangles
color = 'red'
width = 5

# Loop through the xyxy coordinates and draw rectangles
vertices = [d['xyxy'] for d in result]
for xyxy in vertices:
  x1 = xyxy[0]
  y1 = xyxy[1]
  x2 = xyxy[2]
  y2 = xyxy[3]
  draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

# Save the image as a png file
image.save('output.png')
print('The PNG file was created.')
