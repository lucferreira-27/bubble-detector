# Import the required libraries
import io
import os
import json
from PIL import Image, ImageDraw, ImageFont
import math




def text_detect():
    # Load the response body from the file
    with open('One Piece - c0001 (v001) - p009 [VIZ Media] [Digital] [1r0n].png_2023-05-12_02-32_results.json') as f:
        response = json.load(f)

    # Get the text annotations from the response
    texts = response['text_blocks']

    # Define some constants for the heuristic
    DISTANCE_THRESHOLD = 120  # pixels

    # Define a class to represent a word with its coordinates and center point
    class Word:
        def __init__(self, vertices):
            self.vertices = vertices
            self.x1 = vertices[0]['x']
            self.y1 = vertices[0]['y']
            self.x2 = vertices[1]['x']
            self.y2 = vertices[1]['y']
            self.x3 = vertices[2]['x']
            self.y3 = vertices[2]['y']
            self.x4 = vertices[3]['x']
            self.y4 = vertices[3]['y']
            self.cx = (self.x1 + self.x2 + self.x3 + self.x4) / \
                4  # average x coordinate
            self.cy = (self.y1 + self.y2 + self.y3 + self.y4) / \
                4  # average y coordinate

        # Define a method to calculate the distance to another word
        def distance_to(self, other):
            return math.sqrt((other.cx - self.cx) ** 2 + (other.cy - self.cy) ** 2)

        # Define a method to check if the word is close enough to another word to be in the same block
        def is_close_to(self, other):
            return self.distance_to(other) < DISTANCE_THRESHOLD

    # Define a class to represent a block of words with its coordinates and words


    class Block:
        def __init__(self, words):
            self.words = words
            self.min_x, self.min_y, self.max_x, self.max_y = self.get_min_max()
            self.cx, self.cy = self.get_center()

        # Define a method to get the minimum and maximum coordinates of the block
        def get_min_max(self):
            min_x = min_y = math.inf
            max_x = max_y = -math.inf
            for word in self.words:
                min_x = min(min_x, word.x1, word.x2, word.x3, word.x4)
                min_y = min(min_y, word.y1, word.y2, word.y3, word.y4)
                max_x = max(max_x, word.x1, word.x2, word.x3, word.x4)
                max_y = max(max_y, word.y1, word.y2, word.y3, word.y4)
            return min_x, min_y, max_x, max_y

        # Define a method to get the center point of the block
        def get_center(self):
            cx = (self.min_x + self.max_x) / 2  # average x coordinate of the block
            cy = (self.min_y + self.max_y) / 2  # average y coordinate of the block
            return cx, cy


    # Convert the texts into Word objects
    # skip the first element which is the whole text
    words = [Word(text['vertices']) for text in texts[1:]]

    # Group the words into blocks using the union-find algorithm
    parents = list(range(len(words)))  # initialize each word as its own parent
    ranks = [0] * len(words)  # initialize each word with rank 0


    def find_parent(parents, i):
        if parents[i] == i:
            return i
        else:
            return find_parent(parents, parents[i])


    def union(parents, ranks, i, j):
        i_root = find_parent(parents, i)
        j_root = find_parent(parents, j)
        if i_root != j_root:
            if ranks[i_root] < ranks[j_root]:
                parents[i_root] = j_root
            elif ranks[i_root] > ranks[j_root]:
                parents[j_root] = i_root
            else:
                parents[j_root] = i_root
                ranks[i_root] += 1


    # Define a function to sort the blocks by their reading order
    def sort_blocks(blocks):
        # Initialize an empty list to store the sorted blocks
        sorted_blocks = []

        # Loop until all blocks are sorted
        while blocks:
            # Find the highest block among the remaining blocks
            highest_block = min(blocks, key=lambda block: block.min_y)

            # Find all the blocks that are on the same row as the highest block
            same_row = [block for block in blocks if block.min_y <=
                        highest_block.max_y]

            # Sort the blocks on the same row by their rightmost x-coordinate in descending order
            same_row.sort(key=lambda block: -block.max_x)

            # Append the blocks on the same row to the sorted list
            sorted_blocks.extend(same_row)

            # Remove the blocks on the same row from the remaining blocks
            for block in same_row:
                blocks.remove(block)

        # Return the sorted list of blocks
        return sorted_blocks


    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if words[i].is_close_to(words[j]):
                union(parents, ranks, i, j)

    blocks_dict = {}
    for i in range(len(words)):
        parent = find_parent(parents, i)
        if parent not in blocks_dict:
            blocks_dict[parent] = []
        blocks_dict[parent].append(words[i])

    blocks_list = [Block(words) for words in blocks_dict.values()]

    # Sort the blocks by their center points from top-right to bottom-left
    blocks_list = sort_blocks(blocks_list)

    # Draw rectangles and numbers around the blocks
    im = Image.open(
        'One Piece - c0001 (v001) - p009 [VIZ Media] [Digital] [1r0n].png')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", 32)  # load a font for drawing numbers

    for i, block in enumerate(blocks_list):
        draw.rectangle([block.min_x, block.min_y, block.max_x,
                    block.max_y], outline='red')
        draw.text((block.cx, block.cy), str(i + 1), fill='red', font=font,
                anchor='mm')  # draw the number in the middle of the block

    im.show()

text_detect()