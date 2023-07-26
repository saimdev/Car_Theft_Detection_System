# import cv2
# import numpy as np
# import os

# # Set the path to your car images dataset directory
# dataset_path = './dataset'

# # Set the path to save the annotated dataset
# annotated_dataset_path = './annotated'

# # Create the annotated dataset directory if it doesn't exist
# os.makedirs(annotated_dataset_path, exist_ok=True)

# # Load the car images from the dataset directory
# car_images = []
# for root, dirs, files in os.walk(dataset_path):
#     for file in files:
#         # Check if the file is an image
#         if file.endswith('.jpg') or file.endswith('.png'):
#             # Load the image
#             image_path = os.path.join(root, file)
#             image = cv2.imread(image_path)
#             car_images.append(image)

# # Global variables for mouse events
# annotations = []
# current_annotation = {'x': -1, 'y': -1, 'width': -1, 'height': -1}
# annotation_in_progress = False

# # Mouse event callback function
# def annotate_cars(event, x, y, flags, param):
#     global current_annotation, annotation_in_progress, annotated_image

#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Start drawing a new bounding box
#         current_annotation = {'x': x, 'y': y, 'width': -1, 'height': -1}
#         annotation_in_progress = True
#     elif event == cv2.EVENT_LBUTTONUP:
#         # Finish drawing the bounding box
#         current_annotation['width'] = x - current_annotation['x']
#         current_annotation['height'] = y - current_annotation['y']
#         annotations.append(current_annotation.copy())
#         annotation_in_progress = False

#         # Draw the annotated bounding box on the image
#         cv2.rectangle(annotated_image, (current_annotation['x'], current_annotation['y']),
#                       (current_annotation['x'] + current_annotation['width'],
#                        current_annotation['y'] + current_annotation['height']),
#                       (0, 255, 0), 2)

# # Display the images and prompt for annotation
# for i, image in enumerate(car_images):
#     annotated_image = image.copy()

#     # Create a named window and set the mouse callback function
#     cv2.namedWindow('Annotate Cars')
#     cv2.setMouseCallback('Annotate Cars', annotate_cars)

#     while True:
#         # Display the image
#         cv2.imshow('Annotate Cars', annotated_image)

#         # Prompt for annotation
#         key = cv2.waitKey(1) & 0xFF

#         # Annotation process
#         if key == ord('q'):
#             # Quit annotation
#             break
#         elif key == ord('c'):
#             # Continue to the next image
#             break
#         elif key == ord('r'):
#             # Reset the annotation for the current image
#             annotated_image = image.copy()
#             if annotations:
#                 annotations.pop()

#     if key == ord('q'):
#         # Quit annotation
#         break

# cv2.destroyAllWindows()

# # Save the annotations
# for i, annotation in enumerate(annotations):
#     bbox = annotation
#     # Save the annotated image with a unique name
#     cv2.imwrite(os.path.join(annotated_dataset_path, f'image_{i}.jpg'), car_images[i])
#     # Save the bounding box coordinates in a text file or a custom format
#     with open(os.path.join(annotated_dataset_path, f'image_{i}.txt'), 'w') as f:
#         f.write(f'{bbox["x"]} {bbox["y"]} {bbox["width"]} {bbox["height"]}')

# # Print the total number of annotated images
# print("Annotated dataset size:", len(annotations))

import cv2
import os

# Set the path to your car images dataset directory
dataset_path = './dataset'

# Set the path to save the annotated dataset
annotated_dataset_path = './annotated'

# Create the annotated dataset directory if it doesn't exist
os.makedirs(annotated_dataset_path, exist_ok=True)

# Load the car images from the dataset directory
car_images = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        # Check if the file is an image
        if file.endswith('.jpg') or file.endswith('.png'):
            # Load the image
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            car_images.append(image)

# Global variables for mouse events
annotations = []
current_annotation = {'x': -1, 'y': -1, 'width': -1, 'height': -1}
annotation_in_progress = False

# Mouse event callback function
def annotate_cars(event, x, y, flags, param):
    global current_annotation, annotation_in_progress, annotated_image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a new bounding box
        current_annotation = {'x': x, 'y': y, 'width': -1, 'height': -1}
        annotation_in_progress = True
    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing the bounding box
        current_annotation['width'] = x - current_annotation['x']
        current_annotation['height'] = y - current_annotation['y']
        annotations.append(current_annotation.copy())
        annotation_in_progress = False

        # Draw the annotated bounding box on the image
        cv2.rectangle(annotated_image, (current_annotation['x'], current_annotation['y']),
                      (current_annotation['x'] + current_annotation['width'],
                       current_annotation['y'] + current_annotation['height']),
                      (0, 255, 0), 2)

# Display the images and prompt for annotation
for i, image in enumerate(car_images):
    annotated_image = image.copy()

    # Create a named window and set the mouse callback function
    cv2.namedWindow('Annotate Cars')
    cv2.setMouseCallback('Annotate Cars', annotate_cars)

    while True:
        # Display the image
        cv2.imshow('Annotate Cars', annotated_image)

        # Prompt for annotation
        key = cv2.waitKey(1) & 0xFF

        # Annotation process
        if key == ord('q'):
            # Quit annotation
            break
        elif key == ord('c'):
            # Continue to the next image
            break
        elif key == ord('r'):
            # Reset the annotation for the current image
            annotated_image = image.copy()
            if annotations:
                annotations.pop()

    if key == ord('q'):
        # Quit annotation
        break

    # Save the annotated image with a unique name
    cv2.imwrite(os.path.join(annotated_dataset_path, f'image_{i}.jpg'), annotated_image)

    # Save the bounding box coordinates in a text file for each image
    with open(os.path.join(annotated_dataset_path, f'image_{i}.txt'), 'w') as f:
        for annotation in annotations:
            bbox = annotation
            f.write(f'{bbox["x"]} {bbox["y"]} {bbox["width"]} {bbox["height"]}\n')

    # Clear annotations for the next image
    annotations.clear()

cv2.destroyAllWindows()

# Print the total number of annotated images
print("Annotated dataset size:", len(car_images))

