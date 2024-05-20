import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(path, image):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def show_image(image, title='Image'):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def edge_detection(image):
    return cv2.Canny(image, 100, 200)

def stitch_images(images, labels, cols=3):
    # Ensure all images are of the same size
    standard_height, standard_width, _ = images[0].shape
    resized_images = [cv2.resize(img, (standard_width, standard_height)) for img in images]

    rows = (len(images) + cols - 1) // cols
    stitched_image = np.zeros((rows * standard_height, cols * standard_width, 3), dtype=np.uint8)

    for idx, (image, label) in enumerate(zip(resized_images, labels)):
        row = idx // cols
        col = idx % cols
        y_offset = row * standard_height
        x_offset = col * standard_width

        stitched_image[y_offset:y_offset + standard_height, x_offset:x_offset + standard_width] = image
        cv2.putText(stitched_image, label, (x_offset + 10, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return stitched_image

def tensor_to_image(tensor):
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)
    array = tensor.permute(1, 2, 0).numpy()
    return (array * 255).astype(np.uint8)

def apply_filter(image_tensor, filter):
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    filtered_image = torch.nn.functional.conv2d(image_tensor, filter, padding=1)
    return filtered_image.squeeze(0)  # Remove batch dimension

def main():
    image_path = 'data/isp.jpg'
    standard_size = (256, 256)

    image = load_image(image_path)
    print("Loaded original image.")
    images = []
    labels = []

    resized_image = resize_image(image, *standard_size)
    images.append(resized_image)
    labels.append('Original Image')
    save_image('data/resized_image.jpg', resized_image)
    print("Resized image to 256x256 and saved.")

    blurred_image = gaussian_blur(resized_image, 5)
    images.append(blurred_image)
    labels.append('Blurred Image')
    save_image('data/blurred_image.jpg', blurred_image)
    print("Applied Gaussian blur to resized image and saved.")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    gray_tensor = transforms.ToTensor()(gray_image).unsqueeze(0)
    print("Converted resized image to grayscale.")

    # Define the edge detection filter
    edge_filter = torch.tensor([[[[-1, -1, -1],
                                 [-1,  8, -1],
                                 [-1, -1, -1]]]], dtype=torch.float32)

    edge_detected_image = apply_filter(gray_tensor, edge_filter)
    edge_detected_image = edge_detected_image.squeeze().numpy()
    edge_detected_image = (edge_detected_image * 255).astype(np.uint8)
    print("Applied edge detection filter to grayscale image.")

    # Ensure the edge_detected_image has correct dimensions
    if len(edge_detected_image.shape) == 2:
        edge_detected_image = np.expand_dims(edge_detected_image, axis=2)

    edge_detected_image = cv2.cvtColor(edge_detected_image, cv2.COLOR_GRAY2RGB)
    images.append(edge_detected_image)
    labels.append('Edge Detected Image')
    save_image('data/edge_detected_image.jpg', edge_detected_image)
    print("Saved edge detected image.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    resized_gray_image = resize_image(gray_image, *standard_size)
    images.append(resized_gray_image)
    labels.append('Grayscale Image')
    save_image('data/gray_image.jpg', resized_gray_image)
    print("Converted original image to grayscale and saved.")

    equalized_image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)
    resized_equalized_image = resize_image(equalized_image, *standard_size)
    images.append(resized_equalized_image)
    labels.append('Equalized Image')
    save_image('data/equalized_image.jpg', resized_equalized_image)
    print("Applied histogram equalization to grayscale image and saved.")

    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    resized_blurred_image = resize_image(blurred_image, *standard_size)
    images.append(resized_blurred_image)
    labels.append('Gaussian Blurred Image')
    save_image('data/gaussian_blurred_image.jpg', resized_blurred_image)
    print("Applied Gaussian blur to original image and saved.")

    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    resized_sharpened_image = resize_image(sharpened_image, *standard_size)
    images.append(resized_sharpened_image)
    labels.append('Sharpened Image')
    save_image('data/sharpened_image.jpg', resized_sharpened_image)
    print("Applied sharpening filter to original image and saved.")

    bilateral_image = cv2.bilateralFilter(image, 9, 75, 75)
    resized_bilateral_image = resize_image(bilateral_image, *standard_size)
    images.append(resized_bilateral_image)
    labels.append('Bilateral Filtered Image')
    save_image('data/bilateral_image.jpg', resized_bilateral_image)
    print("Applied bilateral filter to original image and saved.")

    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    resized_eroded_image = resize_image(eroded_image, *standard_size)
    images.append(resized_eroded_image)
    labels.append('Eroded Image')
    save_image('data/eroded_image.jpg', resized_eroded_image)
    print("Applied erosion to original image and saved.")

    dilated_image = cv2.dilate(image, kernel, iterations=1)
    resized_dilated_image = resize_image(dilated_image, *standard_size)
    images.append(resized_dilated_image)
    labels.append('Dilated Image')
    save_image('data/dilated_image.jpg', resized_dilated_image)
    print("Applied dilation to original image and saved.")

    _, thresholded_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 128, 255, cv2.THRESH_BINARY)
    thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)
    resized_thresholded_image = resize_image(thresholded_image, *standard_size)
    images.append(resized_thresholded_image)
    labels.append('Thresholded Image')
    save_image('data/thresholded_image.jpg', resized_thresholded_image)
    print("Applied thresholding to grayscale image and saved.")

    stitched_image = stitch_images(images, labels)
    cv2.imwrite('data/stitched_image.jpg', cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
    print("Created and saved stitched image with labels.")

    plt.figure(figsize=(15, 15))
    plt.imshow(stitched_image)
    plt.title('Stitched Image with Labels')
    plt.axis('off')
    plt.show()
    print("Displayed stitched image with labels.")

if __name__ == "__main__":
    main()
