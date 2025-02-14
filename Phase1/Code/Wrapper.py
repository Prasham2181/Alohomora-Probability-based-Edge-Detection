# !/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 
import glob
from sklearn.cluster import KMeans
from skimage.transform import rotate



# _______________________________________________________________GAUSSIAN FILTER_______________________________________________________

def DoG_filters(kernel_size=15, sigmas=[1, 2], orientations=8, output_dir="DoG_Filters", save_as="DoG.png"):
	"""
	Generates a Difference of Gaussian (DoG) filter bank and saves filters as images.

	Parameters:
	- kernel_size: Size of the Gaussian kernel (must be odd).
	- sigmas: List of standard deviations for Gaussian kernels.
	- orientations: Number of orientations for filters.
	- output_dir: Directory to save the filter images.
	- save_as: File name for the combined filter visualization.

	Returns:
	- filter_bank: List of tuples containing rotated Sobel filter responses (g_x_rotated, g_y_rotated).
	"""
	# Define Gaussian kernel
	def calculate_gaussian(kernel_size, sigma):
		if kernel_size % 2 == 0:
			raise ValueError("Kernel size must be odd for symmetry")
		x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
		y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
		X, Y = np.meshgrid(x, y)
		gaussian = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
		gaussian /= gaussian.sum()  # Normalize
		return gaussian

	# Convolve with Sobel filters
	def sobel_convolve(gaussian):
		sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
		sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
		g_x = cv2.filter2D(gaussian, -1, sobel_x)
		g_y = cv2.filter2D(gaussian, -1, sobel_y)
		return g_x, g_y

	# Rotate the filter
	def rotate_filter(filter, angle):
		size = filter.shape[0]
		center = (size // 2, size // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1)
		rotated_filter = cv2.warpAffine(filter, M, (size, size))
		return rotated_filter

	# Generate filter bank
	def create_filter_bank(kernel_size, sigmas, orientations):
		filter_bank = []
		for sigma in sigmas:
			gaussian = calculate_gaussian(kernel_size, sigma)
			g_x, g_y = sobel_convolve(gaussian)
			for orientation in range(orientations):
				angle = orientation * (360 / orientations)
				g_x_rotated = rotate_filter(g_x, angle)
				g_y_rotated = rotate_filter(g_y, angle)
				filter_bank.append((g_x_rotated, g_y_rotated))
		return filter_bank

	# Save filters to disk
	def save_filters(filter_bank, sigmas, orientations, output_dir):
		os.makedirs(output_dir, exist_ok=True)
		for i, (g_x, g_y) in enumerate(filter_bank):
			g_x_normalized = cv2.normalize(g_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			g_y_normalized = cv2.normalize(g_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			scale_index = i // orientations
			orientation_index = i % orientations
			scale = sigmas[scale_index]
			angle = orientation_index * (360 / orientations)
			g_x_filename = os.path.join(output_dir, f"Filter_Scale{scale}_Orientation{int(angle)}_Gx.png")
			g_y_filename = os.path.join(output_dir, f"Filter_Scale{scale}_Orientation{int(angle)}_Gy.png")
			cv2.imwrite(g_x_filename, g_x_normalized)
			cv2.imwrite(g_y_filename, g_y_normalized)

	# Display filters in one window
	def display_filters(folder_path, columns, save_as):
		image_paths = sorted(glob.glob(f"{folder_path}/*.png"))
		images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_paths]
		num_images = len(images)
		rows = (num_images + columns - 1) // columns
		fig, axes = plt.subplots(rows, columns, figsize=(20, rows * 3))
		for idx, image in enumerate(images):
			row = idx // columns
			col = idx % columns
			ax = axes[row, col] if rows > 1 else axes[col]
			ax.imshow(image, cmap='gray')
			ax.axis('off')
		plt.tight_layout()
		plt.savefig(save_as)
		plt.show()

	# Generate filter bank
	filter_bank = create_filter_bank(kernel_size, sigmas, orientations)

	# Save and display filters
	save_filters(filter_bank, sigmas, orientations, output_dir)
	display_filters(output_dir, columns=16, save_as=save_as)

	return filter_bank
# filters = DoG_filters()
# print("DoG filter bank generated. You can now use it for further processing.")
# __________________________________________________________________________________________________________________________________


#_____________________________________________________________ Leung-Malik FILTER_______________________________________________________


def LM_filters(num_orientations=6, kernel_size=49, version="LMS", save_filename="LM.png", visualize=True):
	def calculate(kernel_size, sigma_x, sigma_y=None):
		if sigma_y is None:
			sigma_y = sigma_x  # Use isotropic Gaussian if sigma_y is not provided
		x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
		y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
		X, Y = np.meshgrid(x, y)
		gaussian = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
			-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2))
		)
		gaussian /= gaussian.sum()  # Normalize
		return gaussian, X, Y

	def gaussian_derivative(kernel_size, sigma, order, angle):
		"""Generate Gaussian derivative filters and apply rotation."""
		gaussian, X, Y = calculate(kernel_size, sigma, 3 * sigma)
		if order == 1:
			response = -X * gaussian
		elif order == 2:
			response = (X**2 - sigma**2) * gaussian
		else:
			raise ValueError("Order must be 1 or 2")

		# Rotate the filter
		response = rotate_filter(response, angle)
		return response

	def rotate_filter(filter, angle):
		size = filter.shape[0]
		center = (size // 2, size // 2)
		M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1)
		rotated_filter = cv2.warpAffine(filter, M, (size, size), flags=cv2.INTER_LINEAR)
		return rotated_filter

	def laplacian_of_gaussian(kernel_size, sigma):
		"""Generate Laplacian of Gaussian filter."""
		gaussian, X, Y = calculate(kernel_size, sigma)
		LoG = (X**2 + Y**2 - 2 * sigma**2) / (sigma**4) * gaussian
		return LoG

	def direct_gaussian(kernel_size, sigma):
		"""Generate isotropic Gaussian filter."""
		gaussian, _, _ = calculate(kernel_size, sigma)
		return gaussian

	# Define scales based on the version
	if version == "LMS":
		scales = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]
	elif version == "LML":
		scales = [np.sqrt(2), 2, 2 * np.sqrt(2), 4]
	else:
		raise ValueError("Version must be either 'LMS' or 'LML'")

	orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)

	# Initialize filter bank
	filter_bank = {
		"first_derivative": [],
		"second_derivative": [],
		"laplacian_of_gaussian": [],
		"gaussian": [],
	}

	# First and Second Derivative Filters
	for scale in scales[:3]:
		for angle in orientations:
			# First Derivative
			first_derivative = gaussian_derivative(kernel_size, scale, 1, angle)
			filter_bank["first_derivative"].append(first_derivative)

			# Second Derivative
			second_derivative = gaussian_derivative(kernel_size, scale, 2, angle)
			filter_bank["second_derivative"].append(second_derivative)

	# Laplacian of Gaussian and Gaussian Filters
	for scale in scales:
		# LoG Filters
		filter_bank["laplacian_of_gaussian"].append(laplacian_of_gaussian(kernel_size, scale))
		filter_bank["laplacian_of_gaussian"].append(laplacian_of_gaussian(kernel_size, 3 * scale))

		# Gaussian Filters
		filter_bank["gaussian"].append(direct_gaussian(kernel_size, scale))

	def visualize_filters_to_file(filters_dict, filename):
		"""Visualize all filters and save to a single image file."""
		filters = []
		titles = []

		# Flatten all filters 
		for filter_type, filter_list in filters_dict.items():
			filters.extend(filter_list)
			titles.extend([filter_type] * len(filter_list))

		# Determine grid size
		n_filters = len(filters)
		cols = 12
		rows = (n_filters // cols) + (1 if n_filters % cols != 0 else 0)

		# Create the figure
		fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 1.5))
		for i, ax in enumerate(axes.flat):
			if i < len(filters):
				ax.imshow(filters[i], cmap="gray")
				ax.axis("off")
				ax.set_title(titles[i], fontsize=6)
			else:
				ax.axis("off")

		plt.tight_layout()
		plt.savefig(filename, dpi=300)
		plt.close()

	def visualize_interactive(filters_dict):
		for filter_type, filter_list in filters_dict.items():
			n_filters = len(filter_list)
			cols = 6
			rows = (n_filters // cols) + (1 if n_filters % cols != 0 else 0)

			fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
			fig.suptitle(filter_type.capitalize(), fontsize=16)

			for i, ax in enumerate(axes.flat):
				if i < n_filters:
					ax.imshow(filter_list[i], cmap="gray")
					ax.axis("off")
				else:
					ax.axis("off")

			plt.tight_layout()
			plt.show()

	# Save all filters to an image file
	visualize_filters_to_file(filter_bank, save_filename)

	# Optionally display filters interactively
	if visualize:
		visualize_interactive(filter_bank)

	# Return the filter bank dictionary
	return filter_bank
# LM_filters()


# __________________________________________________________________________________________________________________________________


# _______________________________________________________________GABOR FILTER_______________________________________________________

# Define the Gabor filter generator


def Gabor_filters():
	"""
	Generates and saves a Gabor filter bank as an image file.
	"""

	def calculate_gabor(kernel_size, sigma, orientation, gamma, wavelength, phase):
		x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
		y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
		X, Y = np.meshgrid(x, y)
		x_rot = X * np.cos(orientation) + Y * np.sin(orientation)
		y_rot = -X * np.sin(orientation) + Y * np.cos(orientation)
		gaussian_gabor = np.exp(-((x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2)))
		sinusoid = np.cos((2 * np.pi * x_rot / wavelength) + phase)
		gabor_kernel = gaussian_gabor * sinusoid
		gabor_normalized = cv2.normalize(gabor_kernel, None, 0, 255, cv2.NORM_MINMAX)
		return gabor_normalized.astype(np.uint8)

	# Parameters for the Gabor filter bank
	kernel_size = 33
	sigma = 8
	gamma = 1.4
	orientations = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2, 5 * np.pi / 8, 3 * np.pi / 4, 7 * np.pi / 8]
	wavelengths = [5, 10, 15, 20]
	phase = 0

	# Generate the filter bank
	filter_bank = []
	for wavelength in wavelengths:
		row_filters = []
		for orientation in orientations:
			kernel = calculate_gabor(kernel_size, sigma, orientation, gamma, wavelength, phase)
			row_filters.append(kernel)
		filter_bank.append(row_filters)

	# Combine filters into a single grid image
	grid_rows = len(wavelengths)
	grid_cols = len(orientations)
	canvas_height = kernel_size * grid_rows
	canvas_width = kernel_size * grid_cols
	canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

	for row_idx, row_filters in enumerate(filter_bank):
		for col_idx, kernel in enumerate(row_filters):
			start_y = row_idx * kernel_size
			start_x = col_idx * kernel_size
			canvas[start_y:start_y + kernel_size, start_x:start_x + kernel_size] = kernel

	# Save and display the result
	cv2.imwrite("Gabor.png", canvas)
	plt.imshow(canvas, cmap='gray')
	plt.title("Gabor Filter Bank")
	plt.axis('off')
	plt.show()
	print("Gabor filter bank saved as 'Gabor.png'")
	return filter_bank


# Gabor_filters()
# filter_bank = Gabor_filters()
# print("Gabor filter bank generated. You can now use it for further processing.")

# ####################################################################################################
# Half Disk Map
####################################################################################################

def halfdisk_filter():
	def HalfDiskFilterBank(radii, orientations):
		filter_bank = []
		orients = np.linspace(0, 360, orientations, endpoint=False)  # Orientation angles

		for radius in radii:
			size = 2 * radius + 1  # Size of the mask
			for orient in orients:
				mask = np.zeros([size, size])
				
				# Create the upper half-disk mask
				for i in range(radius):
					for j in range(size):
						dist = (i - radius)**2 + (j - radius)**2
						if dist < radius**2:
							mask[i, j] = 1

				# Rotate the half-disk mask
				half_mask = rotate(mask, orient, preserve_range=True)
				half_mask = np.round(half_mask).astype(np.uint8)

				# Create the complementary mask (rotated by 180 degrees)
				half_mask_rot = rotate(half_mask, 180, cval=1, preserve_range=True)
				half_mask_rot = np.round(half_mask_rot).astype(np.uint8)

				# Add both masks to the filter bank
				filter_bank.append(half_mask)
				filter_bank.append(half_mask_rot)

		return filter_bank

	def plot_save(filters, file_dir, cols):
		rows = np.ceil(len(filters) / cols).astype(int)
		plt.figure(figsize=(15, 15))
		
		for index in range(len(filters)):
			plt.subplot(rows, cols, index + 1)
			plt.axis('off')
			plt.imshow(filters[index], cmap='gray')
		
		# Save the plotted grid as an image
		plt.savefig(file_dir, bbox_inches='tight')
		plt.close()

	# Generate the filter bank
	half_disk_filters = HalfDiskFilterBank([3, 9, 12], 8)

	# Save and plot the filters
	plot_save(half_disk_filters, "HDMask.png", 8)
	
	return half_disk_filters

# Call the function
# filters = halfdisk_filter()
# print("Half disk filter bank generated. You can now use it for further processing.")

# ####################################################################################################
# TEXTON MAP
####################################################################################################
# ________________________________________________________________________________________________________________
def create_texton_map(image_path, filter_bank, num_clusters=128, output_filename="TextonMap"):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Split channels
    channels = cv2.split(image)

    # Apply filters
    filtered_responses = []
    for filter in filter_bank:
        for channel in channels:
            response = cv2.filter2D(channel, -1, filter)
            filtered_responses.append(response)

    # Reshape for clustering
    stacked_responses = np.stack(filtered_responses, axis=-1)
    flattened_responses = stacked_responses.reshape(-1, stacked_responses.shape[-1])

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(flattened_responses)
    cluster_labels = kmeans.labels_.reshape(image.shape[:2])

    # Normalize and save Texton Map
    normalized_map = (cluster_labels * (255 / num_clusters)).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_VIRIDIS)
    output_path = f"{output_filename}.png"
    cv2.imwrite(output_path, color_map)
    print(f"Texton Map saved at: {output_path}")

    # Display Texton Map
    plt.imshow(color_map[:, :, ::-1])  # Convert BGR to RGB
    plt.title("Texton Map")
    plt.show()

    return cluster_labels

# __________________________________________________Brightness map_______________________________________________________________

####################################################################################################
# Brightness Map
####################################################################################################

def generate_brightness_map(image_path, num_clusters=32, output_filename="BrightnessMap"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found. Please check the path.")

    # Flatten and cluster
    flattened_image = image.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(flattened_image)
    brightness_labels = kmeans.labels_.reshape(image.shape)

    # Normalize and save Brightness Map
    normalized_map = (brightness_labels * (255 / num_clusters)).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_VIRIDIS)

    output_path = f"{output_filename}.png"
    cv2.imwrite(output_path, color_map)  # Save the color map
    print(f"Brightness Map saved at: {output_path}")

    # Display Brightness Map
    plt.imshow(cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB))  # Display in RGB
    plt.colorbar()
    plt.title("Brightness Map")
    plt.show()

    return brightness_labels



####################################################################################################
# COLOR MAP
###########################################################################################

def generate_color_map(image_path, num_clusters=32, output_filename="ColorMap", color_space="RGB"):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Image not found. Please check the path.")

    # Convert color space
    if color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "Lab":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif color_space == "YCbCr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_space == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    # Reshape and cluster
    flattened_image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(flattened_image)
    color_labels = kmeans.labels_.reshape(image.shape[:2])

    # Normalize and save Color Map
    normalized_map = (color_labels * (255 / num_clusters)).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_VIRIDIS)
    output_path = f"{output_filename}.png"
    cv2.imwrite(output_path, color_map)  # Save the color map
    print(f"Color Map saved at: {output_path}")

    # Display Color Map
    plt.imshow(cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB))  # Display in RGB
    plt.colorbar()
    plt.title(f"Color Map ({color_space} Space)")
    plt.show()

    return color_labels


####################################################################################################
# Gradient Calculation
####################################################################################################


def compute_gradient(image, bins, half_disk_filters, output_filename="Gradient"):
    gradients = []

    def chi_square_dist(img, bin_id, filter1, filter2):
        tmp = (img == bin_id).astype(np.float32)
        g = cv2.filter2D(tmp, -1, filter1)
        h = cv2.filter2D(tmp, -1, filter2)
        chi_sqr = (g - h) ** 2 / (g + h + 1e-5)
        return chi_sqr / 2

    for i in range(0, len(half_disk_filters), 2):
        chi_sqr_dist = np.zeros_like(image, dtype=np.float32)
        for bin_id in range(bins):
            chi_sqr_dist += chi_square_dist(image, bin_id, half_disk_filters[i], half_disk_filters[i + 1])

        gradients.append(chi_sqr_dist)

    gradient_matrix = np.stack(gradients, axis=-1)
    gradient_magnitude = np.max(gradient_matrix, axis=-1)
    normalized_gradient = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

    # Apply colormap
    color_gradient = cv2.applyColorMap(normalized_gradient, cv2.COLORMAP_VIRIDIS)

    # Save Gradient
    output_path = f"{output_filename}.png"
    cv2.imwrite(output_path, color_gradient)  # Save the color gradient
    print(f"Gradient saved at: {output_path}")

    # Display Gradient
    plt.imshow(cv2.cvtColor(color_gradient, cv2.COLOR_BGR2RGB))  # Display in RGB
    plt.colorbar()
    plt.title("Color Gradient")
    plt.show()

    return color_gradient


def ensure_output_directory(output_dir="output"):
    """
    Ensure that the specified output directory exists.
    Creates the directory if it does not exist.
    
    :param output_dir: Name of the output directory, defaults to 'output'.
    :type output_dir: str
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
        
        
# __________________________________________________________________________________________________________________________________

def main():
    # Define paths and parameters
    images_folder = r"Phase1/BSDS500/Images"  # Folder containing input images
    sobel_folder = r"Phase1/BSDS500/SobelBaseline"  # Folder containing Sobel baseline images
    canny_folder = r"Phase1/BSDS500/CannyBaseline"  # Folder containing Canny baseline images
    output_dir = "output"  # Directory to save outputs
    num_clusters = 64  # Number of clusters for K-Means

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate filter banks
    print("Generating filter banks...")
    DoG_Filters = DoG_filters()
    Gabor_Filters = Gabor_filters()
    LM_Filters_Dict = LM_filters()
    LM_Filters = [
        np.array(filter, dtype=np.float32)
        for filter_list in LM_Filters_Dict.values()
        for filter in filter_list
    ]
    combined_filter_bank = LM_Filters + [
        np.array(filter, dtype=np.float32) for filter_row in Gabor_Filters for filter in filter_row
    ] + [
        np.array(filter[0], dtype=np.float32) for filter in DoG_Filters
    ] + [
        np.array(filter[1], dtype=np.float32) for filter in DoG_Filters
    ]
    print("Filter banks generated successfully.")

    # Generate half-disk filters
    print("Generating half-disk filters...")
    HD_Filters = halfdisk_filter()

    # Process each image
    for image_filename in os.listdir(images_folder):
        # Construct full paths for the image and corresponding baseline files
        image_path = os.path.join(images_folder, image_filename)
        sobel_path = os.path.join(sobel_folder, os.path.splitext(image_filename)[0] + ".png")
        canny_path = os.path.join(canny_folder, os.path.splitext(image_filename)[0] + ".png")

        # Normalize paths
        sobel_path = sobel_path.replace("\\", "/")
        canny_path = canny_path.replace("\\", "/")

        # Debug: Print constructed paths for each file
        print(f"Image Path: {image_path}")
        print(f"Sobel Path: {sobel_path}")
        print(f"Canny Path: {canny_path}")

        # Check file existence
        if not os.path.exists(sobel_path) or not os.path.exists(canny_path):
            print(f"Skipping {image_filename} as corresponding baseline images are missing.")
            continue

        # Skip non-image files
        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {image_filename}")
            continue

        # Create output subfolder for this image
        image_output_dir = os.path.join(output_dir, f"img_{os.path.splitext(image_filename)[0]}")
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        print(f"Processing image: {image_filename}")

        # Generate Texton Map
        texton_map_result = create_texton_map(
            image_path, combined_filter_bank, num_clusters, 
            output_filename=os.path.join(image_output_dir, "TextonMap")
        )
        texton_gradient = compute_gradient(
            texton_map_result, num_clusters, HD_Filters, 
            output_filename=os.path.join(image_output_dir, "TextonGradient")
        )

        # Generate Brightness Map and Gradient
        brightness_map_result = generate_brightness_map(
            image_path, num_clusters=16, 
            output_filename=os.path.join(image_output_dir, "BrightnessMap")
        )
        brightness_gradient = compute_gradient(
            brightness_map_result, bins=16, half_disk_filters=HD_Filters,
            output_filename=os.path.join(image_output_dir, "BrightnessGradient")
        )

        # Generate Color Map and Gradient
        color_map_result = generate_color_map(
            image_path, num_clusters=16, 
            output_filename=os.path.join(image_output_dir, "ColorMap"),
            color_space="Lab"
        )
        color_gradient = compute_gradient(
            color_map_result, bins=16, half_disk_filters=HD_Filters,
            output_filename=os.path.join(image_output_dir, "ColorGradient")
        )

        # Load baselines
        sobel_baseline = cv2.imread(sobel_path, cv2.IMREAD_GRAYSCALE)
        canny_baseline = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)

        if sobel_baseline is None or canny_baseline is None:
            print(f"Skipping {image_filename} as Sobel or Canny baseline image could not be loaded.")
            continue

        # Combine responses to compute Pb-Lite
        if len(texton_gradient.shape) == 3:  # Check if texton_gradient has color channels
            pb_term1 = np.mean(texton_gradient + brightness_gradient + color_gradient, axis=-1) / 3  # Convert to grayscale
        else:
            pb_term1 = (texton_gradient + brightness_gradient + color_gradient) / 3

        pb_term2 = 0.5 * sobel_baseline + 0.5 * canny_baseline
        pb_lite = np.multiply(pb_term1, pb_term2)

        # Normalize and save Pb-Lite
        pb_lite_normalized = cv2.normalize(pb_lite, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        pb_lite_filename = os.path.join(image_output_dir, "PbLite.png")
        cv2.imwrite(pb_lite_filename, pb_lite_normalized)
        print(f"Pb-Lite saved at: {pb_lite_filename}")

        # Display Pb-Lite
        plt.imshow(pb_lite_normalized, cmap='gray')
        plt.title(f"Pb-Lite: {image_filename}")
        plt.axis('off')
        plt.show()

        print(f"Processing complete for {image_filename}.")

    print("All images processed successfully.")


if __name__ == '__main__':
	main()



# #	____________________________________________________________________________________________________________________________	

# ####################################################################################################################################




