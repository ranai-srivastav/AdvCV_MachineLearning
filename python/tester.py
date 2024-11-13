image = image.astype(np.float32)

# Convert to grayscale
gray = color.rgb2gray(image)

# Display grayscale image
plt.imshow(gray, cmap='gray')
plt.show()

# Apply a Gaussian blur
blurred = filters.gaussian(gray, sigma=3)
plt.imshow(blurred, cmap='gray')
plt.show()

# Apply Otsu's threshold
threshold_value = filters.threshold_otsu(gray)
binary_image = gray > threshold_value
plt.imshow(binary_image, cmap='gray')
plt.show()
