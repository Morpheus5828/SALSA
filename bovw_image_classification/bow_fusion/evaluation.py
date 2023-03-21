import features_extraction as extract
import clustering_algorithm as clustering
import file_management as management
import resize_images as resize
import model

from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

# Open train images
all_images = management.open_images_from_dir_of_dir("data")
sea_images = management.open_images_from_dir("data/train/sea")

# Resize images
average_height = resize.get_average_height(all_images)
average_width = resize.get_average_width(all_images)

all_images = resize.resize_to(all_images, average_height, average_width)
sea_images = resize.resize_to(sea_images, average_height, average_width)

# Extract descriptor
all_features, all_descriptors = extract.compute(all_images, extract.get_sift_descriptor)
sea_features, sea_descriptors = extract.compute(sea_images, extract.get_sift_descriptor)

# Train clustering model
nb_clusters = 12
clustering_model = clustering.get_kmeans_model(sea_features, nb_clusters)

# Compute histogram
train_data = model.load_transform_label_train_data("data", all_descriptors, clustering_model, nb_clusters)

# Normalize
stdslr = StandardScaler().fit(train_data["representations"])
train_data["representations"] = stdslr.transform(train_data["representations"])

# Train model
salsa = BaggingClassifier(estimator=MLPClassifier())
salsa = model.learn_model_from_data(train_data, salsa)

# Transform test data
test_images = management.open_images_from_dir("data/test/CC2")
test_images = resize.resize_to(test_images, average_height, average_width)

test_features, test_descriptors = extract.compute(test_images, extract.get_sift_descriptor)
test_data = model.load_transform_test_data("test/CC2", clustering_model, test_descriptors, nb_clusters)

# Normalize
stdslr = StandardScaler().fit(test_data["representations"])
test_data["representations"] = stdslr.transform(test_data["representations"])

# Write predictions
model.write_predictions("", "predictions.txt", test_data, salsa)
