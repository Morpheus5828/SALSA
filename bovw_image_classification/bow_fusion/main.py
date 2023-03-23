import model
import resize
import bag_of_word as bow
import image_representation as representation

from sklearn.svm import LinearSVC

# Load data
train_data = model.load_labeled_train_data("save/data/train")


# Resize images
average_height = resize.get_average_height(train_data["images"])
average_width = resize.get_average_width(train_data["images"])

resized_images = resize.resize_to(train_data["images"], average_height, average_width)


# Compute BOW
sift_bow = bow.SiftBow(resized_images)
color_bow = bow.ColorBow(resized_images, 500)

nb_clusters = 17
sift_bow.learn_clustering_model(nb_clusters)
color_bow.learn_clustering_model(nb_clusters)


# Compute image's representation
representation.compute_fusion_representation(train_data, sift_bow, color_bow)
representation.normalize_representation(train_data)


salsa = LinearSVC()
score = model.estimate_model_score_average(50, train_data, salsa, 0.3)
print(score)

