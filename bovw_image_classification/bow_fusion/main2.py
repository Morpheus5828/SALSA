import file_management as management

all = management.open_images_from_dir("data/test/CC2/all")

all_labeled = management.open_images_from_dir_of_dir("data/test/CC2/labeled")

print(len(all))
print(len(all_labeled))

