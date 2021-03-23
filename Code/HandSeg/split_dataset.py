import shutil #library for moving files
import glob #for accessing the file paths

# globals
handseg_path = "../../../handseg-150k"

def createTestSplit():
    #create directories for test data and place images there
    #separation: 70% train data and 30% test data
    destination_data = Path(handseg_path+'/test_images')
    destination_label = Path(handseg_path+'/test_masks')

    #load all file names into a list
    file_paths = glob.glob(handseg_path+"/images/*.png")
    label_paths = glob.glob(handseg_path+"/masks/*.png")

    #define amount of file for test data set
    amount_test_data = 158000*0.3
    
    # shuffle the data -> create equal distributions
    zipped_list = list(zip(file_paths, label_paths))
    random.shuffle(zipped_list)
    file_paths, label_paths = list(zip(*zipped_list))
    
    
    #for the data images
    for path in file_paths:
        if amount_test_data > 0:
            shutil.move(path, destination_data)
            amount_test_data = amount_test_data - 1
        else:
            break

    amount_test_data = 158000*0.3

    #for the labeled data
    for path in label_paths:
        if amount_test_data > 0:
            shutil.move(path, destination_label)
            amount_test_data = amount_test_data - 1
        else:
            break


def createValidationSplit():
    # create validation data
    #load all file names into a list
    file_paths = glob.glob(handseg_path+"/images/*.png")
    label_paths = glob.glob(handseg_path+"/masks/*.png")

    #define amount of file for test data set 20% of data for validation
    amount_data = len(file_paths)*0.2
    tmp = amount_data

    # shuffle the data -> create equal distributions
    zipped_list = list(zip(file_paths, label_paths))
    random.shuffle(zipped_list)
    file_paths, label_paths = list(zip(*zipped_list))
    
    #for the data images
    for path in file_paths:
        if tmp > 0:
            shutil.move(path, handseg_path+"/val_images/")
            tmp -= 1
        else:
            break

    tmp = amount_data

    #for the labeled data
    for path in label_paths:
        if tmp > 0:
            shutil.move(path, handseg_path+"/val_masks/")
            tmp -= 1
        else:
            break

def createTwoClassAnnotations():
    # manipulate the mask images from three to only two class labels
    label_paths_img = glob.glob(handseg_path+"/masks/*.png")
    label_paths_val = glob.glob(handseg_path+"/val_masks/*.png")
    label_paths_test = glob.glob(handseg_path+"/test_masks/*.png")

    # masks in training data
    for path in tqdm(label_paths_img):
      mask =  cv2.imread(path)
      mask = tf.where(mask == 2, 1, mask).numpy()
      cv2.imwrite(path, mask)

    # masks in validation data
    for path in tqdm(label_paths_val):
      mask =  cv2.imread(path)
      mask = tf.where(mask == 2, 1, mask).numpy()
      cv2.imwrite(path, mask)

    # masks in test data
    for path in tqdm(label_paths_test):
      mask =  cv2.imread(path)
      mask = tf.where(mask == 2, 1, mask).numpy()
      cv2.imwrite(path, mask)  