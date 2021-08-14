'''Dataset creation by splitting raw data into train-test-validation'''
import os
import random
import shutil
import math


def generateDataset(source, category, type, split, verbose=False):
    '''Generate dataset given if it's train-validation-test

    Generate a dataset given the one that we deserve to generate and the
    percentage of split that we deserve. Also, receive as input the source path
    of the original dataset.

    Args:
        source (str): Path of the original dataset that we deserve to split
        category (str): Name of the category where we want to generate the dataset
        type (str): Accepted values:
                        "train"
                        "validation"
                        "test"
        split (int): Only admits values between 0 and 1. It marks the percentage
                     of split that we deserve
        verbose (False): Indicate level of deserved verbosity when running function.

    Returns:
        (None): Dataset generated in specified directory
    '''
    # Define destination path based on dataset type
    dest = os.path.join(source, type)
    # Create destination directory (if it doesn't exist)
    if not os.path.isdir(dest):
        os.mkdir(dest)
    # Add category to source and destination paths
    source = os.path.join(source, category)
    dest = os.path.join(dest, category)
    # Create destination directory (if it doesn't exist)
    if not os.path.isdir(dest):
        os.mkdir(dest)

    # Define number of files to be moved according to split input parameter
    total_files = len(os.listdir(source))
    no_of_files = math.ceil(total_files * split)

    print("%"*25+"{ Details Of Transfer }"+"%"*25)
    print("\n\nList of Files Moved to %s:"%(dest))

    # Using for loop to randomly choose multiple files
    for i in range(no_of_files):
        # Variable random_file stores the name of the random file chosen
        random_file=random.choice(os.listdir(source))

        if verbose:
            print("%d} %s"%(i+1, random_file))

        source_file="%s/%s"%(source, random_file)
        dest_file=dest
        # Move file from one directory to another
        shutil.move(source_file, dest_file)

    print("\n\n"+"$"*33+"[ Files Moved Successfully ]"+"$"*33)

    return

# Initial paths definition for this repository
repo_source_dir = "D:\\Data Warehouse\\plantabit"
# List preprocessing datasets
preprocessing_datasets = os.listdir(repo_source_dir)

for prepro_dir in preprocessing_datasets:
    # Create base_dir
    base_dir = os.path.join(repo_source_dir, prepro_dir)
    # List all plant species in base_dir
    species = os.listdir(base_dir)

    # Generate test dataset
    for i in species:
        generateDataset(base_dir, i, 'test', 0.1)

    # Generate validation dataset
    for i in species:
        generateDataset(base_dir, i, 'validation', 0.2)

    # Gemerate train dataset
    for i in species:
        generateDataset(base_dir, i, 'train', 1)

    # Remove empty folders after dataset generation
    for i in species:
        os.rmdir(os.path.join(base_dir, i))
