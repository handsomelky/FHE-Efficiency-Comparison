# Fully Homomorphic Encryption Algorithm Efficiency Testing - FHE Efficiency Comparison

This project is an innovative image encryption and matching system, based on two advanced fully homomorphic encryption algorithms: BFV and CKKS. It aims to demonstrate the application potential of fully homomorphic encryption technology in the fields of secure image processing and privacy protection.

The system designed in this project not only shows the capabilities of homomorphic encryption technology in secure image processing and privacy protection but also provides a quantitative assessment of the performance of these algorithms through actual testing.

A major highlight of this project is its ability to perform image matching operations while the image data remains encrypted. This operation effectively tests the performance and practicality of homomorphic encryption in real-world applications. Through this project, we aim to provide an experimental foundation and insights for the further development of fully homomorphic encryption technology in real-world applications.

[简体中文](readme/README.zh-CN.md)  | [繁體中文](readme/README.zh-TW.md) | English

## Table of Contents

+ [Features](https://github.com/handsomelky/FHE-Efficiency-Comparison#features)

+ [Installation](https://github.com/handsomelky/FHE-Efficiency-Comparison#installation)

+ [Usage](https://github.com/handsomelky/FHE-Efficiency-Comparison#usage)
  + [tenseal_encrypt.py](https://github.com/handsomelky/FHE-Efficiency-Comparison#tenseal_encryptpy)
  + [DBUploader.py](https://github.com/handsomelky/FHE-Efficiency-Comparison#dbuploaderpy)
  + [tenseal_matching.py](https://github.com/handsomelky/FHE-Efficiency-Comparison#tenseal_matchingpy)

+ [Examples](https://github.com/handsomelky/FHE-Efficiency-Comparison#examples)

+ Design Concept
  + [Project Design](https://github.com/handsomelky/FHE-Efficiency-Comparison#project-design)
  + [Environment Setup](https://github.com/handsomelky/FHE-Efficiency-Comparison#environment-setup)
  + [Preparing Test Data](https://github.com/handsomelky/FHE-Efficiency-Comparison#preparing-test-data)
  + [Encrypting Data](https://github.com/handsomelky/FHE-Efficiency-Comparison#encrypting-data)
  + [Storing in Database](https://github.com/handsomelky/FHE-Efficiency-Comparison#storing-data-in-the-database)
  + [Query Matching](https://github.com/handsomelky/FHE-Efficiency-Comparison#query-matching)
  + [Innovative Points](https://github.com/handsomelky/FHE-Efficiency-Comparison#innovative-points)

+ Experiments and Analysis
  + [Experiment Environment](https://github.com/handsomelky/FHE-Efficiency-Comparison#experimental-environment)
  + [BFV](https://github.com/handsomelky/FHE-Efficiency-Comparison#bfv)
  + [CKKS](https://github.com/handsomelky/FHE-Efficiency-Comparison#ckks)
  + [Comparing Different Parameters](https://github.com/handsomelky/FHE-Efficiency-Comparison#comparison-of-different-parameters)

+ [Contributions](https://github.com/handsomelky/FHE-Efficiency-Comparison#contributions)

+ [License](https://github.com/handsomelky/FHE-Efficiency-Comparison#license)

## Features

- **Implementation and Testing of Fully Homomorphic Encryption Algorithms**: The project implements both BFV and CKKS fully homomorphic encryption algorithms, providing testing and efficiency comparisons for these two algorithms.
- **Image Preprocessing**: Includes an image preprocessing step, cropping, and compressing the images to a specific resolution to fit the encryption algorithm.
- **Efficient Encrypted Data Storage and Management**: Utilizes SQLite database for storing and managing encrypted image data and their label information.
- **Simulation of Real-world Application Scenarios**: Simulates a real-world image query service, demonstrating the application capabilities of homomorphic encryption in actual data processing and privacy protection scenarios.
- **Flexible Encryption Parameter Configuration**: Offers customizable encryption parameter settings, allowing users to adjust encryption strength to test performance based on different application needs.
- **Detailed Performance Measurement and Progress Display**: Shows detailed performance metrics and progress information during the encryption, storage, matching, and decryption processes.

## Installation

To run this project, you need to install Python and several dependent libraries on your system.

- Python 3.8
- NumPy
- Pillow
- TenSEAL

You can install all required packages using the following command:

```shell
pip install -r requirements.txt
```

## Usage

In this project, different Python scripts provide various command-line options, allowing users to customize their run according to their needs. Below are detailed explanations of the option parameters for each script:

### `tenseal_encrypt.py`

This script is used for encrypting images. It accepts the following parameters:

- `-a` or `--algorithm`: Specifies the encryption algorithm. Options are `BFV` or `CKKS`. For example, `-a BFV`.
- `-i` or `--input_folder`: Specifies the folder containing images to be encrypted. For example, `-i data/storage_imgs`.
- `-o` or `--output_folder`: Specifies the output folder for encrypted images. For example, `-o data/enc_storage_imgs`.
- `-p` or `--poly_modulus_degree`: Sets the polynomial modulus degree. For example, `-p 32768`.
- `-m` or `--plain_modulus`: Used only when choosing the BFV algorithm, to set the plaintext modulus. For example, `-m 65537`.
- `-s` or `--global_scale`: Used only when choosing the CKKS algorithm, to set the global scale factor. For example, `-s 2**40`.
- `-c` or `--context`: An optional parameter, specifying a path to a file containing the public key for encrypting images. For example, `-c keys/public_context`.

### `DBUploader.py`

This script is used to upload encrypted images and their corresponding labels to the SQLite database. It accepts the following parameters:

- `-i` or `--input_folder`: Specifies the folder containing encrypted images. For example, `-i data/enc_storage_imgs`.
- `-db` or `--database`: Specifies the name of the database file. For example, `-db encrypted_images.db`.

### `tenseal_matching.py`

This script is used to match encrypted images in the database. It accepts the following parameters:

- `-a` or `--algorithm`: Specifies the algorithm used for encrypting images. Options are `BFV` or `CKKS`. For example, `-a BFV`.
- `-i` or `--input_folder`: Specifies the folder containing encrypted images to be matched. For example, `-i data/enc_matching_imgs`.
- `-o` or `--output_folder`: Specifies the output folder for matching results. For example, `-o data/result`.
- `-c` or `--context`: Specifies a path to a file containing the private key for decrypting matched images. For example, `-c keys/private_context`.
- `-db` or `--database`: Specifies the name of the database file. For example, `-db encrypted_images.db`.

These option parameters allow users to flexibly handle different datasets, encryption algorithms and parameters, as well

 as output options, thus testing and using fully homomorphic encryption technology in various scenarios.

## Examples

Here is an example of testing the BFV algorithm:

```shell
# Encrypting storage images using the BFV algorithm
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs
# Uploading encrypted image data to the database
python .\DBuploader.py -i .\data\enc_storage_imgs\
# Encrypting images for matching
python tenseal_encrypt.py -a BFV -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
# Matching encrypted images in the database and outputting results
python .\tenseal_matching.py -a BFV -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

## Design Philosophy

I have uploaded the complete design philosophy to my blog: [开源库全同态加密算法效率比较 | R1ck's Portal (rickliu.com)](https://rickliu.com/posts/3c0eb5a1ba70/)

### Project Design

For this experiment, we chose TenSEAL as the open-source homomorphic encryption library, hosted on GitHub at the following link:

**TenSEAL**: https://github.com/OpenMined/TenSEAL

This open-source library implements both BFV and CKKS fully homomorphic encryption algorithms.

The experimental process is as follows:

1. **Environment Setup**: Install the TenSEAL library and its dependencies.
2. **Prepare Test Data**: Collect a set of images as test data and preprocess them to a resolution of 64*64.
3. **Encrypt Data**: Encrypt the test images using the selected algorithm and record the time spent.
4. **Store in Database**: Store the encrypted data and corresponding labels in the database.
5. **Query Matching**: Encrypt query images, match them with encrypted images in the database, select the most similar image, decrypt and display it, and record the time for matching and decryption.
6. **Analyze and Compare Algorithm Efficiency**: Analyze the time for each step to compare the efficiency and advantages and disadvantages of different algorithms.

Steps 2, 3, 4, and 5 need to be repeated for different algorithms.

Here we also need to clarify the process of encryption and decryption.

As we aim to **simulate a real-world scenario of a query image service**, encrypted images are stored in the server's database, while matching images are provided by the client.

In this role relationship, the server acts as the authoritative party, so **the generation of public and private keys is initially done on the server**.

The server uses the public key to encrypt stored images and **distributes the public key to the client**, keeping the private key.

The client uses the public key to encrypt the test images and sends them to the server. **The server** matches them with encrypted images in the database, finds the most similar one, **decrypts it using the private key and returns the original image to the client**.

The entire process can be represented by the following flowchart:

![Project Process](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/%E9%A1%B9%E7%9B%AE%E6%B5%81%E7%A8%8B-17013208703374.png)

Of course, this only simulates one scenario in the real world. In other schemes, the holders of public and private keys and the state of data transmission are not fixed. **This scheme weakens the protection of data provided by the user** (which can be provided through additional encryption algorithms), and instead focuses on **protecting the images stored in the database**.

In our scheme, **homomorphic encryption is specifically reflected in the process of encrypted image matching**. If the original image has the highest similarity, then the similarity calculated under encrypted state is also the highest.

### Environment Setup

Create a new Python environment and install TenSEAL using pip:

``` shell
conda create -n TenSEAL python=3.8
conda activate TenSEAL
pip install tenseal
```

![image-20231129084420443](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231129084420443.png)

### Preparing Test Data

To simulate real-world scenarios of homomorphic encryption for storing private information, we used 2500 facial images from the **CelebA (CelebFaces Attribute)** dataset as our test data.

Here is the download link for the dataset:

https://pan.baidu.com/s/1eSNpdRG#list/path=/

We chose `Img/img_align_celeba.zip`, which is the cropped JPG format dataset, and selected the first 2500 images to store in the database. The initial resolution of the images was 178x218.

We wrote a Python script to crop and resize the images to 64x64. First, install the Pillow library for image processing:

```shell
pip install Pillow
```

We used the `crop()` method to crop the images and then `resize()` to scale them:

```python
from PIL import Image
import os

def resize_images(folder_path, output_folder, output_size=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Adjust the file format as needed
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(image_path) as img:
                min_side = min(img.width, img.height)
                # Calculate the cropping area
                left = (img.width - min_side) / 2
                top = (img.height - min_side) / 2
                right = left + min_side
                bottom = top + min_side

                # Crop and resize
                img_cropped = img.crop((left, top, right, bottom))
                img_resized = img_cropped.resize(output_size)

                # Save the processed image
                img_resized.save(output_path)

resize_images('data/storage_imgs', 'data/storage_imgs')
```

Additionally, we extracted the first 1000 data entries from `Anno/identity_CelebA.txt` and `Anno/list_attr_celeba.txt` to use as labels.

### Encrypting Data

Initially, we converted the images to numpy arrays, requiring the installation of the numpy library:

```shell
pip install numpy
```

We needed to determine which encryption algorithm to use, as different algorithms initialize the context with different parameters:

```python
if args.context:
    context = load_context(args.context)
else:
    if args.algorithm == 'BFV':
        context = ts.context(ts.SCHEME_TYPE.BFV, args.poly_modulus_degree, args.plain_modulus)
    elif args.algorithm == 'CKKS':
        context = ts.context(ts.SCHEME_TYPE.CKKS, args.poly_modulus_degree)
        context.global_scale = args.global_scale
```

If a context with a public key was already provided, we used it directly. Next, we generated and saved the public and private keys:

```python
def save_context(context, file_path):
    with open(file_path, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
...
# Generate public and private keys
context.generate_galois_keys()

# Serialize and save the context containing the private key
save_context(context, os.path.join(keys_folder, "private_context"))

# Remove the private key
context.make_context_public()

# Serialize and save the context with the public key
save_context(context, os.path.join(keys_folder, "public_context"))
```

When saving the context to a file, we used the `serialize` method and set `save_secret_key` to true to ensure the private key is included in the serialized context.

First, we needed to read the image to be encrypted and flatten it into a one-dimensional array:

```python
# Load the image and convert it to a numpy array
with Image.open(image_path) as img:
    img_data = np.array(img)

img_data_flattened = img_data.flatten()
```

The encryption functions for BFV and CKKS are `bfv_vector` and `ckks_vector`, respectively. We used the `context` and the flattened image array for encryption:

```python
if algorithm == 'BFV':
    encrypted_data = ts.bfv_vector(context, img_data_flattened.tolist())  # Encrypt the entire vector
elif algorithm == 'CKKS':
    encrypted_data = ts.ckks_vector(context, img_data_flattened.tolist())
else:
    raise ValueError("Unsupported algorithm. Please choose 'BFV' or 'CKKS'.")
```

Finally, we serialized the encrypted data and saved it:

```python
encrypted_data_serialize = encrypted_data.serialize()

encrypted_filename = os.path.splitext(os.path.basename(image_path))[0] + "_encrypted"
encrypted_file_path = os.path.join(output_folder, encrypted_filename)

with open(

encrypted_file_path, "wb") as f:
    f.write(encrypted_data_serialize)
```

During encryption, we used Python's built-in `time` library to track the time taken:

```python
start_time = time.time()  # Start timing
...
end_time = time.time()
encryption_time = end_time - start_time
```

At the end of the code, we calculated and displayed the total encryption time and the average encryption time per image:

```python
total_time, count = encrypt_images_in_folder(args.input_folder, args.algorithm, context, args.output_folder)
if count > 0:
    average_time = total_time / count
    print(f"Total encryption time: {total_time:.3f} seconds")
    print(f"Average encryption time per image: {average_time:.3f} seconds")
    print(f"Number of images encrypted: {count}")
else:
    print("No images were encrypted.")
```

### Storing Data in the Database

For database storage, we chose **SQLite** to simulate a data storage scenario, mainly because SQLite is easier to set up and is integrated into Python with the `sqlite3` library.

In real-world scenarios, a user employing an image query service would expect to receive information about the image. Therefore, in each database entry, we stored not only the encrypted image vector but also labels for the image.

We used the `Anno/list_attr_celeba.txt` and `Anno/identity_CelebA.txt` files from the CelebA dataset to provide feature and identity labels for the portrait images.

First, we created the database:

```python
def create_database(db_name):

    print(f"=== Creating database: {db_name} ===", )
    if os.path.exists(db_name):
        # If it exists, delete the file
        os.remove(db_name)
        print("Existing database removed.")

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            filename TEXT PRIMARY KEY,
            identity TEXT,
            attributes TEXT,
            encrypted_data BLOB
        )
    ''')
    conn.commit()
    conn.close()
    print("=== Database created successfully. ===\n")
```

Before processing the images, we handled the labels from the two files and stored them as data structures:

```python
def parse_identity_file(file_path):
    identity = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            identity[parts[0]] = parts[1]
    return identity

def parse_attr_file(file_path):
    attributes = {}
    with open(file_path, "r") as file:
        # Read and ignore the first line
        next(file)
        attr_names = file.readline().strip().split()
        for line in file:
            parts = line.strip().split()
            attributes[parts[0]] = dict(zip(attr_names, parts[1:]))
    return attributes
...
identity_data = parse_identity_file("Anno/identity_CelebA.txt")
attr_data = parse_attr_file("Anno/list_attr_celeba.txt")
# Process images
process_images(args.input_folder, args.database, identity_data, attr_data)
```

After encrypting the images, we found the labels for each image and stored the encrypted data along with the corresponding labels in the database:

```python
identity = identity_data.get(image_filename)
attributes = attr_data.get(image_filename)
if identity is not None and attributes is not None:
    image_path = os.path.join(input_folder, filename)
    with open(image_path, "rb") as f:
        encrypted_data = f.read()
    insert_into_database(db_name, image_filename, identity, attributes, encrypted_data)
```

### Query Matching

Due to the division operation involved in calculating cosine similarity, decryption is required before obtaining the final result. Therefore, we opted to use **Euclidean distance** as the measure of similarity between images.

The Euclidean distance formula between image matrices A and B is as follows:

$d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$

Based on this, we can write the matching code:

```python
def compute_euclidean_distance(enc1, enc2, context):
    sk = context.secret_key()
    
    # Calculate the difference between the two vectors
    diff_enc = enc1 - enc2
    diff = diff_enc.decrypt(sk)
    # Calculate the square of the differences
    diff_squared = [x**2 for x in diff]
    diff_squared_sum = sum(diff_squared)

    return diff_squared_sum
```

If the best match image is found, we need to display some relevant information about this image, including the similarity.

We need an additional function to convert the Euclidean distance into a percentage similarity:

```python
def distance_to_similarity(distance, max_distance):
    if distance >= max_distance:
        return 0
    else:
        return (1 - distance / max_distance) * 100
...

if best_match:
    decrypted_image = decrypt_image(best_match[1], algorithm, context)
    similarity = distance_to_similarity(min_distance, max_distance)
    output_filename = os.path.splitext(query_filename)[0] + "_matched_with_" + os.path.splitext(best_match[0])[0] + ".jpg"
    output_file_path = os.path.join(output_folder, output_filename)
    Image.fromarray(decrypted_image).save(output_file_path)
    print(f"Query image '{query_filename}' best matches with database image '{best_match[0]}'")
    print(f"Similarity is '{similarity:.2f}%'")
    identity_tag, feature_tags = get_image_tags(db_name, best_match[0])
    print(f"Identity Tag: {identity_tag}")
    print(f"Feature Tags: {', '.join(feature_tags)}\n")
    print("=== Matching and decryption completed. ===\n")
```

### Innovative Points

#### Simulating Real-World Scenarios

In this project, we particularly focus on simulating real-world data processing and privacy protection scenarios. By building a complete process involving homomorphic encryption and image matching, we simulate an image querying and matching service in a real-world scenario. This not only demonstrates the practicality of homomorphic encryption technology but also provides a feasible solution for data privacy protection.

In this simulated scenario, we clearly distinguish between the roles of the server and the client. The server is responsible for the encrypted storage and matching of image data, while the client provides the query image. This role allocation allows us to demonstrate how to securely exchange and process information between the server and client against a backdrop of data privacy protection.

The setting of the `--context` option is crucial for simulating the real-world distribution of keys in the encryption and matching programs:

```python
def load_context(file_path):
    print("Loading context from: ", file_path)
    with open(file_path, "rb") as f:
        return ts.Context.load(f.read())
...
if args.context:
    context = load_context(args.context)
else:
    if args.algorithm == 'BFV':
        context = ts.context(ts.SCHEME_TYPE.BFV, args.poly_modulus_degree, args.plain_modulus, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60])
    elif args.algorithm == 'CKKS':
        context = ts.context(ts.SCHEME_TYPE.CKKS, args.poly_modulus_degree, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60])
        context.global_scale = args.global_scale
    # Generate public and private keys
    context.generate_galois_keys()
    # Serialize and save context with private key
    save_context(context, os.path.join(keys_folder, "private_context"))
    # Remove private key
    context.make_context_public()
    # Serialize and save context with public key
    save_context(context, os.path.join(keys_folder, "public_context"))
```

Additionally, our system design allows for efficient image matching while fully protecting the privacy of image data. By using homomorphic encryption technology, all image data (including images in the database and query images) are processed in an encrypted state. This method ensures that even in the event of security vulnerabilities during data transmission or storage, the content of the images remains secure.

#### Optimized Encrypted Data Storage Structure

The size of data encrypted with fully homomorphic encryption significantly increases. To address this issue, an appropriate method for storing encrypted data is needed. Compared to directly storing encrypted data and annotations in the filesystem, **storing them in a database

 is more efficient**, and it enables **quicker retrieval of corresponding data labels during matching queries**.

Our algorithm uses the `DBUploader.py` script to store the encrypted image data and corresponding label information in a SQLite database. In the database design, we created a table to store the encrypted image data and labels. This table contains the following fields: image filename, encrypted image data, image label information.

```python
for i, filename in enumerate(files):
    file_number = re.sub('[^0-9]', '', filename)
    image_filename = file_number + '.jpg'
    identity = identity_data.get(image_filename)
    attributes = attr_data.get(image_filename)
    if identity is not None and attributes is not None:
        image_path = os.path.join(input_folder, filename)
        with open(image_path, "rb") as f:
            encrypted_data = f.read()
        insert_into_database(db_name, image_filename, identity, attributes, encrypted_data)
```

I recommend using **DB Browser (SQLite)** to observe and verify the generated database file.

#### Adjusting Encryption Algorithms and Parameters through Options

In this project, we have implemented a flexible mechanism for selecting homomorphic encryption algorithms. Through command-line options or a program interface, users can choose different homomorphic encryption algorithms (such as BFV or CKKS) according to their needs. This dynamic selection mechanism allows the project to adapt to different application scenarios and security requirements.

In addition to selecting different encryption algorithms, users can also adjust various encryption parameters, such as the polynomial modulus degree (poly_modulus_degree), plaintext modulus (plain_modulus), and global scale (global_scale). These parameters directly affect the performance and security of encryption. During experimental testing, we do not need to modify the source files but can simply enter different parameter options to adjust the algorithm parameters.

At the code level, this functionality is achieved by parsing the user-inputted parameters. For example, in the `tenseal_encrypt.py` script, thanks to Python's **argparse** library, we can easily receive and process user input for the encryption algorithm and parameters.

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", choices=["BFV", "CKKS"], help="Encryption algorithm to use")
parser.add_argument("-p", "--poly_modulus_degree", type=int, help="Polynomial modulus degree")
parser.add_argument("-m", "--plain_modulus", type=int, help="Plaintext modulus (for BFV)")
parser.add_argument("-s", "--global_scale", type=int, help="Global scale (for CKKS)")
args = parser.parse_args()

# Set encryption algorithm and parameters based on user choice
if args.algorithm == "BFV":
    # Set BFV algorithm parameters
elif args.algorithm == "CKKS":
    # Set CKKS algorithm parameters
```

Of course, there are also options for users to choose the directory of the original images, the storage directory of the encrypted data, etc.

#### Intuitive and Detailed Display

We provide clear performance metrics by precisely measuring the time consumed in each step of the encryption, matching, and decryption processes. Recording the time required to encrypt a single image, for instance, helps users understand the impact of different parameter settings on encryption speed. To enhance user experience during the processing of large datasets, we implemented a **progress bar** to visually display the current progress. Additionally, during the image matching process, apart from displaying the results, we also provide **detailed label information of the matched images**. This is particularly important for evaluating algorithm efficiency, as it shows not only which images have been matched but also provides **extra information about the quality of the matches**.

The specific implementation is as follows:

1. **Time Display**:
   
   We record the current time at the beginning of each step and again at the end, calculating the duration of the step from the difference between these two times.

   ``` python
   import time
   
   start_time = time.time()
   # Encryption or other processing
   end_time = time.time()
   
   elapsed_time = end_time - start_time
   print(f"Time taken for this step: {elapsed_time} seconds")
   ```

2. **Progress Bar Display**:

   For processing large amounts of data, it is essential to display a progress bar to show the completion progress. We crafted a progress bar code segment **without using the ready-made tqdm library**. This was mainly for ease of controlling when the progress bar updates (for instance, increasing the bar every 1% when dealing with a lot of data).

   ``` python
   # Update the progress bar
   percent_done = (i + 1) / total_files
   bar_length = 50
   block = int(bar_length * percent_done)
   text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done * 100)
   sys.stdout.write(text)
   sys.stdout.flush()
   ```

3. **Label Information**:

   In the image matching process, in addition to displaying the results, we can also show detailed label information of the matched images, which helps users to more comprehensively assess the performance of the matching algorithm.

   ``` python
   # Retrieve label information from the database
   def get_image_tags(image_id):
       # Database query logic
       return identity_tag, feature_tags
   
   identity_tag, feature_tags = get_image_tags(db_name, best_match[0])
   print(f"Identity Tag: {identity_tag}")
   print(f"Feature Tags: {', '.join(feature_tags)}\n")
   ```

## Experiments and Analysis

### Experimental Environment

The configuration of the machine used for the experiments is as follows:

- CPU: AMD Ryzen 7 5800H
- GPU: NVIDIA RTX 3070 Laptop 8G
- Memory: 16GB
- Operating System: Windows10
- Python Version: 3.8.18

We sequentially encrypted images, stored them in a database, and performed query matching to compare the efficiency of different algorithms.

First, we tested the time for BFV.

### BFV

The encryption parameters for BFV are as follows, ensuring the encryption capacity can accommodate a 64*64*3 size image vector:

``` python
poly_modulus_degree = 32768
plain_modulus = 65537
coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
```

Initially, we simulate the server-side data initialization operation, i.e., encrypting image data and storing it in the database.

Execute the following command to encrypt images using BFV fully homomorphic encryption:

``` shell
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs
```

This uses the default parameters, but you can specify them directly if preferred:

``` shell
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs -p 32768 -m 65537
```

![image-20231210222324400](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210222324400.png)

It was found that **the encryption time for a single image is 0.019 seconds**.

In the `keys` directory, we can see the public and private keys generated during the initialization of the context.

![image-20231210223816001](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223816001.png)

In the output directory `data/enc_storage_imgs`, you can view the encrypted data.

![image-20231210222516240](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210222516240-170221831701

71.png)

The encryption data size for a single image is about 2MB, approximately **1000 times the original image size**.

![image-20231210222559511](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210222559511.png)

Run the `DBUploader.py` program to store the encrypted data in the SQLite database. This generates a database file in the project's root directory.

``` python
python .\DBuploader.py -i .\data\enc_storage_imgs\
```

![image-20231210223335373](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223335373.png)

We can observe the successful creation of the database file `encrypted_images.db`.

![image-20231210223416616](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223416616.png)

Next, we simulate the client-side operation.

Encrypt the images in the `data/matching_imgs` folder for matching.

``` shell
python tenseal_encrypt.py -a BFV -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
```

This uses the public key generated in the `keys` directory by the previous command.

![image-20231210224301515](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210224301515.png)

The encryption time for encrypting the images to be matched is about 0.021 seconds per image.

Finally, the query operation.

Execute the following command, saving the query results in the `data/result` directory.

``` shell
python .\tenseal_matching.py -a BFV -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

![image-20231211005417766](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005417766.png)

The **matching operation (i.e., homomorphic computation) time for a single image is 0.02 seconds, and the decryption time is about 0.007 seconds**.

Our set of images to be matched includes two that are already in the database and three that are not.

The images existing in the database are successfully matched to the same images.

![image-20231211005551939](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005551939.png)

For those not in the database, a match with a high degree of similarity is found.

![image-20231211005609661](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005609661.png)

We can open the `data/matching_imgs` and `data/result` folders to compare the results.

![image-20231211001850485](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211001850485.png)

![image-20231211001900621](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211001900621.png)

The matched images show similarity in terms of facial position and hairstyle, **consistent with the expected outcome of Euclidean distance matching**.

### CKKS

The encryption parameters for CKKS are as follows:

```python
global_scale = 2**40
```

Run the following commands in sequence:

```shell
python tenseal_encrypt.py -a CKKS -i data/storage_imgs -o data/enc_storage_imgs
python .\DBuploader.py -i .\data\enc_storage_imgs\
python tenseal_encrypt.py -a CKKS -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
python .\tenseal_matching.py -a CKKS -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

It was observed that **the encryption time for a single image is approximately 0.024 seconds**.

![image-20231211013257836](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211013257836-170229596159633.png)

The storage size is also around 2MB.

![image-20231211013436177](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211013436177-170229596159635.png)

Testing the query matching:

![image-20231211015531385](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211015531385-170229596159637.png)

The **matching operation (i.e., encrypted computation) time for a single image is 0.02 seconds, and the decryption time is approximately 0.008 seconds**.

![image-20231211111653823](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211111653823.png)

The final recognition results are consistent with those of BFV.

### Comparison of Different Parameters

The collected data are presented in a more intuitive table:

| Algorithm | Encryption Time per Image | Encrypted Image Size | Matching Time per Image | Decryption Time per Image |
| --------- | ------------------------- | -------------------- | ----------------------- | ------------------------- |
| BFV       | 0.019s                    | 2MB                  | 0.02s                   | 0.007s                    |
| CKKS      | 0.024s                    | 2MB                  | 0.019s                  | 0.008s                    |

When the parameter settings are similar, BFV and CKKS algorithms have similar effects in terms of encrypting image data, indicating that these two algorithms have a certain equivalence in handling such data.

However, **BFV is faster than CKKS in terms of encryption time**.

This phenomenon may be due to CKKS being an algorithm used on floating-point data, and it does not support integer operations as well as BFV.

We adjusted the `plain_modulus` parameter of BFV to 786433, and the `global_scale` parameter of CKKS to 2**60.

![image-20231211143026527](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211143026527.png)

The table below shows the test data:

| Algorithm | plain_modulus | global_scale | Encryption Time per Image |
| --------- | ------------- | ------------ | ------------------------- |
| BFV       | 65537         |              | 0.019s                    |
| BFV       | 786433        |              | 0.021s                    |
| CKKS      |               | 2**40        | 0.024s                    |
| CKKS      |               | 2**60        | 0.024s                    |

It is observed that as the ciphertext space increases and the security of encryption increases, the encryption time of BFV increases more significantly.

In summary, BFV is more suitable for scenarios requiring precise integer calculations, while CKKS is better suited for floating-point calculations that can tolerate a certain level of approximation. In terms of encryption efficiency, BFV is more efficient than CKKS when dealing with large data volumes, and CKKS is more sensitive to the size of the ciphertext space.

## Contributions

Contributions to this project are welcome! If you have any suggestions for improvement or encounter any issues, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0.

For more details, see the [LICENSE](../LICENSE) file.