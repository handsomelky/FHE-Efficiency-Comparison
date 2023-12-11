# 全同态加密算法效率测试 - FHE Efficiency Comparison

本项目是一个创新的图像加密与匹配系统，基于两种先进的全同态加密算法：BFV和CKKS。它旨在展示全同态加密技术在安全图像处理和隐私保护领域的应用潜力。

项目设计的系统不仅演示了同态加密技术在安全图像处理和隐私保护方面的能力，还通过实际测试为这些算法的性能提供了定量的评估。

本项目的一大亮点是它能够在图像数据保持加密的状态下，执行图像匹配操作。图像匹配操作能够很好地测试同态加密在实际应用中的性能和实用性。通过本项目，我们旨在为全同态加密技术在现实世界应用中的进一步发展提供实验基础和启发。

[English](../README.md) | [繁體中文](README.zh-TW.md) | 简体中文

## 目录

+ [功能特性](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#功能特性)

+ [安装](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#安装)

+ [使用方法](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#使用方法)
  + [tenseal_encrypt.py](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#tenseal_encryptpy)
  + [DBUploader.py](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#dbuploaderpy)
  + [tenseal_matching.py](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#tenseal_matchingpy)

+ [示例](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#示例)

+ 设计思路
  + [项目设计](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#项目设计)
  + [环境安装](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#环境安装)
  + [准备测试数据](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#准备测试数据)
  + [加密数据](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#加密数据)
  + [存入数据库](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#存入数据库)
  + [查询匹配](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#查询匹配)
  + [创新点](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#创新点)

+ 实验及分析
  + [实验环境](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#实验环境)
  + [BFV](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#bfv)
  + [CKKS](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#ckks)
  + [对比不同参数](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#对比不同参数)

+ [贡献](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#贡献)

+ [许可证](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-CN.md#许可证)

## 功能特性

- **全同态加密算法的实现与测试**：项目实现了BFV和CKKS两种全同态加密算法，提供了这两种算法的测试和效率比较。
- **图像预处理**：包括一个图像预处理步骤，将图像裁剪并压缩到特定分辨率以适应加密算法。
- **高效的加密数据存储与管理**：利用SQLite数据库来存储和管理加密后的图像数据及其标签信息。
- **模拟实际应用场景**：模拟真实世界的图像查询服务，展现同态加密在实际数据处理和隐私保护场景中的应用能力。
- **灵活的加密参数配置**：提供可自定义的加密参数设置，使用户能够根据不同的应用需求调整加密强度来测试性能。
- **详细的性能度量与进度显示**：在加密、存储、匹配及解密过程中，展示详细的性能度量和进度信息。

## 安装

要运行本项目，需要在系统上安装Python和几个依赖库。

- Python 3.8
- NumPy
- Pillow
- TenSEAL

您可以使用以下命令安装所有所需的包：

```shell
pip install -r requirements
```

## 使用方法

在本项目中，不同的Python脚本提供了多种命令行选项，允许用户根据自己的需求定制化运行。以下是各个脚本的选项参数及其作用的详细说明：

### `tenseal_encrypt.py`

这个脚本用于加密图像。它接受以下参数：

- `-a` 或 `--algorithm`：指定加密算法。可选值为`BFV`或`CKKS`。例如，`-a BFV`。
- `-i` 或 `--input_folder`：指定包含要加密的图像的文件夹。例如，`-i data/storage_imgs`。
- `-o` 或 `--output_folder`：指定加密图像的输出文件夹。例如，`-o data/enc_storage_imgs`。
- `-p` 或 `--poly_modulus_degree`：用于设置多项式模数度。例如，`-p 32768`。
- `-m` 或 `--plain_modulus`：仅当选择BFV算法时使用，用于设置明文模数。例如，`-m 65537`。
- `-s` 或 `--global_scale`：仅当选择CKKS算法时使用，用于设置全局比例因子。例如，`-s 2**40`。
- `-c` 或 `--context`：可选参数，指定一个包含公钥的文件的路径，用于加密图像。例如，`-c keys/public_context`。

### `DBUploader.py`

这个脚本用于将加密后的图像和对应的标签上传到SQLite数据库。它接受以下参数：

- `-i` 或 `--input_folder`：指定包含加密图像的文件夹。例如，`-i data/enc_storage_imgs`。
- `-db` 或 `--database`：指定数据库文件的名称。例如，`-db encrypted_images.db`。

### `tenseal_matching.py`

这个脚本用于在数据库中匹配加密的图像。它接受以下参数：

- `-a` 或 `--algorithm`：指定用于加密图像的算法。可选值为`BFV`或`CKKS`。例如，`-a BFV`。
- `-i` 或 `--input_folder`：指定包含要匹配的加密图像的文件夹。例如，`-i data/enc_matching_imgs`。
- `-o` 或 `--output_folder`：指定匹配结果的输出文件夹。例如，`-o data/result`。
- `-c` 或 `--context`：指定一个包含私钥的文件的路径，用于解密匹配到的图像。例如，`-c keys/private_context`。
- `-db` 或 `--database`：指定数据库文件的名称。例如，`-db encrypted_images.db`。

使用这些选项参数，用户可以灵活地处理不同的数据集、加密算法和参数，以及输出选项，从而在各种场景下测试和使用全同态加密技术。

## 示例

以下是测试BFV算法的示例：

```shell
# 使用BFV算法加密存储图像
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs
# 将加密后的图像数据上传到数据库
python .\DBuploader.py -i .\data\enc_storage_imgs\
# 加密用于匹配的图像
python tenseal_encrypt.py -a BFV -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
# 在数据库中匹配加密图像，并输出结果
python .\tenseal_matching.py -a BFV -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

## 设计思路

我已经将完整的设计思路上传到了博客中：[开源库全同态加密算法效率比较 | R1ck's Portal (rickliu.com)](https://rickliu.com/posts/3c0eb5a1ba70/)

### 项目设计

本实验选择使用TenSEAL作为开源同态加密库，其代码托管于GitHub，链接如下：

**TenSEAL**：https://github.com/OpenMined/TenSEAL

该同态加密的开源库中实现了**BFV和CKKS这两种全同态加密算法**

实验流程如下：

1. **环境安装**：安装开源库TenSEAL及其依赖环境
2. **准备测试数据**：搜集一组图片当作测试数据，并将其预处理为64*64的分辨率
3. **加密数据**：将测试图片使用选定的算法加密，记录花费的时间
4. **存入数据库**：将加密数据与对应标签存入数据库
5. **查询匹配**：加密查询图片，将其与数据库中的加密图像依次匹配，选取其中最相似的图片，并将其解密并展示，记录匹配的时间以及解密的时间，
6. **分析对比算法效率**：分析各步骤的时间，以对比不同算法的效率和优缺点。

其中第2、3、4、5步需要对不同算法重复进行

这里我们还需要明确加解密的流程

因为我们想要**模拟一个查询图片服务的现实场景**

加密图片会储存在服务端的数据库，而需要匹配的图片则由用户端提供

在这种角色关系中，服务端相当于权威方，所以**一开始的公私钥生成在服务端进行**

服务端使用公钥加密存储的图片，并**将公钥分发给客户端**，将私钥保留

客户端使用公钥加密测试图片后发送給服务端，**服务端**将其与数据库中的加密图片匹配，找到最相似的一张后**使用私钥解密得到原图并返回给客户端**

整个过程可以用以下流程图来表示

![项目流程](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/%E9%A1%B9%E7%9B%AE%E6%B5%81%E7%A8%8B-17013208703374.png)

当然这只是模拟现实世界的一种场景，在其他方案中公私钥的保存者和数据传输状态都不是固定的，**本方案中弱化了对用户提供的数据的保护**（可以通过额外的加密算法来提供安全性），而是将**保护重心放在了数据库存储的图片上**。

在我们的方案中，**同态加密具体体现在图片的密态匹配过程**，若原图的相似度最高，则密态下计算出的相似度也是最高的

### 环境安装

创建一个新的python环境，并使用pip安装TenSEAL

``` shell
conda create -n TenSEAL python=3.8
conda activate TenSEAL
pip install tenseal
```

![image-20231129084420443](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231129084420443.png)

### 准备测试数据

为了**模拟现实场景中**同态加密存储**隐私信息**的应用，我们使用**CelebA(CelebFaces Attribute)**数据集中的2500张人脸图片作为测试数据

下面是数据集的下载地址

https://pan.baidu.com/s/1eSNpdRG#list/path=/

我们选择`Img/img_align_celeba.zip`，即经过裁剪的JPG格式数据集

选取前2500张图片存入数据库

此时图片分辨率为178*218

我们可以写一个python脚本将图片裁剪压缩为64*64

在环境中安装图像处理库Pillow

``` shell
pip install Pillow
```

我们使用`crop()`方法来裁剪图片，然后使用`resize()`进行缩放

``` python
from PIL import Image
import os

def resize_images(folder_path, output_folder, output_size=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 或者根据实际情况调整文件格式
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(image_path) as img:
                min_side = min(img.width, img.height)
                # 计算裁剪区域
                left = (img.width - min_side) / 2
                top = (img.height - min_side) / 2
                right = left + min_side
                bottom = top + min_side

                # 裁剪和缩放
                img_cropped = img.crop((left, top, right, bottom))
                img_resized = img_cropped.resize((output_size))

                # 保存处理后的图片
                img_resized.save(output_path)

resize_images('data/storage_imgs', 'data/storage_imgs')
```

同时，我们截取`Anno/identity_CelebA.txt`和`Anno/list_attr_celeba.txt`的前1000项数据作为label

### 加密数据

一开始处理图像时，我们将其转换为numpy数组，这需要我们安装numpy库

``` shell
pip install numpy
```

对于加密算法，我们需要进行判断，因为不同的算法初始化上下文时的参数不同

``` python
if args.context:
    context = load_context(args.context)

else:
    if args.algorithm == 'BFV':
        context = ts.context(ts.SCHEME_TYPE.BFV, args.poly_modulus_degree, args.plain_modulus)
    elif args.algorithm == 'CKKS':
        context = ts.context(ts.SCHEME_TYPE.CKKS, args.poly_modulus_degree)
        context.global_scale = args.global_scale
```

当然如果已经提供了带公钥的的上下文，我们直接赋值即可

接下来是生成并保存公私钥

``` python
def save_context(context, file_path):
    with open(file_path, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
...
# 生成公私钥
context.generate_galois_keys()

# 序列化并保存包含私钥的上下文
save_context(context, os.path.join(keys_folder, "private_context"))

# 移除私钥
context.make_context_public()

# 序列化并保存公钥上下文
save_context(context, os.path.join(keys_folder, "public_context"))
```

在保存为上下文的文件时，我们可以使用`serialize`方法将上下文序列化，同时我们要设置save_secret_key为true，这样能保证私钥上下文在序列化时依然保留私钥

首先我们需要读取待加密的图片，并将其碾平为一维数组

``` python
# 加载图片并转换为 numpy 数组
with Image.open(image_path) as img:
    img_data = np.array(img)
    
img_data_flattened = img_data.flatten()
```

BFV和CKKS的加密函数分别为`bfv_vector`和`ckks_vector`，我们将上下文`context`和待加密的图像数组即可

``` python
if algorithm == 'BFV':
    encrypted_data = ts.bfv_vector(context, img_data_flattened.tolist())  # 加密整个向量
elif algorithm == 'CKKS':
    encrypted_data = ts.ckks_vector(context, img_data_flattened.tolist())
else:
    raise ValueError("Unsupported algorithm. Please choose 'BFV' or 'CKKS'.")
```

最后我们将加密数据序列化后保存

``` python
encrypted_data_serialize = encrypted_data.serialize()

encrypted_filename = os.path.splitext(os.path.basename(image_path))[0] + "_encrypted"
encrypted_file_path = os.path.join(output_folder, encrypted_filename)

with open(encrypted_file_path, "wb") as f:
    f.write(encrypted_data_serialize)
```

当然，在加密时，我们使用python自带的time库来统计加密时间

``` python
start_time = time.time()  # 开始计时
...
end_time = time.time()
encryption_time = end_time - start_time
```

在代码最后统计并输出总加密时间以及单张图像的加密时间

``` python
total_time, count = encrypt_images_in_folder(args.input_folder, args.algorithm, context, args.output_folder)
if count > 0:
    average_time = total_time / count
    print(f"Total encryption time: {total_time:.3f} seconds")
    print(f"Average encryption time per image: {average_time:.3f} seconds")
    print(f"Number of images encrypted: {count}")
else:
    print("No images were encrypted.")
```



### 存入数据库

在数据库上，我们**选用SQLite**来模拟数据的存储场景

选择SQLite而非MySQL的原因主要是本项目的规模较小，且SQLite搭建时更加方便，最重要的是**python内置sqlite3库**

在现实场景中，一个用户之所以使用查询图片服务，当然是希望获取这张图片的一些信息

所以在数据库中的每一项，我们**除了存入加密的图片向量，还需要存入该图片的一些label**

正好CelebA提供人像图片的特征标签以及身份标签，即`Anno/list_attr_celeba.txt`和`Anno/identity_CelebA.txt`两个文件

首先创建数据库

``` python
def create_database(db_name):

    print(f"=== Creating database:{db_name} ===", )
    if os.path.exists(db_name):
        # 如果存在，则删除该文件
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

我们在处理图像前处理两个文件中的标签，存为数据结构

``` python
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
        # 读取并忽略第一行
        next(file)
        attr_names = file.readline().strip().split()
        for line in file:
            parts = line.strip().split()
            attributes[parts[0]] = dict(zip(attr_names, parts[1:]))
    return attributes
...
identity_data = parse_identity_file("Anno/identity_CelebA.txt")
attr_data = parse_attr_file("Anno/list_attr_celeba.txt")
# 处理图像
process_images(args.input_folder, args.database, identity_data, attr_data)
```

在加密完图片后，查找该图片的标签，并将加密数据与对应标签一并存入数据库

``` python
identity = identity_data.get(image_filename)
attributes = attr_data.get(image_filename)
if identity is not None and attributes is not None:
    image_path = os.path.join(input_folder, filename)
    with open(image_path, "rb") as f:
    	encrypted_data = f.read()
    insert_into_database(db_name, image_filename, identity, attributes, encrypted_data)
```

### 查询匹配

由于余弦相似度的计算设计除法，在获得最终结果前必须解密，所以我们选择使用**欧几里得距离**来表示图像间的相似性

图像矩阵A和B之间的欧式距离计算公式如下

$d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$

由此我们可以写出匹配的代码

``` python
def compute_euclidean_distance(enc1, enc2, context):
    sk = context.secret_key()
    
    # 计算两个向量的差异
    diff_enc = enc1 - enc2
    diff = diff_enc.decrypt(sk)
    # 计算差异的平方
    diff_squared = [x**2 for x in diff]
    diff_squared_sum = sum(diff_squared)

    return diff_squared_sum
```

如果找到最佳匹配图像后，我们需要显示这个图像的一些相关信息

包括相似度

我们需要额外写一个函数，将欧氏距离转化为相似度百分数

``` python
def distance_to_similarity(distance, max_distance):
    if distance >= max_distance:
        return 0
    else:
        return (1 - distance / max_distance) * 100
...

if best_match:
    decrypted_image = decrypt_image(best_match[1], algorithm, context)
    similarity = distance_to_similarity(min_distance,max_distance)
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

### 创新点

#### 模拟现实场景

在本项目中，我们特别注重模拟真实世界中的数据处理和隐私保护场景。通过构建一个包含同态加密和图像匹配的完整流程，我们模拟了一个现实场景中的图像查询和匹配服务。这不仅展示了同态加密技术的实用性，也为数据隐私保护提供了切实可行的解决方案。

在这个模拟的场景中，我们明确区分了服务端和客户端的角色。服务端负责图像数据的加密存储和匹配处理，而客户端则负责提供查询图像。通过这种角色分配，我们能够展示在数据隐私保护的背景下，如何在服务端和客户端之间安全地交换和处理信息。

通过选项`--context`的设置，加密程序以及匹配程序能够使用之前程序产生的密钥文件，这一点对于模拟真正场景下的密钥分配至关重要

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
        # 生成公私钥
        context.generate_galois_keys()
        # 序列化并保存包含私钥的上下文
        save_context(context, os.path.join(keys_folder, "private_context"))
        # 移除私钥
        context.make_context_public()
        # 序列化并保存公钥上下文
        save_context(context, os.path.join(keys_folder, "public_context"))
```

同时，我们的系统设计允许在完全保护图像数据隐私的前提下进行高效的图像匹配。通过使用同态加密技术，所有图像数据（包括数据库中的图像和查询图像）都在加密状态下进行处理。这种方法确保了即使在数据传输或存储过程中发生安全漏洞，图像内容也不会泄露。

#### 优化的加密数据存储结构

全同态加密加密后的数据体积显著增加。为了解决这个问题，需要选择合适的方式存储加密数据。相比于直接将加密后的数据和annotations存在文件系统中，**存在数据库中时的存储效率更高**，而且在匹配查询时能**更快捷的查找到相应数据的标签**。

本算法使用`DBUploader.py`脚本将加密后的图像数据和对应的标签信息存储到SQLite数据库中。数据库设计上，我们创建了一个表来存储加密的图像数据和标签。这个表包含以下几个字段：图像文件名、加密图像数据、图像的标签信息。

``` python
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

这里，我推荐使用**DB Broswer(SQLite)**来观察和验证生成的数据库文件

#### 通过选项来调整加密算法及参数

在本项目中，我们实现了一种灵活的加密算法选择机制。通过命令行选项或程序界面，用户可以根据需要选择不同的同态加密算法（如BFV或CKKS）。这种动态选择机制使得项目能够适应不同的应用场景和安全需求。

除了选择不同的加密算法外，用户还可以调整各种加密参数，如多项式模数度（poly_modulus_degree）、明文模数（plain_modulus）和全局比例因子（global_scale）。这些参数对加密的性能和安全性有着直接影响。在实验测试的过程中，我们不需要修改源文件，而是简单地输入不同参数选项即可调整算法参数。

在代码层面，这一功能是通过解析用户输入的参数实现的。例如，在`tenseal_encrypt.py`脚本中，得益于Python的**argparse库**，我们可以很方便地接收和处理用户输入的加密算法和参数选项。

``` python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", choices=["BFV", "CKKS"], help="Encryption algorithm to use")
parser.add_argument("-p", "--poly_modulus_degree", type=int, help="Polynomial modulus degree")
parser.add_argument("-m", "--plain_modulus", type=int, help="Plaintext modulus (for BFV)")
parser.add_argument("-s", "--global_scale", type=int, help="Global scale (for CKKS)")
args = parser.parse_args()

# 根据用户选择设置加密算法和参数
if args.algorithm == "BFV":
    # 设置BFV算法参数
elif args.algorithm == "CKKS":
    # 设置CKKS算法参数
```

当然，还有一些选项供用户选择原图像的目录、加密数据的存储目录等

#### 显示直观且详细

我们通过精确测量加密、匹配以及解密过程中**每一步骤所消耗的时间**，为用户提供了一个明确的性能指标。例如，记录加密单张图片所需的时间，可以**帮助用户理解不同参数设置对加密速度的影响**。为了提高用户体验，在处理大量数据时，我们实现了一个**进度条**来直观地展示当前操作的进度。在图像匹配过程中，除了展示匹配结果，我们还提供了**匹配图像的详细标签信息**。这一点在测试算法效率时尤为重要，因为它不仅显示了哪些图像被匹配到，还**提供了关于匹配质量的额外信息**。

具体实现如下：

1. **时间展示**：

   在每个步骤开始前记录当前时间，步骤结束后再次记录时间，通过两者的差值计算步骤耗时

   ``` python
   import time
   
   start_time = time.time()
   # 加密或其他处理过程
   end_time = time.time()
   
   elapsed_time = end_time - start_time
   print(f"该步骤耗时: {elapsed_time}秒")
   ```

2. **进度条显示**：

   对于处理大量数据的情况，通过进度条来展示完成进度很有必要，我们**没有使用现成的tqdm库，而是手搓了一段进度条代码**，主要是为了方便控制进度条增长的条件（当数据很多时，可以设置每隔1%进度条增长一次）

   ``` python
   # 更新进度条
   percent_done = (i + 1) / total_files
   bar_length = 50
   block = int(bar_length * percent_done )
   text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done*100)
   sys.stdout.write(text)
   sys.stdout.flush()
   ```

3. **标签信息**：

   在图像匹配过程中，除了显示匹配结果外，还可以显示匹配图像的详细标签信息，这有助于用户更全面地评估匹配算法的效果。

   ``` python
   # 从数据库中提取标签信息
   def get_image_tags(image_id):
       # 数据库查询逻辑
       return identity_tag, feature_tags
   
   identity_tag, feature_tags = get_image_tags(db_name, best_match[0])
   print(f"Identity Tag: {identity_tag}")
   print(f"Feature Tags: {', '.join(feature_tags)}\n")
   ```

## 实验及分析

### 实验环境

下面是进行实验时，本机的配置

- CPU: AMD Ryzen 7 5800H
- GPU: NVIDIA RTX 3070 Laptop 8G
- Memory: 16GB
- Operating System: Windows10
- Python Version: 3.8.18

我们依次对已有图像数据进行加密、存入数据库、查询匹配的操作，并比对不同算法的效率

首先测试BFV的时间

### BFV

BFV的加密参数如下所示，我们需要保证加密容量能够允许加密64*64\*3大小的图像向量

``` python
poly_modulus_degree = 32768
plain_modulus = 65537
coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
```

首先我们模拟服务端对数据的初始化操作，即加密图像数据并存入数据库

执行如下指令，对图像进行BFV全同态加密

``` shell
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs
```

这里使用的是默认参数，当然也可以直接指定参数

``` shell
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs -p 32768 -m 65537
```

![image-20231210222324400](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210222324400.png)

统计得到**单张图像的加密时间为0.019秒**

在`keys`目录下我们可以查看到上下文初始化得到的公钥和私钥

![image-20231210223816001](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223816001.png)

在我们选择的输出目录`data/enc_storage_imgs`下可以查看加密后的数据

![image-20231210222516240](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210222516240-17022183170171.png)

单张图片的加密数据大小为2MB，约**为原图像大小的1000倍**

![image-20231210222559511](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210222559511.png)

运行`DBUploader.py`程序，将加密数据存入SQLite数据库，这里在项目根目录下生成数据库文件

``` python
python .\DBuploader.py -i .\data\enc_storage_imgs\
```

![image-20231210223335373](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223335373.png)

可以观察到成功生成了数据库文件`encrypted_images.db`

![image-20231210223416616](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223416616.png)

接下来我们模拟用户端的操作

将`data/matching_imgs`文件夹下的待匹配图像加密

``` shell
python tenseal_encrypt.py -a BFV -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
```

这里使用的是前面指令在`keys`目录下生成的公钥

![image-20231210224301515](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210224301515.png)

加密待匹配图像时单张图像的加密时间为0.021秒

最后是查询操作

执行如下指令，我们将查询结果保存在`data/result`目录下

``` shell
python .\tenseal_matching.py -a BFV -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

![image-20231211005417766](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005417766.png)

单张图片的**匹配操作（即密态计算）时间为0.02秒，解密时间为0.007秒左右**

我们的待匹配图像中有两张数据库中已有图像，以及三张数据库中未出现过的图像

数据库中存在的图片都成功匹配到相同图片

![image-20231211005551939](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005551939.png)

而不存在的图片则匹配到一个相似度较高的图

![image-20231211005609661](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005609661.png)

我们可以打开`data/matching_imgs`和`data/result`文件夹来对比结果的效果

![image-20231211001850485](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211001850485.png)

![image-20231211001900621](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211001900621.png)

匹配到的图像与原图在人脸的位置以及发型上都较为相似，**符合欧氏距离应有的匹配效果**

### CKKS

BFV的加密参数如下所示

``` python
global_scale = 2**40
```

依次运行以下指令

``` shell
python tenseal_encrypt.py -a CKKS -i data/storage_imgs -o data/enc_storage_imgs
python .\DBuploader.py -i .\data\enc_storage_imgs\
python tenseal_encrypt.py -a CKKS -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
python .\tenseal_matching.py -a CKKS -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

统计得到**单张图像的加密时间为0.024秒左右**

![image-20231211013257836](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211013257836.png)

存储大小同样是2MB左右

![image-20231211013436177](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211013436177.png)

测试查询匹配

![image-20231211015531385](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211015531385.png)

单张图片的**匹配操作（即密态计算）时间为0.02秒，解密时间为0.008秒左右**

![image-20231211111653823](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211111653823.png)

最后的识别结果和BFV一致

### 对比不同参数

将收集的数据列为更为直观的表格

| 算法 | 单张图像加密时间 | 单张加密图像大小 | 单张图像匹配时间 | 单张图像解密时间 |
| ---- | ---------------- | ---------------- | ---------------- | ---------------- |
| BFV  | 0.019s           | 2MB              | 0.02s            | 0.007s           |
| CKKS | 0.024s           | 2MB              | 0.019s           | 0.008s           |

当参数设置相近时，BFV和CKKS算法在图像数据加密结果方面的效果相似，表明这两种算法在处理此类数据时具有一定的等效性。

而**在加密花费的时间上，BFV相对于CKKS速度更快**

造成这种现象的原因，可能是CKKS是运用在浮点级数据上的算法，对整数运算的支持程度并没有BFV高

我们将BFV的`plain_modulus`参数调大一些，设为786433

而CKKS的`global_scale`参数设为2**60

![image-20231211143026527](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211143026527.png)

可以得到下表测试数据

| 算法 | plain_modulus | global_scale | 单张图像加密时间 |
| ---- | ------------- | ------------ | ---------------- |
| BFV  | 65537         |              | 0.019s           |
| BFV  | 786433        |              | 0.021s           |
| CKKS |               | 2**40        | 0.024s           |
| CKKS |               | 2**60        | 0.024s           |

可以发现，当密文空间增大、加密安全性增加时，BFV的加密时间增长幅度更大

总的来说，BFV更适合需要精确整数计算的场景，而CKKS更适合于可以接受一定近似的浮点数计算。在加密的效率上，当数据量较大时，BFV比CKKS的效率更高，且CKKS对密文空间大小的敏感程度更高

## 贡献

欢迎对此项目进行贡献！如果您有改进建议或遇到任何问题，请随时打开一个问题或提交一个拉取请求。

## 许可证

该项目在GNU通用公共许可证v3.0下获得许可

详情请见[LICENSE](../LICENSE)文件。