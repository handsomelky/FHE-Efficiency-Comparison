# 全同态加密演算法效能測試 - FHE Efficiency Comparison

本專案是一個創新的圖像加密與匹配系統，基於兩種先進的全同态加密演算法：BFV和CKKS。它旨在展示全同态加密技術在安全圖像處理和隱私保護領域的應用潛力。

專案設計的系統不僅演示了同态加密技術在安全圖像處理和隱私保護方面的能力，還通過實際測試為這些演算法的性能提供了定量的評估。

本專案的一大亮點是它能夠在圖像數據保持加密的狀態下，執行圖像匹配操作。圖像匹配操作能夠很好地測試同态加密在實際應用中的性能和實用性。通過本專案，我們旨在為全同态加密技術在現實世界應用中的進一步發展提供實驗基礎和啟發。

[English](https://chat.openai.com/README.md) | [繁體中文](https://chat.openai.com/c/README.zh-TW.md) | 簡體中文

## 目錄

- [功能特性](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#功能特性)
- [安裝](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#安裝)
- [使用方法](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#使用方法)
  - [tenseal_encrypt.py](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#tenseal_encryptpy)
  - [DBUploader.py](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#dbuploaderpy)
  - [tenseal_matching.py](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#tenseal_matchingpy)
- [範例](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#範例)

- 設計思路
  - [專案設計](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#專案設計)
  - [環境安裝](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#環境安裝)
  - [準備測試數據](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#準備測試數據)
  - [加密數據](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#加密數據)
  - [存入資料庫](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#存入資料庫)
  - [查詢匹配](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#查詢匹配)
  - [創新點](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#創新點)
- 實驗及分析
  - [實驗環境](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#模擬現實場景)
  - [BFV](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#bfv)
  - [CKKS](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#ckks)
  - [對比不同參數](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#對比不同參數)
- [貢獻](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#貢獻)
- [許可證](https://github.com/handsomelky/FHE-Efficiency-Comparison/blob/main/readme/README.zh-TW.md#許可證)

## 功能特性

- **全同态加密演算法的實現與測試**：專案實現了BFV和CKKS兩種全同态加密演算法，提供了這兩種演算法的測試和效能比較。
- **圖像預處理**：包括一個圖像預處理步驟，將圖像裁剪並壓縮到特定分辨率以適應加密演算法。
- **高效的加密數據存儲與管理**：利用SQLite資料庫來存儲和管理加密後的圖像數據及其標籤信息。
- **模擬實際應用場景**：模擬真實世界的圖像查詢服務，展現同态加密在實際數據處理和隱私保護場景中的應用能力。
- **靈活的加密參數配置**：提供可自定義的加密參數設置，使用戶能夠根據不同的應用需求調整加密強度來測試性能。
- **詳細的性能度量與進度顯示**：在加密、存儲、匹配及解密過程中，展示詳細的性能度量和進度信息。

## 安裝

要運行本專案，需要在系統上安裝Python和幾個依賴庫。

- Python 3.8
- NumPy
- Pillow
- TenSEAL

您可以使用以下命令安裝所有所需的包：

```shell
pip install -r requirements
```

## 使用方法

在本專案中，不同的Python腳本提供了多種命令行選項，允許使用戶根據自己的需求定制化運行。以下是各個腳本的選項參數及其作用的詳細說明：

### `tenseal_encrypt.py`

這個腳本用於加密圖像。它接受以下參數：

- `-a` 或 `--algorithm`：指定加密演算法。可選值為`BFV`或`CKKS`。例如，`-a BFV`。
- `-i` 或 `--input_folder`：指定包含要加密的圖像的文件夾。例如，`-i data/storage_imgs`。
- `-o` 或 `--output_folder`：指定加密圖像的輸出文件夾。例如，`-o data/enc_storage_imgs`。
- `-p` 或 `--poly_modulus_degree`：用於設置多項式模數度。例如，`-p 32768`。
- `-m` 或 `--plain_modulus`：僅當選擇BFV演算法時使用，用於設置明文模數。例如，`-m 65537`。
- `-s` 或 `--global_scale`：僅當選擇CKKS演算法時使用，用於設置全局比例因子。例如，`-s 2**40`。
- `-c` 或 `--context`：可選參數，指定一個包含公鑰的文件的路徑，用於加密圖像。例如，`-c keys/public_context`。

### `DBUploader.py`

這個腳本用於將加密後的圖像和對應的標籤上傳到SQLite資料庫。它接受以下參數：

- `-i` 或 `--input_folder`：指定包含加密圖像的文件夾。例如，`-i data/enc_storage_imgs`。
- `-db` 或 `--database`：指定資料庫文件的名稱。例如，`-db encrypted_images.db`。

### `tenseal_matching.py`

這個腳本用於在資料庫中匹配加密的圖像。它接受以下參數：

- `-a` 或 `--algorithm`：指定用於加密圖像的演算法。可選值為`BFV`或`CKKS`。例如，`-a BFV`。
- `-i` 或 `--input_folder`：指定包含要匹配的加密圖像的文件夾。例如，`-i data/enc_matching_imgs`。
- `-o` 或 `--output_folder`：指定匹配結果的輸出文件夾。例如，`-o data/result`。
- `-c` 或 `--context`：指定一個包含私鑰的文件的路徑，用於解密匹配到的圖像。例如，`-c keys/private_context`。
- `-db` 或 `--database`：指定資料庫文件的名稱。例如，`-db encrypted_images.db`。

使用這些選項參數，使用戶可以靈活地處理不同的數據集、加密演算法和參數，以及輸出選項，從而在各種場景下測試和使用全同态加密技術。

## 範例

以下是測試BFV演算法的範例：

```shell
# 使用BFV演算法加密存儲圖像
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs
# 將加密後的圖像數據上傳到資料庫
python .\DBuploader.py -i .\data\enc_storage_imgs\
# 加密用於匹配的圖像
python tenseal_encrypt.py -a BFV -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
# 在資料庫中匹配加密圖像，並輸出結果
python .\tenseal_matching.py -a BFV -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

## 設計思路

我已經將完整的設計思路上傳到了部落格中：[開源庫全同态加密演算法效能比較 | R1ck's Portal (rickliu.com)](https://rickliu.com/posts/3c0eb5a1ba70/)

### 專案設計

本實驗選擇使用TenSEAL作為開源同态加密庫，其代碼託管於GitHub，鏈接如下：

**TenSEAL**：https://github.com/OpenMined/TenSEAL

該同态加密的開源庫中實現了**BFV和CKKS這兩種全同态加密演算法**

實驗流程如下：

1. **環境安裝**：安裝開源庫TenSEAL及其依賴環境
2. **準備測試數據**：搜集一組圖片作為測試數據，並將其預處理為64*64的分辨率
3. **加密數據**：將測試圖片使用選定的演算法加密，記錄花費的時間
4. **存入資料庫**：將加密數據與對應標籤存入資料庫
5. **查詢匹配**：加密查詢圖片，將其與資料庫中的加密圖像依次匹配，選取其中最相似的圖片，並將其解密並展示，記錄匹配的時間以及解密的時間，
6. **分析對比演算法效能**：分析各步驟的時間，以對比不同演算法的效能和優缺點。

其中第2、3、4、5步需要對不同演算法重複進行

這裡我們還需要明確加解密的流程

因為我們想要**模擬一個查詢圖片服務的現實場景**

加密圖片會儲存在服務端的資料庫，而需要匹配的圖片則由用戶端提供

在這種角色關係中，服務端相當於權威方，所以**一開始的公私鑰生成在服務端進行**

服務端使用公鑰加密存儲的圖片，並**將公鑰分發給客戶端**，將私鑰保留

客戶端使用公鑰加密測試圖片後發送給服務端，**服務端**將其與資料庫中的加密圖像匹配，找到最相似的一張後**使用私鑰解密得到原圖並返回給客戶端**

整個過程可以用以下流程圖來表示

![專案流程](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/%E9%A1%B9%E7%9B%AE%E6%B5%81%E7%A8%8B-17013208703374.png)

當然這只是模擬現實世界的一種場景，在其他方案中公私鑰的保存者和數據傳輸狀態都不是固定的，**本方案中弱化了對用戶提供的數據的保護**（可以通過額外的加密演算法來提供安全性），而是將**保護重心放在了資料庫存儲的圖片上**。

在我們的方案中，**同态加密具體體現在圖片的密態匹配過程**，若原圖的相似度最高，則密態下計算出的相似度也是最高的。

### 環境安裝

創建一個新的python環境，並使用pip安裝TenSEAL

```shell
conda create -n TenSEAL python=3.8
conda activate TenSEAL
pip install tenseal
```

![image-20231129084420443](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231129084420443.png)

### 準備測試數據

為了**模擬現實場景中**同态加密存儲**隱私信息**的應用，我們使用**CelebA(CelebFaces Attribute)**數據集中的2500張人臉圖片作為測試數據

下面是數據集的下載地址

https://pan.baidu.com/s/1eSNpdRG#list/path=/

我們選擇`Img/img_align_celeba.zip`，即經過裁剪的JPG格式數據集

選取前2500張圖片存入資料庫

此時圖片分辨率為178*218

我們可以寫一個python腳本將圖片裁剪壓縮為64*64

在環境中安裝圖像處理庫Pillow

```shell
pip install Pillow
```

我們使用`crop()`方法來裁剪圖片，然後使用`resize()`進行縮放

```python
from PIL import Image
import os

def resize_images(folder_path, output_folder, output_size=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 或者根據實際情況調整文件格式
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(image_path) as img:
                min_side = min(img.width, img.height)
                # 計算裁剪區域
                left = (img.width - min_side) / 2
                top = (img.height - min_side) / 2
                right = left + min_side
                bottom = top + min_side

                # 裁剪和縮放
                img_cropped = img.crop((left, top, right, bottom))
                img_resized = img_cropped.resize((output_size))

                # 保存處理後的圖片
                img_resized.save(output_path)

resize_images('data/storage_imgs', 'data/storage_imgs')
```

同時，我們截取`Anno/identity_CelebA.txt`和`Anno/list_attr_celeba.txt`的前1000項數據作為label。

### 加密數據

一開始處理圖像時，我們將其轉換為numpy陣列，這需要我們安裝numpy庫。

```shell
pip install numpy
```

對於加密演算法，我們需要進行判斷，因為不同的演算法初始化上下文時的參數不同。

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

當然如果已經提供了帶公鑰的的上下文，我們直接賦值即可。

接下來是生成並保存公私鑰。

```python
def save_context(context, file_path):
    with open(file_path, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
...
# 生成公私鑰
context.generate_galois_keys()

# 序列化並保存包含私鑰的上下文
save_context(context, os.path.join(keys_folder, "private_context"))

# 移除私鑰
context.make_context_public()

# 序列化並保存公鑰上下文
save_context(context, os.path.join(keys_folder, "public_context"))
```

在保存為上下文的文件時，我們可以使用`serialize`方法將上下文序列化，同時我們要設置save_secret_key為true，這樣能保證私鑰上下文在序列化時依然保留私鑰。

首先我們需要讀取待加密的圖片，並將其碾平為一維陣列。

```python
# 加載圖片並轉換為 numpy 陣列
with Image.open(image_path) as img:
    img_data = np.array(img)
    
img_data_flattened = img_data.flatten()
```

BFV和CKKS的加密函數分別為`bfv_vector`和`ckks_vector`，我們將上下文`context`和待加密的圖像陣列即可。

```python
if algorithm == 'BFV':
    encrypted_data = ts.bfv_vector(context, img_data_flattened.tolist())  # 加密整個向量
elif algorithm == 'CKKS':
    encrypted_data = ts.ckks_vector(context, img_data_flattened.tolist())
else:
    raise ValueError("Unsupported algorithm. Please choose 'BFV' or 'CKKS'.")
```

最後我們將加密數據序列化後保存。

```python
encrypted_data_serialize = encrypted_data.serialize()

encrypted_filename = os.path.splitext(os.path.basename(image_path))[0] + "_encrypted"
encrypted_file_path = os.path.join(output_folder, encrypted_filename)

with open(encrypted_file_path, "wb") as f:
    f.write(encrypted_data_serialize)
```

當然，在加密時，我們使用python自帶的time庫來統計加密時間。

```python
start_time = time.time()  # 開始計時
...
end_time = time.time()
encryption_time = end_time - start_time
```

在代碼最後統計並輸出總加密時間以及單張圖像的加密時間。

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

### 存入資料庫

在資料庫上，我們**選用SQLite**來模擬數據的存儲場景。

選擇SQLite而非MySQL的原因主要是本專案的規模較小，且SQLite搭建時更加方便，最重要的是**python內置sqlite3庫**。

在現實場景中，一個用戶之所以使用查詢圖片服務，當然是希望獲取這張圖片的一些信息。

所以在資料庫中的每一項，我們**除了存入加密的圖片向量，還需要存入該圖片的一些label**。

正好CelebA提供人像圖片的特徵標籤以及身份標籤，即`Anno/list_attr_celeba.txt`和`Anno/identity_CelebA.txt`兩個文件。

首先創建資料庫。

```python
def create_database(db_name):

    print(f"=== Creating database:{db_name} ===", )
    if os.path.exists(db_name):
        # 如果存在，則刪除該文件
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

我們在處理圖像前處理兩個文件中的標籤，存為數據結構。

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
        # 讀取並忽略第一行
        next(file)
        attr_names = file.readline().strip().split()
        for line in file:
            parts = line.strip().split()
            attributes[parts[0]] = dict(zip(attr_names, parts[1:]))
    return attributes
...
identity_data = parse_identity_file("Anno/identity_CelebA.txt")
attr_data = parse_attr_file("Anno/list_attr_celeba.txt")
# 處理圖像
process_images(args.input_folder, args.database, identity_data, attr_data)
```

在加密完圖片後，查找該圖片的標籤，並將加密數據與對應標籤一併存入資料庫。

```python
identity = identity_data.get(image_filename)
attributes = attr_data.get(image_filename)
if identity is not None and attributes is not None:
    image_path = os.path.join(input_folder, filename)
    with open(image_path, "rb") as f:
    	encrypted_data = f.read()
    insert_into_database(db_name, image_filename, identity, attributes, encrypted_data)
```

### 查詢匹配

由於餘弦相似度的計算設計涉及除法，在獲得最終結果前必須解密，所以我們選擇使用**歐幾里得距離**來表示圖像間的相似性。

圖像矩陣A和B之間的歐式距離計算公式如下：

$d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$

由此我們可以寫出匹配的程式碼：

```python
def compute_euclidean_distance(enc1, enc2, context):
    sk = context.secret_key()
    
    # 計算兩個向量的差異
    diff_enc = enc1 - enc2
    diff = diff_enc.decrypt(sk)
    # 計算差異的平方
    diff_squared = [x**2 for x in diff]
    diff_squared_sum = sum(diff_squared)

    return diff_squared_sum
```

如果找到最佳匹配圖像後，我們需要顯示這個圖像的一些相關信息

包括相似度。

我們需要額外寫一個函數，將歐氏距離轉化為相似度百分數。

```python
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

### 創新點

#### 模擬現實場景

在本專案中，我們特別注重模擬真實世界中的數據處理和隱私保護場景。通過構建一個包含同态加密和圖像匹配的完整流程，我們模擬了一個現實場景中的圖像查詢和匹配服務。這不僅展示了同态加密技術的實用性，也為數據隱私保護提供了切實可行的解決方案。

在這個模擬的場景中，我們明確區分了服務端和客戶端的角色。服務端負責圖像數據的加密存儲和匹配處理，而客戶端則負責提供查詢圖像。通過這種角色分配，我們能夠展示在數據隱私保護的背景下，如何在服務端和客戶端之間安全地交換和處理信息。

通過選項`--context`的設置，加密程序以及匹配程序能夠使用之前程序產生的密鑰文件，這一點對於模擬真正場景下的密鑰分配至關重要。

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
        # 生成公私鑰
        context.generate_galois_keys()
        # 序列化並保存包含私鑰的上下文
        save_context(context, os.path.join(keys_folder, "private_context"))
        # 移除私鑰
        context.make_context_public()
        # 序列化並保存公鑰上下文
        save_context(context, os.path.join(keys_folder, "public_context"))
```

同時，我們的系統設計允許在完全保護圖像數據隱私的前提下進行高效的圖像匹配。通過使用同态加密技術，所有圖像數據（包括資料庫中的圖像和查詢圖像）都在加密狀態下進行處理。這種方法確保了即使在數據傳輸或存儲過程中發生安全漏洞，圖像內容也不會洩露。

#### 優化的加密數據存儲結構

全同态加密加密後的數據體積顯著增加。為了解決這個問題，需要選擇合適的方式存儲加密數據。相比於直接將加密後的數據和annotations存在文件系統中，**存在資料庫中時的存儲效率更高**，而且在匹配查詢時能**更快捷的查找到相應數據的標籤**。

本算法使用`DBUploader.py`腳本將加密後的圖像數據和對應的標籤信息存儲到SQLite資料庫中。資料庫設計上，我們創建了一個表來存儲加密的圖像數據和標籤。這個表包含以下幾個字段：圖像文件名、加密圖像數據、圖像的標籤信息。

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

這裡，我推薦使用**DB Broswer(SQLite)**來觀察和驗證生成的資料庫文件。

#### 通過選項來調整加密演算法及參數

在本專案中，我們實現了一種靈活的加密演算法選擇機制。通過命令行選項或程式界面，使用戶可以根據需要選擇不同

的同态加密演算法（如BFV或CKKS）。這種動態選擇機制使得專案能夠適應不同的應用場景和安全需求。

除了選擇不同的加密演算法外，使用戶還可以調整各種加密參數，如多項式模數度（poly_modulus_degree）、明文模數（plain_modulus）和全局比例因子（global_scale）。這些參數對加密的性能和安全性有著直接影響。在實驗測試的過程中，我們不需要修改源文件，而是簡單地輸入不同參數選項即可調整演算法參數。

在程式碼層面，這一功能是通過解析使用戶輸入的參數實現的。例如，在`tenseal_encrypt.py`腳本中，得益於Python的**argparse庫**，我們可以很方便地接收和處理使用戶輸入的加密演算法和參數。

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", choices=["BFV", "CKKS"], help="Encryption algorithm to use")
parser.add_argument("-p", "--poly_modulus_degree", type=int, help="Polynomial modulus degree")
parser.add_argument("-m", "--plain_modulus", type=int, help="Plaintext modulus (for BFV)")
parser.add_argument("-s", "--global_scale", type=int, help="Global scale (for CKKS)")
args = parser.parse_args()

# 根據使用者選擇設置加密演算法和參數
if args.algorithm == "BFV":
    # 設置BFV演算法參數
elif args.algorithm == "CKKS":
    # 設置CKKS演算法參數
```

當然，還有一些選項供使用者選擇原圖像的目錄、加密數據的存儲目錄等。

#### 顯示直觀且詳細

我們通過精確測量加密、匹配以及解密過程中**每一步驟所消耗的時間**，為使用者提供了一個明確的性能指標。例如，記錄加密單張圖片所需的時間，可以**幫助使用者理解不同參數設置對加密速度的影響**。為了提高使用者體驗，在處理大量數據時，我們實現了一個**進度條**來直觀地展示當前操作的進度。在圖像匹配過程中，除了展示匹配結果，我們還提供了**匹配圖像的詳細標籤信息**。這一點在測試演算法效能時尤為重要，因為它不僅顯示了哪些圖像被匹配到，還**提供了關於匹配質量的額外信息**。

具體實現如下：

1. **時間展示**：

   在每個步驟開始前記錄當前時間，步驟結束後再次記錄時間，通過兩者的差值計算步驟耗時。

   ```python
   import time
   
   start_time = time.time()
   # 加密或其他處理過程
   end_time = time.time()
   
   elapsed_time = end_time - start_time
   print(f"該步驟耗時: {elapsed_time}秒")
   ```

2. **進度條顯示**：

   對於處理大量數據的情況，通過進度條來展示完成進度很有必要，我們**沒有使用現成的tqdm庫，而是手搓了一段進度條代碼**，主要是為了方便控制進度條增長的條件（當數據很多時，可以設置每隔1%進度條增長一次）。

   ```python
   # 更新進度條
   percent_done = (i + 1) / total_files
   bar_length = 50
   block = int(bar_length * percent_done)
   text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done*100)
   sys.stdout.write(text)
   sys.stdout.flush()
   ```

3. **標籤信息**：

   在圖像匹配過程中，除了顯示匹配結果外，還可以顯示匹配圖像的詳細標籤信息，這有助於使用者更全面地評估匹配演算法的效果。

   ```python
   # 從資料庫中提取標籤信息
   def get_image_tags(image_id):
       # 資料庫查詢邏輯
       return identity_tag, feature_tags
   
   identity_tag, feature_tags = get_image_tags(db_name, best_match[0])
   print(f"Identity Tag: {identity_tag}")
   print(f"Feature Tags: {', '.join(feature_tags)}\n")
   ```

## 實驗及分析

### 實驗環境

下面是進行實驗時，本機的配置：

- CPU: AMD Ryzen 7 5800H
- GPU: NVIDIA RTX 3070 Laptop

 8G
- Memory: 16GB
- Operating System: Windows10
- Python Version: 3.8.18

我們依次對已有圖像數據進行加密、存入資料庫、查詢匹配的操作，並比對不同演算法的效能。

首先測試BFV的時間。

### BFV

BFV的加密參數如下所示，我們需要保證加密容量能夠允許加密64*64\*3大小的圖像向量：

```python
poly_modulus_degree = 32768
plain_modulus = 65537
coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
```

首先我們模擬服務端對數據的初始化操作，即加密圖像數據並存入資料庫。

執行如下指令，對圖像進行BFV全同态加密：

```shell
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs
```

這裡使用的是默認參數，當然也可以直接指定參數。

```shell
python tenseal_encrypt.py -a BFV -i data/storage_imgs -o data/enc_storage_imgs -p 32768 -m 65537
```

統計得到**單張圖像的加密時間為0.019秒**。

在`keys`目錄下我們可以查看到上下文初始化得到的公鑰和私鑰。

![image-20231210223816001](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223816001.png)

在我們選擇的輸出目錄`data/enc_storage_imgs`下可以查看加密後的數據。

![image-20231210222516240](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210222516240-17022183170171.png)

單張圖片的加密數據大小為2MB，約**為原圖像大小的1000倍**。

![image-20231210222559511](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210222559511.png)

運行`DBUploader.py`程序，將加密數據存入SQLite資料庫，這裡在項目根目錄下生成資料庫文件。

```python
python .\DBuploader.py -i .\data\enc_storage_imgs\
```

![image-20231210223335373](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223335373.png)

可以觀察到成功生成了資料庫文件`encrypted_images.db`。

![image-20231210223416616](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210223416616.png)

接下來我們模擬用戶端的操作。

將`data/matching_imgs`文件夾下的待匹配圖像加密。

```shell
python tenseal_encrypt.py -a BFV -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
```

這裡使用的是前面指令在`keys`目錄下生成的公鑰。

![image-20231210224301515](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231210224301515.png)

加密待匹配圖像時單張圖像的加密時間為0.021秒。

最後是查詢操作。

執行如下指令，我們將查詢結果保存在`data/result`目錄下。

```shell
python .\tenseal_matching.py -a BFV -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

![image-20231211005417766](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005417766.png)

單張圖片的**匹配操作（即密態計算）時間為0.02秒，解密時間為0.007秒左右**。

我們的待匹配圖像中有兩張資料庫中已有圖像，以及三張資料庫中未出現過的圖像。

資料庫中存在的圖片都成功匹配到相同圖片。

![image-20231211005551939](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005551939.png)

而不存在的圖片則匹配到一個相似度較高的圖。

![image-20231211005609661](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211005609661.png)

我們可以打開`data/matching_imgs`和`data/result`文件夾來對比結果的效果。

![image-20231211001850485](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211001850485.png)

![image-20231211001900621](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211001900621.png)

匹配到的圖像與原圖

在人臉的位置以及髮型上都較為相似，**符合歐氏距離應有的匹配效果**。

### CKKS

BFV的加密參數如下所示：

```python
global_scale = 2**40
```

依次執行以下指令：

```shell
python tenseal_encrypt.py -a CKKS -i data/storage_imgs -o data/enc_storage_imgs
python .\DBuploader.py -i .\data\enc_storage_imgs\
python tenseal_encrypt.py -a CKKS -i data/matching_imgs -o data/enc_matching_imgs -c .\keys\public_context
python .\tenseal_matching.py -a CKKS -i data/enc_matching_imgs -o data/result -c keys/private_context -db .\encrypted_images.db
```

統計得到**單張圖像的加密時間為0.024秒左右**。

![image-20231211013257836](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211013257836.png)

存儲大小同樣是2MB左右。

![image-20231211013436177](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211013436177.png)

測試查詢匹配。

![image-20231211015531385](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211015531385.png)

單張圖片的**匹配操作（即密態計算）時間為0.02秒，解密時間為0.008秒左右**。

![image-20231211111653823](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211111653823.png)

最後的識別結果和BFV一致。

### 對比不同參數

將收集的數據列為更為直觀的表格：

| 算法 | 單張圖像加密時間 | 單張加密圖像大小 | 單張圖像匹配時間 | 單張圖像解密時間 |
| ---- | ---------------- | ---------------- | ---------------- | ---------------- |
| BFV  | 0.019s           | 2MB              | 0.02s            | 0.007s           |
| CKKS | 0.024s           | 2MB              | 0.019s           | 0.008s           |

當參數設置相近時，BFV和CKKS演算法在圖像數據加密結果方面的效果相似，表明這兩種演算法在處理此類數據時具有一定的等效性。

而**在加密花費的時間上，BFV相對於CKKS速度更快**。

造成這種現象的原因，可能是CKKS是運用在浮點級數據上的演算法，對整數運算的支持程度並沒有BFV高。

我們將BFV的`plain_modulus`參數調大一些，設為786433。

而CKKS的`global_scale`參數設為2**60。

![image-20231211143026527](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231211143026527.png)

可以得到下表測試數據：

| 算法 | plain_modulus | global_scale | 單張圖像加密時間 |
| ---- | ------------- | ------------ | ---------------- |
| BFV  | 65537         |              | 0.019s           |
| BFV  | 786433        |              | 0.021s           |
| CKKS |               | 2**40        | 0.024s           |
| CKKS |               | 2**60        | 0.024s           |

可以發現，當密文空間增大、加密安全性增加時，BFV的加密時間增長幅度更大。

總的來說，BFV更適合需要精確整數計算的場景，而CKKS更適合於可以接受一定近似的浮點數計算。在加密的效率上，當數據量較大時，BFV比CKKS的效率更高，且CKKS對密文空間大小的敏感程度更高。

## 貢獻

歡迎對此專案進行貢獻！如果您有改善建議或遇到任何問題，請隨時打開一個問題或提交一個拉取請求。

## 許可證

該專案在GNU通用公共許可證v3.0下獲得許可。

詳情請見[LICENSE](../LICENSE)文件。

