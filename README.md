# Multi-aspect Knowledge Distillation with Large Language-Model, CVPRW 2025

**Our paper has been accepted to the CVPR 2025 Workshop (FGVC12) and will appear in the official CVPR proceedings (8 pages).**

This repository is the official implementation of **Multi-aspect Knowledge Distillation with Large Language-Model**.
<br/>
**Taegyeong Lee*** ,
**Jinsik Bang***,
Soyeong Kwon, 
Taehwan Kim
<br/>

https://www.arxiv.org/abs/2501.13341

## Abstract
Recent advancements in deep learning have significantly improved performance on computer vision tasks. Previous image classification methods primarily modify model architectures or add features, and they optimize models using cross-entropy loss on class logits. Since they focus on classifying images with considering class labels, these methods may struggle to learn various \emph{aspects} of classes (e.g., natural positions and shape changes). In contrast, humans classify images by naturally referring to multi-aspects such as context, shape, color, and other features. Inspired by this, rethinking the previous approach from a novel view, we propose a multi-aspect knowledge distillation method using Multimodal Large Language Models (MLLMs). Our approach involves: 1) querying Large Language Model with multi-aspect questions relevant to the knowledge we want to transfer to the model, 2) extracting corresponding logits from MLLM, and 3) expanding the model's output dimensions to distill these multi-aspect logits. We then apply cross-entropy loss to class logits and binary cross-entropy loss to multi-aspect logits. Through our method, the model can learn not only the knowledge about visual aspects but also the abstract and complex aspects that require a deeper understanding. We primarily apply our method to image classification, and to explore the potential for extending our model, we expand it to other tasks, such as object detection. In all experimental results, our method improves the performance of the baselines. Additionally, we analyze the effect of multi-aspect knowledge distillation. These results demonstrate that our method can transfer knowledge about various aspects to the model and the aspect knowledge can enhance model performance in computer vision tasks. This paper demonstrates the great potential of multi-aspect knowledge distillation, and we believe it offers a promising direction for future research in computer vision and beyond.

## News
- [2025/01/13] We released multi-aspect question.txt
- [2025/01/14] We released Logit extraction code and MaKD Training code.
- [2025/01/27] We released Datasets and Logits files (Caltech101, OxfordPets, StanfordPets, Flowers, DTD)
- [2025/03/31] **Our paper has been accepted to the CVPR 2025 Workshop (FGVC12) and will appear in the official CVPR proceedings (8 pages).**

## Approach
### **1. Multi-aspect question generation and logit extraction**
<img src="figures/approach02.png" height="400" alt="Image description">


### **2. Mutli-aspect logit distillation**


<img src="figures/approach01.png" height="300" alt="Image description">

## Usage
### 0. Preparation stage
- Install InternVL 2.5(MLLM) for extracting logits on multi-aspect questions (<a href='https://internvl.readthedocs.io/en/latest/get_started/installation.html'>install document</a>).**
- To obtain Yes/No logits for multi-aspect questions using MLLM, you need to install InternVL2.5.
- You can apply internvl_logits/simple_make_makd_logits_json.py ! (you can select simple version or previous version)
- Previous version : Then, replace the temporary files ``modeling_internvl_chat.py``, ``transformers\tokenization_utils_base.py``, and ``transformers\generation\utils.py`` with the our github files from ``InternVL_logits folder``.
- Download the fine-grained datasets (Caltech101, OxfordPets, CUB-2011, Flowers, StanfordCars, DTD, Mini-ImageNet, FGVC Craft) from <a href="https://drive.google.com/file/d/1cdg5uW526R_Ut38aVc5dpSOTbqrZA1gM/view?usp=sharing">here</a>.
- Download the multi-aspect questions we created using GPT-4 from <a href="https://drive.google.com/file/d/1yIw5XNWOXN2lt_1l4OFhnWvIFouZBXDv/view?usp=sharing">here</a>.
- Download the multi-aspect logits extracted using InternVL-2.5 8B from <a href="https://drive.google.com/file/d/1A5sKlqi3DrFAlFS1O9n8GwcVRJCjO_lu/view?usp=sharing">here</a>.
- You can easily find the installation path of the Transformer and the temporary file path of Hugging Face using the code below.

### **1. Create multi-aspect questions suitable for the dataset using ChatGPT.**
We create a total of $N$ multi-aspect questions based on the class labels of the dataset using LLM.
Then, considering visual, categorical, and environmental aspects, we filter and select $Q$ multi-aspect questions using the LLM.
$Q$ is the number of multi-aspect questions we want to transfer to our model.
 ```
Instruction : The dataset consists of $C$ classes and $M$ images.
The class list is as follows: $[CLASS]$, Generate $N$ feature-specific yes or no questions,
focusing on clear and distinct aspects of the objects in the images in the dataset.
```

```
Instruction : Select $Q$ of the most relevant and distinct questions from the list,
focusing on various key features that distinguish different class in the dataset
```

 
 These generated aspect questions represent the knowledge we aim to transfer to the models based on datasets. 
**You can download the multi-aspect questions we generated <a href="https://drive.google.com/drive/folders/1-c5K4kTUbLiyH8oPYTHSmC7mW7k8A0AW?usp=sharing">here</a>.**


### 2. Logit extraction for multi-aspect questions.

We generate questions about aspects to be transferred to the model from the LLM. 
Using an MLLM, we input the dataset and the generated multi-aspect questions, prompting it to answer yes or no. 
We then extract the logits corresponding to yes and no tokens, and apply the softmax function to both the yes and no logits. 
We use the softmax results of the yes logits as the targets and generate multi-aspect-logits.json.

**version :**
You need to change ``path``, ``multi_aspect_questions_path``, ``image_folder_path``, ``output_json_path``.

```
python InternVL_logits/make_makd_logits_json.py
```

### 3. Training neural networks with MaKD on fine-grained datasets.

You need to modify ``configs/fine_grained/makd.yaml`` according to the fine-grained dataset you want to train on. 
Then, run the following command.

```
python tools/our_train.py configs/fine_grained/makd.yaml
```

### 4. Experiment results.

<img src="figures/result.png" height="600" alt="Image description">


## Acknowledgement
Sincere gratitude to the contributors of mdistiller and InternVL for your distinguished efforts.
