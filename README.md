# Classify Apple Pytorch 1.7
This project for classify green and red apples by using Pytorch 1.7 and 
test model and print confusion matrix parameters.

# Data Set
Fruits 360 dataset: A dataset of images containing fruits and vegetables
https://www.kaggle.com/datasets/moltean/fruits

Only used green and red apple data set in this project.That means we have two clases.
Sample of images:

   <img width="200" src=images/0_100.jpg></a>
   <img width="200" src=images/107_100.jpg></a>
   

# Clone 
We begin by cloning the classify_apple repository and setting up the dependencies required to run. You might need sudo rights to install some of the packages.

In a terminal, type:

```bash
git clone https://github.com/kadircosar/classify_apple.git
```
# İnstall requirements
I recommend you create a new conda or a virtualenv environment to run your classify_apple as to not mess up dependencies of any existing project. Once you have activated the new environment, install the dependencies using pip.

Before running this code in terminal make sure activate your venv that you created for this project and run this code in path that you cloned classify_apple.

```bash
pip install -r requirements.txt

```
# Run Model

Now you can run model.
Activate venv and run this code in path that you cloned classify_apple.

Run with terminal:
```bash
python3 model.py
```

# Train Results 
   <img width="500" src=images/train.png></a>
 
 
# Test Results, Confusion Matrix

   <img width="500" src=images/test_confusion.png></a>



