# Environment Setup and Requirements Installation

Here's a step-by-step guide to set up the environment and install requirements for your assignments, ensuring each step is covered from environment setup to dependencies installation.

## Step 1: Set Up a Python Virtual Environment

A virtual environment helps isolate project-specific dependencies. Here's how to create one:

1. **Navigate to your project directory** or create a new directory for the assignments. clone this repository by the command

```
git clone https://github.com/yashbudhia/emmetra.git
```

2. **Create the virtual environment** using Python (replace `env` with your preferred name for the environment):

   ```bash
   python -m venv env
   ```

3. **Activate the virtual environment**:

   - **On Windows**:

     ```bash
     .\env\Scripts\activate
     ```

   - **On macOS/Linux**:

     ```bash
     source env/bin/activate
     ```

## Step 2: Install CUDA-Compatible PyTorch

Since your assignments require PyTorch with CUDA support, install PyTorch for CUDA 11.8 specifically (replace `11.8` if you have a different CUDA version installed).

- . Use the following command to install PyTorch for CUDA 11.8:

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## Step 3: Install Additional Requirements

Your assignments require additional packages beyond PyTorch, such as OpenCV, NumPy, ReportLab, and potentially TensorFlow for model-based denoising.

- . **Install the dependencies from the `requirements.txt` file**:

  ```bash
  pip install -r requirements.txt
  ```

This should display the installed packages, confirming that PyTorch, OpenCV, ReportLab, NumPy, and TensorFlow are available.

## Summary of Commands

Here's a consolidated list for quick reference:

```bash
# 1. Set up and activate the virtual environment
python -m venv env
source env/bin/activate # On macOS/Linux
.\env\Scripts\activate # On Windows

# 2. Install PyTorch for CUDA 11.8 (or install specific version of pytorch for your version of cuda)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install other requirements
pip install -r requirements.txt

# 4. Verify installed packages
pip list
```

### Accessing Projects

## 1st Assignment

1. Go inside First Assignment Folder

```
cd 1st-assignment

```

2. Run Flask server by typing the command in the terminal

```
python app.py
```

3. The UI for Custom Image Signal Processing pipeline will appear on the web browser in the url [localhost:5000](http://127.0.0.1:5000)

4. If you want to see OpenCV implementation - Go inside the folder

```
cd openCV-version
```

- And run flask server by typing the command in the terminal

```
python app.py
```

5. If you want to see Custom ISP code without any UI . Just Naviagate to Custom-ISP folder and run python ISP.py in the terminal

```
cd ..
cd Custom-ISP
```

- Feel free to tweak the parameters

# Emmetra: 3rd Assignment
Overview
Briefly explain the purpose of the assignment, its main functionality, and any core concepts or algorithms it implements.

Environment Setup and Requirements Installation
## Step 1: Clone the Repository
```bash 
git clone https://github.com/yashbudhia/emmetra.git
cd emmetra/3rd-assignment
```
## Step 2: Set Up a Virtual Environment
Creating a virtual environment isolates project dependencies.

```bash
python -m venv env
```
Activate the environment:
Windows: .\env\Scripts\activate
macOS/Linux: source env/bin/activate

## Step 3: Install Requirements
Install all required packages:

```bash
pip install -r requirements.txt
```
## Step 4: Running the Assignment
Include any instructions for running scripts, such as:

Navigate to the assignment folder:
```bash
cd 3rd-assignment
```
Run the main script:
```bash
python app.py
```
