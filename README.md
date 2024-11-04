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

# 1st Assignment

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

# 2nd Assignment

For Running 2nd assignment - follow the steps

## Step 1 : Navigate to 2nd-assignment folder if the environment is setup

```bash
cd emmetra/2nd-assignment
```

## Step 2 : First go to ffdnet model folder and generate a denoised image

```
cd ffdnet-pytorch

```

- after reaching here run the command in terminal to denoise sample image obtained from the ISP from assignment 1(image.png)

```
python test_ffdnet_ipol.py --input image.png --noise_sigma 25 --add_noise True

```

- After the model has generated the image now go back

```
cd ..
```

- Now run

```
python main.py
```

- It will then generate reports of the image in various denoising and edge effects.

- You can configure the input by replacing image.png in inputs folder with your image (rename it to image.png) and in ffdnet folder too

- Remember - Run the FFDnetModel before running the main.py script.

### Custom model usage -

- Custom model was still in development(unfortunately) so would recommend just leave it as is. anyway run

```
python main_processing.py
```

to generate reports using custom model

# 3rd Assignment

Overview
Briefly explain the purpose of the assignment, its main functionality, and any core concepts or algorithms it implements.

Environment Setup and Requirements Installation

## Step 1: Navigate to 3rd-assignment folder if the environment is setup

```bash
cd emmetra/3rd-assignment
```

## Step 2: Running the Assignment

Include any instructions for running scripts, such as:

Run the main script:

```bash
python app.py
```

- The UI Will then appear on the Web Browser at [localhost:5000](http://127.0.0.1:5000)

## Step 3: Upload the images

- Upload the images at given exposures and obtain the HDR image. Sample inputs are present at inputs folder at 3rd-assignment folder.
