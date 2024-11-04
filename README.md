# Environment Setup and Requirements Installation

Here's a step-by-step guide to set up the environment and install requirements for your assignments, ensuring each step is covered from environment setup to dependencies installation.

## Step 1: Set Up a Python Virtual Environment

A virtual environment helps isolate project-specific dependencies. Here's how to create one:

1. **Navigate to your project directory** or create a new directory for the assignments.
2. **Create the virtual environment** using Python (replace `env` with your preferred name for the environment):

   ```bash
   python -m venv env
   ```

````

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

1. Use the following command to install PyTorch for CUDA 11.8:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Step 3: Install Additional Requirements

Your assignments require additional packages beyond PyTorch, such as OpenCV, NumPy, ReportLab, and potentially TensorFlow for model-based denoising.

1. **Create a `requirements.txt` file in your project directory with the following content**:

   ```
   numpy opencv-python reportlab torch torchvision torchaudio # Additional packages like TensorFlow, if required tensorflow
   ```

2. **Install the dependencies from the `requirements.txt` file**:

   ```bash
   pip install -r requirements.txt
   ```

3. If `requirements.txt` isn't available, generate one based on your current environment setup:

   ```bash
   pip freeze > requirements.txt
   ```

## Step 4: Verifying Installation

After installation, verify all required packages are installed correctly.

```bash
pip list
```

This should display the installed packages, confirming that PyTorch, OpenCV, ReportLab, NumPy, and TensorFlow are available.

## Summary of Commands

Here's a consolidated list for quick reference:

```bash
# 1. Set up and activate the virtual environment
python -m venv env
source env/bin/activate # On macOS/Linux
.\env\Scripts\activate # On Windows

# 2. Install PyTorch for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install other requirements
pip install -r requirements.txt

# 4. Verify installed packages
pip list
```

```

```
````
