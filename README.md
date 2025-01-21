# Kokoro-TTS

**Note:** This is not the official repository. Alternatives [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI), [kokoro](https://github.com/hexgrad/kokoro), [kokoro-web](https://huggingface.co/spaces/webml-community/kokoro-web), [Kokoro-Custom-Voice](https://huggingface.co/spaces/ysharma/Make_Custom_Voices_With_KokoroTTS)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/Kokoro-82M-WebUI/blob/main/Kokoro_82M_Colab.ipynb) <br>
[![HuggingFace Space Demo](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/NeuralFalcon/Kokoro-TTS)


---

### Installation Tutorial

My Python Version is 3.10.9.

#### 1. Clone the GitHub Repository:
```bash
git clone https://github.com/NeuralFalconYT/Kokoro-82M-WebUI.git
cd Kokoro-82M-WebUI
```

#### 2. Create a Python Virtual Environment:
```bash
python -m venv myenv
```
This command creates a new Python virtual environment named `myenv` for isolating dependencies.

#### 3. Activate the Virtual Environment:
- **For Windows:**
  ```bash
  myenv\Scripts\activate
  ```
- **For Linux:**
  ```bash
  source myenv/bin/activate
  ```
This activates the virtual environment, enabling you to install and run dependencies in an isolated environment.
Here’s the corrected version of point 4, with proper indentation for the subpoints:


#### 4. Install PyTorch:

- **For GPU (CUDA-enabled installation):**
  - Check CUDA Version (for GPU setup):
    ```bash
    nvcc --version
    ```
    Find your CUDA version example ```11.8```

  - Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and install the version compatible with your CUDA setup.:<br>
    - For CUDA 11.8:
    ```
    pip install torch  --index-url https://download.pytorch.org/whl/cu118
    ```
    - For CUDA 12.1:
    ```
    pip install torch  --index-url https://download.pytorch.org/whl/cu121
    ```
    - For CUDA 12.4:
    ```
    pip install torch  --index-url https://download.pytorch.org/whl/cu124
    ```
- **For CPU (if not using GPU):**
  ```bash
  pip install torch
  ```
  This installs the CPU-only version of PyTorch.


#### 5. Install Required Dependencies:
```bash
pip install -r requirements.txt
```
This installs all the required Python libraries listed in the `requirements.txt` file.

#### 6. Download Model and Get Latest VoicePack:
```bash
python download_model.py
```

---

#### 7. Install eSpeak NG

- **For Windows:**
  1. Download the latest eSpeak NG release from the [eSpeak NG GitHub Releases](https://github.com/espeak-ng/espeak-ng/releases/tag/1.51).
  2. Locate and download the file named **`espeak-ng-X64.msi`**.
  3. Run the installer and follow the installation steps. Ensure that you install eSpeak NG in the default directory:
     ```
     C:\Program Files\eSpeak NG
     ```
     > **Note:** This default path is required for the application to locate eSpeak NG properly.

- **For Linux:**
  1. Open your terminal.
  2. Install eSpeak NG using the following command:
     ```bash
     sudo apt-get -qq -y install espeak-ng > /dev/null 2>&1
     ```
     > **Note:** This command suppresses unnecessary output for a cleaner installation process.

---

#### 8. Run Gradio App

To run the Gradio app, follow these steps:

1. **Activate the Virtual Environment:**
   ```bash
   myenv\Scripts\activate
   ```

2. **Run the Application:**
   ```bash
   python app.py
   ```

   Alternatively, on Windows, double-click on `run_app.bat` to start the application.

---

![app](https://github.com/user-attachments/assets/123e9465-3713-4056-8051-0f526a608645)
![multispeech](https://github.com/user-attachments/assets/a8c703c3-7950-4966-938c-7f32ae3cfbad)
![srt](https://github.com/user-attachments/assets/2009cdc1-dfb1-446c-a5ae-e67cace01dbe)
![voices](https://github.com/user-attachments/assets/9e9c9386-d163-4c6a-b5c8-4d325db72ddc)

### Credits
[Kokoro HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)

