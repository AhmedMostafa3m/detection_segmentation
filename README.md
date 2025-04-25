In this document I’ll guide you step-by-step through creating a **Django web application** with two interfaces: one for **object detection** (using a pre-trained DETR model) and another for **image segmentation** (using a pre-trained Mask R-CNN model from `torchvision`). The app will allow users to upload images, process them, and view results with bounding boxes (for detection) or masks (for segmentation).

The steps are detailed, but I’ll break them into manageable parts with clear commands and code. 

---

### **Overview of the Django Web App**
- **Functionality**:
  - **Object Detection Window**: Users upload an image, and the app uses DETR (`facebook/detr-resnet-50`) to detect objects, draw bounding boxes, and display results.
  - **Segmentation Window**: Users upload an image, and the app uses Mask R-CNN (`maskrcnn_resnet50_fpn`) to segment objects, draw masks, and display results.
  - Both interfaces will be accessible via separate URLs (e.g., `/detection` and `/segmentation`).
- **Tech Stack**:
  - **Django**: For the web framework.
  - **OpenCV, Pillow, NumPy**: For image processing.
  - **Hugging Face Transformers**: For DETR (object detection).
  - **PyTorch/Torchvision**: For Mask R-CNN (segmentation).

---

### **Steps to Create and Deploy the Django Web App**

#### **Step 1: Set Up Your Environment**
1. **Install Python**:
   - Ensure **Python 3.8 or higher** is installed. Download from [python.org](https://www.python.org/downloads/) if needed.
   - Verify in Command Prompt:
     ```bash
     python --version
     ```

2. **Install Git**:
   - Install **Git for Windows** from [git-scm.com](https://git-scm.com/download/win) 
   - Verify:
     ```bash
     git --version
     ```

3. **Create a Project Directory**:
   - Open Command Prompt and create a directory for the project:
     ```bash
     mkdir django_object_segmentation
     cd django_object_segmentation
     ```

4. **Set Up a Virtual Environment**:
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
     You should see `(venv)` in your prompt.

5. **Install Dependencies**:
   - Install Django, Gunicorn (for Heroku), and other required packages:
     ```bash
     pip install django gunicorn opencv-python pillow numpy torch torchvision transformers
     ```
   - Generate a `requirements.txt` file:
     ```bash
     pip freeze > requirements.txt
     ```

---

#### **Step 2: Create the Django Project and App**
1. **Create a Django Project**:
   - Run:
     ```bash
     django-admin startproject object_segmentation
     ```
   - This creates a directory structure:
     ```
     django_object_segmentation/
     ├── object_segmentation/
     │   ├── __init__.py
     │   ├── asgi.py
     │   ├── settings.py
     │   ├── urls.py
     │   └── wsgi.py
     ├── manage.py
     ├── requirements.txt
     └── venv/
     ```

2. **Create a Django App**:
   - Navigate to the project directory:
     ```bash
     cd object_segmentation
     ```
   - Create an app named `detector`:
     ```bash
     python manage.py startapp detector
     ```
   - Add the `detector` app to `INSTALLED_APPS` in `object_segmentation/settings.py`:
     ```python
     INSTALLED_APPS = [
         ...
         'detector.apps.DetectorConfig',
     ]
     ```

3. **Set Up the Project Structure**:
   - Create folders for templates, static files, and uploads:
     ```bash
     mkdir detector\templates detector\static detector\static\uploads detector\static\css
     ```
   - Your structure should now look like:
     ```
     django_object_segmentation/
     ├── object_segmentation/
     │   ├── __init__.py
     │   ├── asgi.py
     │   ├── settings.py
     │   ├── urls.py
     │   └── wsgi.py
     ├── detector/
     │   ├── __init__.py
     │   ├── admin.py
     │   ├── apps.py
     │   ├── migrations/
     │   ├── models.py
     │   ├── static/
     │   │   ├── css/
     │   │   │   └── style.css
     │   │   └── uploads/
     │   ├── templates/
     │   ├── tests.py
     │   └── views.py
     ├── manage.py
     ├── requirements.txt
     └── venv/
     ```

---

#### **Step 3: Write the Django App Code**
Below are the code files for the Django app, including views for object detection and segmentation, templates for the UI, and static CSS for styling.

1. **Update `settings.py`**:
   - Configure static files, media uploads, and allowed hosts for Heroku.
   - Edit `object_segmentation/settings.py`:

![object_segmentation/settings.py](https://github.com/AhmedMostafa3m/detection_segmentation/blob/326d12e3528fff1ed105ed63ce3c812013fe674c/object_segmentation/object_segmentation/settings.py)

2. **Define URLs**:
   - Edit `object_segmentation/urls.py` to route requests to the `detector` app:

![object_segmentation/urls.py](https://github.com/AhmedMostafa3m/detection_segmentation/blob/4ee65fc285ca365b7cc6c8b9162257fc1cea2739/object_segmentation/object_segmentation/urls.py)

3. **Create App URLs**:
   - Create `detector/urls.py` to define routes for detection and segmentation:
![detector/urls.py](https://github.com/AhmedMostafa3m/detection_segmentation/blob/251819fa5633ddd05309947734e303d1ee8b124c/object_segmentation/detector/urls.py)

4. **Implement Views**:
   - Edit `detector/views.py` to handle image uploads, process images with DETR and Mask R-CNN, and render results:
![detector/views.py](https://github.com/AhmedMostafa3m/detection_segmentation/blob/251819fa5633ddd05309947734e303d1ee8b124c/object_segmentation/detector/views.py)

5. **Create Templates**:
   - Create the following templates in `detector/templates/`:
     - `index.html`: Homepage with links to detection and segmentation.
     - `detection.html`: Form for uploading images for detection.
     - `detection_result.html`: Display detection results.
     - `segmentation.html`: Form for uploading images for segmentation.
     - `segmentation_result.html`: Display segmentation results.

6. **Add CSS**:
   - Create `detector/static/css/style.css` for styling:
![detector/static/css/style.css](https://github.com/AhmedMostafa3m/detection_segmentation/blob/251819fa5633ddd05309947734e303d1ee8b124c/object_segmentation/detector/static/css/style.css)
---

#### **Step 4: Test the App Locally**
1. **Collect Static Files**:
   - Run:
     ```bash
     python manage.py collectstatic
     ```
   - This copies static files to `staticfiles/` for production.

2. **Run the Development Server**:
   - Ensure you’re in the `object_segmentation` directory:
     ```bash
     cd D:\path_to\django_object_segmentation\object_segmentation
     ```
   - Start the server:
     ```bash
     python manage.py runserver
     ```
### **Explanation of the Code**
- **settings.py**: Configures Django for static files, media uploads, and Heroku deployment.
- **urls.py**: Routes requests to the `detector` app for detection and segmentation.
- **views.py**:
  - `index`: Renders the homepage with links.
  - `detection`: Handles image uploads, processes with DETR, draws bounding boxes, and displays results.
  - `segmentation`: Handles image uploads, processes with Mask R-CNN, draws masks, and displays results.
- **Templates**: Provide a user-friendly UI with forms and result displays.
- **CSS**: Styles the app for a clean, responsive look.

3. **Test the App**:
   - Open `http://127.0.0.1:8000` in a browser.
   - Verify:
     - The homepage shows links to “Object Detection” and “Image Segmentation”.
     - Upload an image in the detection page (`/detection`) to see bounding boxes.
     - Upload an image in the segmentation page (`/segmentation`) to see masks.
   - If you encounter errors (e.g., template not found), double-check file placement in `detector/templates/` and `detector/static/`.

### **the home page look like that :**

![tape image](https://github.com/AhmedMostafa3m/django_Detection_segmentation/blob/91ff02163733d64e19e0386f1e495654bdc51f0e/Images/Screenshot%202025-04-25%20113840.png)

### **the detction page look like that :**

![tape image](https://github.com/AhmedMostafa3m/django_Detection_segmentation/blob/91ff02163733d64e19e0386f1e495654bdc51f0e/Images/Screenshot%202025-04-25%20113947.png)

### **the original Image :**

![tape image](https://github.com/AhmedMostafa3m/django_Detection_segmentation/blob/91ff02163733d64e19e0386f1e495654bdc51f0e/Images/dog.jpeg)

### **the processed Image :**

![tape image](https://github.com/AhmedMostafa3m/django_Detection_segmentation/blob/02f80cd0d836ad22e4ae0194711a03f2eadc6da2/Images/processed_dog.jpeg)

### **the segmentation page look like that :**

![tape image](https://github.com/AhmedMostafa3m/django_Detection_segmentation/blob/91ff02163733d64e19e0386f1e495654bdc51f0e/Images/Screenshot%202025-04-25%20114043.png)

### **the original Image :**

![tape image](https://github.com/AhmedMostafa3m/django_Detection_segmentation/blob/a877c142637e306102d85425a28247dfb2c0175e/Images/dog.jpeg)

### **the processed Image :**

![tape image](https://github.com/AhmedMostafa3m/django_Detection_segmentation/blob/a877c142637e306102d85425a28247dfb2c0175e/Images/processed_dog%20(2).png)

