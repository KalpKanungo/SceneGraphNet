---
title: Scene Graph Generator
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# 🧠 Scene Graph Generator (Multimodal AI System)

A multimodal computer vision system that takes an input image, detects objects, predicts relationships between them, constructs a structured scene graph, and generates a natural language description of the scene.

🔗 **Live Demo (Hugging Face Spaces):**  
https://huggingface.co/spaces/kalpkanungo/SceneGraphNet

---

# 🚀 Features

- 🖼️ Object Detection using DETR (ResNet-50)
- 🔗 Relationship Prediction (Custom Trained Model)
- 📐 Spatial Reasoning (Hybrid AI with Geometry Rules)
- 🧩 Scene Graph Construction (Directed Graph)
- 📊 Graph Visualization (NetworkX + Matplotlib)
- 🧠 Graph-to-Text Generation (FLAN-T5)
- 🌐 Interactive UI (Gradio)
- ☁️ Deployed on Hugging Face Spaces (CPU)

---

# 🧠 How It Works (End-to-End Pipeline)

### 1. Input
- User uploads an image (JPG/PNG) via Gradio UI
- Image is converted from PIL → OpenCV format

---

### 2. Object Detection
- Uses `facebook/detr-resnet-50` from Hugging Face
- Outputs:
  - Object labels (COCO classes)
  - Bounding boxes
  - Confidence scores
- Applies threshold (≥ 0.7) to filter noise

---

### 3. Pairwise Object Processing
- Generates object pairs using `itertools.combinations`
- Extracts bounding boxes for each pair
- Creates union region for relation inference
- Filters duplicate object pairs

---

### 4. Relationship Prediction
- Custom-trained classifier on Visual Genome subset (~10K samples)
- Predicts semantic relations:
  - `on`, `holding`, `behind`, etc.
- Trained using PyTorch (10 epochs)

---

### 5. Spatial Reasoning (Hybrid AI)
- Uses bounding box geometry to compute:
  - `left_of`, `right_of`, `above`, `below`, `near`
- Hybrid logic:
  - Semantic relations from model (if confident)
  - Otherwise fallback to spatial rules
- Reduces bias (e.g., “everything = on”)

---

### 6. Graph Construction
- Builds a **directed graph (NetworkX DiGraph)**
  - Nodes → objects
  - Edges → relationships
- Removes duplicates and limits edges for clarity

---

### 7. Graph Visualization
- Uses NetworkX + Matplotlib
- Displays:
  - Directed edges with labels
  - Clean layout for readability

---

### 8. Graph → Text (NLP)
- Uses `google/flan-t5-small`
- Converts structured triples into natural language

Example:
laptop → on → table
mouse → next_to → laptop

Output:
"A laptop is placed on a table with a mouse next to it."

---

### 9. UI (Gradio)
- Upload image
- View:
  - Scene graph
  - Generated description
- Fully interactive and browser-based

---

# 🏗️ Tech Stack
```

- **Computer Vision:** DETR (Hugging Face Transformers)
- **Deep Learning:** PyTorch
- **Graph Processing:** NetworkX
- **NLP:** FLAN-T5
- **Image Processing:** OpenCV
- **Frontend/UI:** Gradio
- **Deployment:** Hugging Face Spaces

```

---

# 📁 Project Structure
scene-graph-generator/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│ ├── pipeline.py
│ ├── detection.py
│ ├── spatial_rules.py
│ ├── relationship_infer.py
│ ├── scene_graph.py
│ ├── visualization.py
│ ├── text_generation.py

---

# ⚙️ Installation (Local Setup)

```bash
git clone https://github.com/<your-username>/scene-graph-generator.git
cd scene-graph-generator

pip install -r requirements.txt
python app.py
