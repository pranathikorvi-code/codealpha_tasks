# Astral-Logic Tasks

Welcome to my official CodeAlpha project repository. This collection contains three independent production-grade solutions demonstrating advancements across **Computer Vision (CV)**, **Natural Language Processing (NLP)**, and **Web-API Integrations**.

---

## 📁 Repository Directory Index

```text
codealpha_tasks/
├── FAQ CHATBOT/
│   └── chatbot.py               # NLP Engine & Gradio Interface
├── LANGUAGE TRANSLATOR TOOL/
│   └── TRANSLATOR.HTML          # Web Translator App via REST API
└── object_tracking_project/
    ├── tracker.py               # YOLOv8 Tracker Script
    └── sample.mp4               # Raw Input Video Asset

```

---

## 🛠️ Complete Project Overviews & Testing Guide

### 🤖 1. Enterprise FAQ AI Chatbot

* **Directory Location:** `FAQ CHATBOT/`
* **🚀 Live Web Deployment:** https://huggingface.co/spaces/kpranathi/my-faq-chatbot

**Description:** Converts user text queries into numerical weights using a TF-IDF Matrix Vectorizer and evaluates intent similarities using Cosine Similarity matching vectors. Whenever a user query passes an optimized threshold of 0.25, the application instantly pulls the corresponding solution from a structured Pandas DataFrame and displays it inside a live web interface powered by Gradio and hosted on Hugging Face Spaces.

**🧪 Verification / How to Test:**

1. Open the Live Web App Link directly in your browser: https://huggingface.co/spaces/kpranathi/my-faq-chatbot
2. Type: `"How to return an item?"`
3. **Expected Output Answer:** 🤖 `"We offer a 30-day return policy on all unused items in their original packaging."`

---

### 🎥 2. Real-Time Object Detection, Tracking & Counting

* **Directory Location:** `object_tracking_project/`

**Description:** Ingests live frame arrays from a local source video stream asset, passes matrices frame-by-frame through a pre-trained convolutional YOLOv8 deep learning architecture, tracks detected features using persistent boundary trackers across bounding boxes, and overlays a graphical counting statistic system built with OpenCV.

**🧪 Verification / How to Test:**

1. Open your terminal inside the main repository folder and run:

```bash
cd object_tracking_project
call venv\Scripts\activate
python tracker.py

```

2. **Expected System Outcome:** A display window opens playing the video sequence. Floating bounding frames track objects natively with green numerical trackers appearing, while a text banner displays a changing calculation string reading: `Objects in Frame: X`.
3. **Exit Protocol:** Press `q` to exit the graphical window display.

---

### 🌐 3. Universal Language Translation Tool

* **Directory Location:** `LANGUAGE TRANSLATOR TOOL/`

**Description:** A lightweight, high-performance web dashboard application that leverages front-end JavaScript runtime hooks to execute asynchronous API endpoint queries (`async/await`). It queries translation strings natively over a main REST gateway (Lingva API) and utilizes an automated fail-safe network switch that hooks into a secondary fallback database structure (MyMemory API) if network connections experience latency.

**🧪 Verification / How to Test:**

1. Locate `TRANSLATOR.HTML` on your system and double-click it to execute the page inside any browser.
2. Type your source statement inside the left-side text area: `"Hello, welcome to my software evaluation."`
3. Set your target languages (e.g., From: English, To: Spanish) and click **"Translate Now"**.
4. **Expected System Outcome:** The interface queries the server infrastructure and outputs a translated response string on screen reading: `"Hola, bienvenido a mi evaluación de software."`
