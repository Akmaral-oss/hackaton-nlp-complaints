
# 🚍 AI Complaint Analysis System for City Transport


## 🧩 Project Overview
An **AI-powered pipeline** that transforms unstructured citizen complaints into actionable insights for city transport management.  

**Key Goals:**
- Extract relevant information from free-text complaints  
- Classify complaint aspects (punctuality, safety, etc.)  
- Determine complaint priority (low → critical)  
- Provide actionable recommendations for city managers  
- Visualize trends to improve decision-making

---

## 🏗️ Features

| Feature | Description | Example |
|---------|------------|---------|
| **Text Normalization** | Cleans text, fixes spelling & grammar | `"№12 автобус опоздал"` → `"автобус номер 12 опоздал"` |
| **NER** | Extracts routes, locations, times, objects | `"№25 автобус сынып қалды"` → `route: №25, object: автобус` |
| **Aspect Classification** | Categorizes complaints: punctuality, safety, bus condition, etc. | `"автобус кешігіп келді"` → `"пунктуальность"` |
| **Priority Classification** | Determines severity: low, medium, high, critical | `"опасная ситуация"` → `"критический"` |
| **LLM Recommendation** | Generates actionable steps for city management | `"№25 автобустарды техникалық тексеру қажет"` |
| **Visualization** | Graphs: routes, levels, aspects | ![Example](docs/example_chart.png) |

---

## 🛠️ Tech Stack
- **Python 3.12+**
- [Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/)
- **SymSpell** (spell correction)
- **Langdetect** (language detection)
- **Scikit-learn** (KMeans clustering)
- **Pandas & Matplotlib** (data visualization)
- **Streamlit** (interactive dashboard, optional)

---

## ⚙️ Installation
1. Clone the repo:
```bash
git clone <repo_url>
cd project
Install dependencies:

bash
Копировать код
pip install -r requirements.txt
(Optional) Download dictionary for SymSpell:

python
Копировать код
# sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)
🏃 Usage
python
Копировать код
from main import main

complaint_text = "№25 автобус кешігіп келді, автобустар сынып қалды."
result = main(complaint_text)

print(result)
Example Output:

json
Копировать код
{
  "text": "автобус номер 25 кешігіп келді",
  "route": "№25",
  "place": "Автобус станциясы",
  "object": "Автобус",
  "time": "08:30",
  "aspect": "состояние автобуса",
  "priority": "высокий",
  "recommendation": "№25 автобустарды техникалық тексеруді күшейту және қосымша көлік жіберу қажет."
}
📊 Visualization
After collecting multiple complaints:

python
Копировать код
from visualization.visualizer import visualize_complaints
visualize_complaints(complaint_data)
Generates:

Top problematic routes

Complaint levels distribution

Aspect frequency distribution

<!-- Place screenshot or GIF -->

🧠 Future Work
Train a domain-specific BERT model for transport complaints

Improve entity extraction accuracy

Build a real-time Streamlit dashboard

Expand support for multiple languages
