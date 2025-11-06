import os
import json
from main import main

# --- Step 1: Load complaints ---
def load_complaints(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    return texts


# --- Step 2: Ensure results folder exists ---
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# --- Step 3: Run full pipeline ---
if __name__ == "__main__":
    input_path = "data/complaints.jsonl"
    output_path = "results/complaints_output.json"

    print("🔹 Loading complaints...")
    complaints = load_complaints(input_path)
    print(f"Loaded {len(complaints)} complaints.\n")

    ensure_dir(output_path)

    all_results = []

    print("🚀 Running full AI complaint analysis pipeline...\n")

    for i, complaint_text in enumerate(complaints):
        print(f"⚙️ Processing complaint {i+1}/{len(complaints)}: {complaint_text[:80]}...")
        try:
            result = main(complaint_text)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error processing complaint {i+1}: {e}")
            continue

    # --- Step 4: Save output ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All {len(all_results)} complaints processed successfully!")
    print(f"📁 Results saved to: {output_path}")
