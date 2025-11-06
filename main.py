# main.py
from preprocessing.preprosing import normalize_complaint
from nlp.aspect_classifier import classify_aspect_unsupervised
from nlp.ner import extract_entities_with_ner
from nlp.priority_classifier import classify_complaint_level
from nlp.llm_utils import generate_solution_with_llm
from visualization.data_preparer import collect_complaints, prepare_complaint_data


complaint_data = []

def main(complaint_text):
    normalized_text = normalize_complaint(complaint_text)
    ner_results = extract_entities_with_ner(normalized_text)
    priority = classify_complaint_level(complaint_text)
    aspect = classify_aspect_unsupervised([complaint_text])[0]

    solution = generate_solution_with_llm(
        complaint_text,
        ner_results['routes'][0][0] if ner_results["routes"] else "N/A",
        ner_results['times'][0][0] if ner_results["times"] else "N/A",
        ner_results['places'][0][0] if ner_results["places"] else "N/A",
        aspect,
        priority
    )

    complaint_output = prepare_complaint_data(
        complaint_text, ner_results, aspect, priority, solution
    )
    collect_complaints(complaint_data, complaint_output)
    return complaint_output
