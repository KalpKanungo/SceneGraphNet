from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


def graph_to_text(relations):
    if not relations:
        return "No relationships detected."

    # convert relations to prompt
    triples = [
        f"{r['subject']} {r['relation']} {r['object']}"
        for r in relations
    ]

    prompt = "Convert the following relationships into a natural sentence:\n"
    prompt += ", ".join(triples)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(**inputs, max_length=50)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)