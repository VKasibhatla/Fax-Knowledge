# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = "Thomas Jefferson founded the University of Virginia."
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

print()
str = "Thomas Jefferson founded the University of Virginia."
new_str =  str.replace(list(doc.noun_chunks)[-1].text, "[MASK]")
print("New string:", new_str)
# print(doc.ents[-1].text.rfind)

