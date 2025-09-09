import spacy

nlp = spacy.load("nl_core_news_sm")  # Dutch model


def analyze_text(text):
    doc = nlp(text)
    strategies = []
    for sent in doc.sents:
        if "vraag" in sent.text.lower():
            strategies.append({"type": "question", "text": sent.text})
    return strategies
