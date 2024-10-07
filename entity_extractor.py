# entity_extractor.py contains the EntityExtractor class that extracts entities from the processed user input.

class EntityExtractor:
    def __init__(self, nlp):
        self.nlp = nlp

    def extract_entities(self, doc):
        """Extract entities from the processed user input."""
        entities = {}
        for ent in doc.ents:
            entities[ent.label_] = ent.text
        return entities
