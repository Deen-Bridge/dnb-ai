from arabic_phonemes import arabic_text_to_phonemes
class TajweedProcessor:
    def process(self, text):
        return [{"symbol":p,"duration_factor":1.0,"marks":[]} for p in arabic_text_to_phonemes(text)]