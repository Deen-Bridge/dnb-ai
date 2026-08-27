PHONEMES = {"a","i","u","aa","ii","uu","b","t","th","j","H","kh","d","dh","r","z","s","sh","S","D","T","Z","3","gh","f","q","k","l","m","n","h","w","y"}
def arabic_text_to_phonemes(text):
    return [c for c in text if c in PHONEMES]