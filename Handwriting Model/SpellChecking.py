from spellchecker import SpellChecker

spell = SpellChecker()

words = ["Locotion", "navar", "bronch", "nevar"]

for word in words:
    suggestions = spell.candidates(word)
    print(f"Suggested Correction for the word \"{word}\" : {list(suggestions)}")


 