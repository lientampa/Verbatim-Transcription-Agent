"""Exact edit-distance metrics; NFC Vietnamese syllable-oriented tokenization."""
import re
import unicodedata


def normalize(text, relaxed=False):
    value = " ".join(unicodedata.normalize("NFC", text).split())
    return value.casefold() if relaxed else value


def tokens(text, relaxed=False):
    text = normalize(text, relaxed)
    return re.findall(r"\[không rõ\]|\w+", text) if relaxed else re.findall(r"\[không rõ\]|\S+", text)


def ratio(numerator, denominator):
    return {"value": numerator/denominator if denominator else None, "numerator":numerator,
            "denominator":denominator, "available":bool(denominator),
            **({"reason":"ZERO_DENOMINATOR"} if not denominator else {})}


def edit_counts(reference, model):
    """Levenshtein, linear memory; tie order diagonal, deletion, insertion."""
    reference, model = list(reference), list(model)
    original_length = len(reference)
    first = 0
    while first < min(len(reference),len(model)) and reference[first] == model[first]:
        first += 1
    reference, model = reference[first:], model[first:]
    while reference and model and reference[-1] == model[-1]:
        reference.pop(); model.pop()
    # Each cell stores (cost, substitutions, deletions, insertions).
    row = [(j,0,0,j) for j in range(len(model)+1)]
    for i, expected in enumerate(reference,1):
        following = [(i,0,i,0)]
        for j, actual in enumerate(model,1):
            c,s,d,ins = row[j-1]
            diagonal = (c+(expected!=actual),s+(expected!=actual),d,ins)
            c,s,d,ins = row[j]
            deletion = (c+1,s,d+1,ins)
            c,s,d,ins = following[-1]
            insertion = (c+1,s,d,ins+1)
            following.append(min((diagonal,deletion,insertion),key=lambda x:x[0]))
        row = following
    cost,s,d,i = row[-1]
    return {"substitutions":s,"deletions":d,"insertions":i,"reference_count":original_length,
            "error_rate":ratio(cost,original_length)}


def text_metrics(reference, model):
    return {"strict_wer":edit_counts(tokens(reference),tokens(model)),
            "relaxed_wer":edit_counts(tokens(reference,True),tokens(model,True)),
            "cer":edit_counts(normalize(reference),normalize(model))}
