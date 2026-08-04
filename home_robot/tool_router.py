"""Pick the smallest TOOLS subset an utterance could plausibly need.

Why this exists: on FastFlowLM/NPU the whole tool schema is re-read on every
single request — there is no prefix caching (measured 2026-07-31: the same
call twice took 6.7 s then 6.6 s). Generation itself is fast; the prefill is
what the user experiences as "it takes ages to answer":

    all 15 tools   6.7 s   1941 prompt tokens
    3 tools        4.1 s    879
    no tools       3.8 s    471

So "τι ώρα είναι;" was paying ~3 s to re-read how to pick up a cup. Sending
only the relevant group is worth ~2-3 s on every utterance, and the model's
answers are unchanged — same model, same prompt, same max_tokens.

The risk is a false negative: an action word we failed to list means the tool
never reaches the model and the robot silently does nothing. Three things
bound that risk:

  1. Groups match on stems (`πηγαιν\\w*`), not whole words, so inflections and
     the STT's habit of gluing punctuation on are covered.
  2. ACTION_RE is a catch-all — any generic "do something" verb that matched no
     group sends the *full* set rather than none. Unusual phrasings degrade to
     today's behaviour (slow but correct), never to a dropped command.
  3. Only an utterance with no action signal at all gets the empty set, and
     those are exactly the ones that cannot need a tool.

`stop` is deliberately not routed here — stop_command.py gates it with keywords
*before* the model is called at all, because it must not be probabilistic.
"""
import re

from .stop_command import strip_accents


# Stems, not words: 'πηγαιν' covers πήγαινε/πηγαίνεις/πηγαίνετε. Written
# accent-free because every input is passed through strip_accents() first.
_GROUPS = [
    # Navigation is split into four narrow groups rather than one broad one:
    # as a single group "Πήγαινε στο σαλόνι" dragged along explore/patrol/
    # follow/move and saved only 1.2 s. Split, it sends 3 tools instead of 9.
    # goto_object rides with goto: both answer "πήγαινε στο …", and only the
    # model can tell whether what follows is a taught room or a thing it has
    # seen. Splitting them by keyword would need a list of every object name.
    (['goto', 'goto_object', 'dock', 'tidy'],
     # 'πηγαι\w*' not 'πηγαιν\w*': the STT writes "Πήγαισε" for "πήγαινε"
     # often enough that the narrower stem loses real commands.
     # Deliberately NOT here: 'παω'/'παμε' — first person is the *user*
     # narrating ("Πάω για ύπνο"), not an order. "Πάμε στο σαλόνι" still
     # routes, on the room name.
     r'πηγαι\w*|παε|ελα|ερχ\w*|σαλονι\w*|κουζιν\w*|διαδρομ\w*|'
     r'τουαλετ\w*|δωματι\w*|μπαμπα|βαση|φορτισ\w*|dock|καθαρ\w*|σκουπ\w*|'
     r'τακτοποι\w*|συγυρ\w*'),

    (['move'],
     r'μπροστα|πισω|αριστερ\w*|δεξι\w*|στριψ\w*|στροφ\w*|γυρν\w*|γυρισ\w*|'
     r'προχωρ\w*|κουνησ\w*|μετακιν\w*|οπισθεν'),

    (['patrol', 'explore', 'stop_explore'],
     r'περιπολ\w*|εξερευν\w*|χαρτογραφ\w*|γυρε\s|βολτ\w*'),

    (['follow', 'stop_follow'],
     r'ακολουθ\w*'),

    (['open_app', 'close_app', 'list_windows'],
     r'ανοιξ\w*|ανοιγ\w*|κλεισ\w*|οθον\w*|εφαρμογ\w*|παραθυρ\w*|'
     r'υπολογιστ\w*|firefox|φαιρφοξ|τερματικ\w*|κειμενογραφ\w*|'
     r'αριθμομηχαν\w*|κομπιουτεραρ\w*|βιντεο|pdf|vscode|browser|'
     r'φυλλομετρ\w*|αρχει\w*'),

    (['system_status'],
     r'κατασταση|καταστασ\w*|μπαταρι\w*|cpu|επεξεργαστ\w*|μνημη|ram|'
     r'δισκ\w*|θερμοκρασ\w*|βαθμ\w*|uptime|ζεστα\w*|φορτι\w*'),

    # 'μαζεψ/μαζευ/τακτοποι/συγυρ' also appear in the goto/tidy group above, so
    # both reach the model and it picks — `tidy` drives and reports, `sort`
    # actually carries things. Only the model can tell which the user meant.
    (['sort'],
     r'μαζεψ\w*|μαζευ\w*|τακτοποι\w*|συγυρ\w*|παιχνιδ\w*|στη\s+θεση'),

    (['report_clutter', 'pick', 'fetch'],
     r'σκορπι\w*|ακαταστασ\w*|clutter|αντικειμεν\w*|πιασ\w*|φερ\w*|'
     r'μαζεψ\w*|μαζευ\w*|σηκωσ\w*|κουπ\w*|μπουκαλ\w*|βιβλι\w*|ποτηρ\w*|'
     r'πιατ\w*|παιχνιδ\w*'),

    # ‼️ The arm has its own group, separate from pick/fetch. "κάνε τον
    # βραχίονα λίγο μπροστά" is not a request to grab anything, and routing it
    # to `pick` would send the robot hunting for an object that was never
    # named. 'χερι' is here too — people say "σήκωσε το χέρι σου" and mean the
    # arm; `pick`'s own keywords (πιασ/φερ) stay out of it.
    (['arm'],
     r'βραχιον\w*|χερι\w*|αρθρωσ\w*|ωμο\w*|αγκων\w*|καρπ\w*|δαγκαν\w*|'
     r'γαντζ\w*|απλωσ\w*|τεντωσ\w*|διπλωσ\w*'),

    (['remember'],
     r'θυμ\w*|σημειωσ\w*|κρατα|μαθε'),

    # "πήγαινε να ΔΕΙΣ αν έκλεισα το παράθυρο", "τσέκαρε την κουζίνα".
    # 'δες'/'δει' deliberately narrow: 'δειχν\w*' is "δείξε μου", a different
    # thing, and bare 'δ' stems would swallow half the language.
    (['check'],
     r'τσεκαρ\w*|ελεγξ\w*|ελεγχ\w*|\bδες\b|\bδει\b|\bδεις\b|επιβεβαιωσ\w*|'
     r'παραθυρ\w*|πορτ\w*|φως|φωτα|κλειστ\w*|ανοιχτ\w*|ξεχασ\w*'),

    # "πόσο μακριά είναι;" / "πόση απόσταση έχει ο άνθρωπος;". 'κοντα' is left
    # out on purpose: "έλα κοντά μου" is a movement command, not a measurement.
    (['distance_to'],
     r'αποστασ\w*|μακρι\w*|ποσα\s+μετρα|ποσο\s+ψηλ\w*'),

    # "πήγαινε εκεί" while pointing. The deictic word is the whole signal —
    # 'εκει'/'εδω' mean nothing to any other tool, so this group is narrow and
    # safe. It rides with `goto` because the STT often drops the demonstrative
    # ("πήγαινε ↴ εκεί" → "πήγαινε"), and then only the model, seeing both
    # tools, can tell a pointed spot from a named room.
    (['goto_pointed', 'goto'],
     r'\bεκει\b|\bεδω\b|\bαυτο\s+το\s+σημειο\b|\bδειχν\w*'),

    # "τι έγινε σήμερα;", "πότε ήρθε ο Μαξ;", "τι είπα χθες;".
    # ‼️ A bare day word is NOT enough. Routing on 'σημερα' alone caught
    # "τι μέρα έχουμε σήμερα;" — chitchat that must send no tools at all. So
    # the trigger is the recall verb ('τι έγινε', 'θυμάσαι τι', 'τι είπε') or
    # the interrogative 'πότε'; the day word only narrows the window later,
    # inside episodic.parse_time_window.
    # "βρες τα κλειδιά", "ψάξε το πορτοφόλι". Open vocabulary, so it cannot be
    # keyed on object names — the whole point is that the noun is unknown in
    # advance. Keyed on the SEARCH verb instead. 'βρε\w*' also matches "βρες"
    # and "βρήκες"; it is already in ACTION_RE, so no new false-positive class.
    (['find'],
     r'\bβρες\b|\bβρε\b|ψαξ\w*|ψαχν\w*|εντοπισ\w*|\bχαθηκ\w*|\bεχασα\b'),

    (['recall'],
     r'\bποτε\b|εγινε|εγιναν|συνεβη|θυμασαι\s+τι|τι\s+ειπ\w*|'
     r'ιστορικ\w*|νωριτερα'),
]

_COMPILED = [(names, re.compile(pat)) for names, pat in _GROUPS]

# Catch-all "the user is asking for something to be done" signal. Only consulted
# when no group matched; it sends the full set, i.e. today's behaviour. Matching
# here is a *cost* (slow path), never a correctness problem, so it can be loose.
ACTION_RE = re.compile(
    # 'κανε' (imperative) only — 'καν\\w*' also matches "τι κάνεις;", which is
    # a greeting and was dragging the full 15-tool schema into every hello.
    r'\bκανε\b|βοηθ\w*|θελω\s+να|μπορεις\s+να|σε\s+παρακαλ\w*|'
    r'πρεπει\s+να|ξεκιν\w*|αρχισ\w*|σταματ\w*|τελειωσ\w*|δωσ\w*|βαλ\w*|'
    r'βγαλ\w*|ψαξ\w*|βρε\w*|δειξ\w*|ελεγξ\w*'
)


def select_tools(text, tools):
    """Return the subset of `tools` worth sending for this utterance.

    An empty list means "pure conversation" — send no tools at all, which is
    the cheapest prefill and cannot break anything, because a reply that needs
    no action is the only thing that reaches it.
    """
    normalized = strip_accents(text)

    wanted = set()
    for names, pattern in _COMPILED:
        if pattern.search(normalized):
            wanted.update(names)

    if not wanted:
        # Nothing specific matched. An action-ish verb still means "do
        # something" we failed to classify — fall back to the full schema
        # rather than dropping the command.
        return list(tools) if ACTION_RE.search(normalized) else []

    return [t for t in tools if t['function']['name'] in wanted]
