#!/bin/bash
# Διαδραστικός φωνητικός βοηθός: qwen3:8b (Ollama) -> απάντηση στα ελληνικά -> spd-say

MODEL="qwen3-el"
SYSTEM_PROMPT="Είσαι ένας εξυπηρετικός βοηθός. Απαντάς πάντα στα ελληνικά, σύντομα και ξεκάθαρα, σε φυσική γλώσσα κατάλληλη για εκφώνηση (χωρίς markdown, αστεράκια, λίστες, σύμβολα κώδικα ή emoji)."

# Αφαιρεί emoji/εικονιδιακά σύμβολα ώστε να μην τα "διαβάζει" το spd-say
strip_emoji() {
    python3 -c '
import sys, re
text = sys.stdin.read()
pattern = re.compile(
    "["
    "\U0001F000-\U0001FFFF"
    "\U00002600-\U000027BF"
    "\U00002300-\U000023FF"
    "\U00002B00-\U00002BFF"
    "\U00002190-\U000021FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)
sys.stdout.write(pattern.sub("", text))
'
}

echo "Φωνητικός βοηθός Qwen3 (ελληνικά). Γράψε 'exit' για έξοδο."
echo

while true; do
    read -r -e -p "Εσύ: " prompt
    [ -z "$prompt" ] && continue
    [[ "$prompt" == "exit" || "$prompt" == "έξοδος" ]] && break

    response=$(curl -s http://localhost:11434/api/chat -d "$(jq -n \
        --arg sys "$SYSTEM_PROMPT" \
        --arg msg "$prompt" \
        --arg model "$MODEL" \
        '{model: $model, messages: [{role:"system", content:$sys},{role:"user", content:$msg}], stream: false, think: false}')" \
        | jq -r '.message.content')

    echo "Qwen: $response"
    echo "$response" | strip_emoji | spd-say -l el -e
    echo
done
