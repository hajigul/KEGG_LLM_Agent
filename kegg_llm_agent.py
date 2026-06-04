"""
KEGG50k LLM Agent (local, no OpenAI key)
-----------------------------------------
A local LLM (via Ollama) that uses your trained relation-prediction model
as a tool to answer drug-relation-discovery questions in natural language.

Requires:
    - Ollama installed and running (https://ollama.com)
    - a pulled model, e.g.:  ollama pull llama3.2:3b
    - pip install torch numpy ollama
    - kegg_agent.py (from before) in the same folder, model already trained.

Run:
    python kegg_llm_agent.py
"""

import json
import re
import ollama

# Reuse the trained predictor from the previous script.
from kegg_agent import RelationDiscoveryAgent

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
LLM_MODEL = "llama3.2:3b"   # change to "llama3.2:1b" or "qwen2.5:3b" if needed
TOP_K = 5

# ----------------------------------------------------------------------
# THE TOOL: a function the LLM can call
# ----------------------------------------------------------------------
print("Loading relation-prediction model...")
predictor = RelationDiscoveryAgent()
print("Model loaded.\n")


def predict_relation(head: str, tail: str, top_k: int = TOP_K):
    """Tool: predict the relation(s) between a head and tail entity."""
    result = predictor.predict(head, tail, top_k=top_k)
    if isinstance(result, str):          # error message (unknown entity)
        return {"error": result}
    return {
        "head": head,
        "tail": tail,
        "predictions": [
            {"relation": rel, "confidence": conf} for rel, conf in result
        ],
    }


# Tool schema given to the LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_relation",
            "description": (
                "Predict the most likely relation(s) between a head entity "
                "and a tail entity in the KEGG50k drug-discovery knowledge "
                "graph. Use this whenever the user asks how two biomedical "
                "entities (drugs, genes, diseases, pathways, networks) are "
                "related. Entity IDs look like D01920 (drug), HSA:495 (gene), "
                "H00787 (disease), hsa04080 (pathway), N00060 (network)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "head": {
                        "type": "string",
                        "description": "The head entity ID, e.g. 'H00787'.",
                    },
                    "tail": {
                        "type": "string",
                        "description": "The tail entity ID, e.g. 'hsa04080'.",
                    },
                },
                "required": ["head", "tail"],
            },
        },
    }
]

AVAILABLE_FUNCTIONS = {"predict_relation": predict_relation}

SYSTEM_PROMPT = (
    "You are a biomedical knowledge-graph assistant for the KEGG50k "
    "drug-discovery dataset. When a user asks about the relationship "
    "between two entities, call the predict_relation tool with the head "
    "and tail IDs. After receiving the tool result, explain the predicted "
    "relation(s) and their confidence scores in clear, plain language. "
    "If the tool reports an unknown entity, tell the user the ID was not "
    "found in the dataset. Be concise."
)


# ----------------------------------------------------------------------
# AGENT LOOP
# ----------------------------------------------------------------------
def run_agent(user_message, history):
    history.append({"role": "user", "content": user_message})

    # First call: let the LLM decide whether to use the tool.
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        tools=TOOLS,
    )
    msg = response["message"]
    history.append(msg)

    # If the LLM requested a tool call, execute it.
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for call in tool_calls:
            fname = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            func = AVAILABLE_FUNCTIONS.get(fname)
            if func is None:
                result = {"error": f"Unknown tool: {fname}"}
            else:
                result = func(**args)
            history.append({
                "role": "tool",
                "content": json.dumps(result),
            })

        # Second call: let the LLM explain the tool result.
        final = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        )
        final_msg = final["message"]
        history.append(final_msg)
        return final_msg["content"]

    # No tool call — just return the LLM's text.
    return msg["content"]


def extract_ids_fallback(text):
    """If the model doesn't call the tool, try to pull two IDs from text."""
    ids = re.findall(r"[A-Za-z]+\d+|HSA:\d+|hsa\d+|[A-Z]\d{4,}", text)
    return ids[:2] if len(ids) >= 2 else None


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" KEGG50k Drug-Relation Discovery LLM Agent")
    print(f" LLM: {LLM_MODEL} (local via Ollama)")
    print("=" * 60)
    print("Ask things like:")
    print("  What is the relation between H00787 and hsa04080?")
    print("  How are D01920 and HSA:495 connected?")
    print("Type 'quit' to exit.\n")

    history = []
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("quit", "exit", "q"):
            break
        if not user:
            continue

        try:
            answer = run_agent(user, history)
            print(f"\nAgent> {answer}\n")
        except Exception as e:
            # Fallback path if tool-calling misbehaves on a tiny model
            print(f"[tool-calling failed: {e}] trying direct extraction...")
            ids = extract_ids_fallback(user)
            if ids:
                res = predict_relation(ids[0], ids[1])
                print(f"\nAgent> {json.dumps(res, indent=2)}\n")
            else:
                print("\nAgent> Sorry, I couldn't parse two entity IDs.\n")


if __name__ == "__main__":
    main()