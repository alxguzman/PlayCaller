"""
play_caller.py (SCAFFOLD - Phase 4)
-----------------------------------
Claude API layer: turns the model's numeric output into a natural-language
play call, like a coordinator talking into the headset.

Example output:
  "It's 3rd & 7 from their 35, down 4 in the 4th quarter. Their defense
   has been giving up 0.15 EPA per pass over the last month - air it out.
   Recommendation: PASS (62% success probability)."

Requires ANTHROPIC_API_KEY in a .env file (see .env.example).
"""

# TODO: pip install anthropic python-dotenv
# TODO: load the API key with dotenv:  load_dotenv(); os.environ[...]
# TODO: client = anthropic.Anthropic()
# TODO: def explain_recommendation(situation: dict, model_output: dict) -> str
#       - build a prompt containing the game situation, both success
#         probabilities, and the relevant team-form stats
#       - call client.messages.create(model="claude-haiku-4-5-20251001", ...)
#         (Haiku is fast + cheap; ideal for short explanations)
#       - return the text response
# TODO: cache/limit calls so a demo doesn't burn API credits


def explain_recommendation(situation, model_output):
    """Generate a natural-language play call with Claude. (Not built yet.)"""
    raise NotImplementedError("Phase 4 - see TODOs above.")
