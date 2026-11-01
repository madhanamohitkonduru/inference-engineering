"""Send 50 fixed prompts at temperature 0 (100 tokens max) to the running server; save the outputs.
Usage: python quality_gen.py <label>      e.g. A_run1, A_run2, B, C   (label = which server config is running)
Plain /v1/completions (no chat template), one request at a time."""
import json, sys, requests

PROMPTS = [
    "The capital of France is", "Explain what a hash table is in two sentences:", "Write a haiku about autumn rain:",
    "Python code to reverse a string:\n", "The three primary colors are", "In 1969, humans first",
    "A good way to learn a new language is", "The difference between a list and a tuple in Python is",
    "Photosynthesis is the process by which", "Q: What is 17 times 23?\nA:", "Once upon a time, in a small village by the sea,",
    "The main causes of the French Revolution were", "To make a cup of tea, first", "The Pythagorean theorem states that",
    "A binary search works by", "The water cycle consists of", "Write a short thank-you note to a teacher:",
    "SQL query to count rows in a table named users:\n", "The speed of light is approximately", "Three tips for a job interview:\n1.",
    "Machine learning is", "The Great Wall of China was built to", "Summarize the plot of Romeo and Juliet in three sentences:",
    "An example of a renewable energy source is", "The human heart has four chambers:", "What is the difference between weather and climate?",
    "Dear customer, we are sorry to inform you that", "A neural network learns by", "The largest planet in our solar system is",
    "Write a limerick about a cat:", "List five fruits and their colors:\n", "Why is the sky blue? Because",
    "The first step in solving a quadratic equation is", "JavaScript function that adds two numbers:\n", "Mount Everest is located in",
    "The benefits of regular exercise include", "Explain recursion to a ten-year-old:", "Newton's three laws of motion are",
    "A typical day for a software engineer starts with", "The Roman Empire fell because", "To center a div in CSS you can",
    "The chemical formula for water is H2O, which means", "Describe the taste of a ripe mango:", "The internet works by",
    "An important rule of good writing is", "Compare cats and dogs as pets:", "The Industrial Revolution began in",
    "If a train travels 60 miles in 1.5 hours, its average speed is", "Meditation can help people by", "The theory of evolution says that",
]
assert len(PROMPTS) == 50
label = sys.argv[1]
out = []
for p in PROMPTS:
    r = requests.post("http://localhost:8000/v1/completions", timeout=120,
                      json={"model": "Qwen/Qwen3-0.6B", "prompt": p, "max_tokens": 100, "temperature": 0})
    r.raise_for_status()
    out.append({"prompt": p, "text": r.json()["choices"][0]["text"]})
json.dump(out, open(f"outputs_{label}.json", "w"), indent=1)
print(f"{label}: saved {len(out)} outputs; example: {out[0]['prompt']!r} -> {out[0]['text'][:80]!r}")
