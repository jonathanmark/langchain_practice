"""Bare-bones ReAct + chain-of-thought loop. No LangChain, no logging, no helpers.

  1. Send prompt + scratchpad to the model as one string.
  2. The model replies with plain text: Thought / Action / Action Input.
  3. Parse that text, run the matching Python function, turn its return value into text.
  4. Append "<model text>\\nObservation: <result>\\nThought:" to the scratchpad and go to 1.
  5. Stop when the reply contains "Final Answer:".
"""

import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

MODEL = "deepseek-flash"
MAX_STEPS = 10


## Tools (the model never sees these bodies)


def get_product_price(product):
    prices = {"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    return prices.get(product.lower(), 0.0)


def apply_discount(price, discount_tier):
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


## Prompt (this text is the model's only knowledge of the tools)

TOOL_LIST = """get_product_price(product) -> float: price of a product (laptop, mouse, keyboard)
apply_discount(price: float, discount_tier) -> float: final price after a bronze, silver or gold discount"""

PROMPT = f"""You are a helpful assistant with access to these tools:

{TOOL_LIST}

Answer using exactly this format:

Thought: what you should do next
Action: one of [get_product_price, apply_discount]
Action Input: the arguments, comma separated, in the order the tool expects
Observation: the tool result, written by the system, never by you

Repeat Thought/Action/Action Input/Observation as needed, then write:

Thought: I now know the final answer
Final Answer: the answer

Question: {{question}}
Thought:"""


## HTTP


def ask_llm(text):
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            # Without this the model would write the Observation itself.
            "stop": ["\nObservation:"],
        },
        timeout=60,
    )
    return response.json()["choices"][0]["message"]["content"]


## Text -> real function call


def run_tool(action, action_input):
    if action == "get_product_price":
        return get_product_price(action_input)

    if action == "apply_discount":
        price, discount_tier = [part.strip() for part in action_input.split(",")]
        return apply_discount(float(price), discount_tier)

    return f"Error: unknown tool {action}"


## Agent loop


def run_agent(question):
    prompt = PROMPT.format(question=question)
    scratchpad = ""

    for _ in range(MAX_STEPS):
        text = ask_llm(prompt + scratchpad)
        print(text.strip())

        if "Final Answer:" in text:
            return text.split("Final Answer:", 1)[1].strip()

        action_line = re.search(r"Action:\s*(.+)", text)
        input_line = re.search(r"Action Input:\s*(.+)", text)
        if action_line is None or input_line is None:
            return text.strip()  # model ignored the format; hand back what it said

        observation = run_tool(action_line.group(1).strip(), input_line.group(1).strip())
        print(f"Observation: {observation}")
        scratchpad += f"{text.strip()}\nObservation: {observation}\nThought:"

    return "Stopped: too many steps"


if __name__ == "__main__":
    print(run_agent("Mouse price with a bronze discount?"))
