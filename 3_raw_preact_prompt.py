"""A ReAct agent written without any agent framework.

The only LLM integration is a raw REST call to the DeepSeek chat completions
endpoint. The prompt, the Thought/Action/Observation loop, and the tool
dispatch are all plain Python.
"""

import datetime
import inspect
import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

MAX_ITERATIONS = 10  # Maximum number of iterations for the agent loop
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_COMPLETIONS_PATH = "/chat/completions"

# Rewritten on every run so the file only ever holds the latest trace.
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_run_log.txt")

## Tool Definitions

def get_product_price(product: str) -> float:
    """
    Tool to look up the price of a product in the product catalog.
    """

    print(f"Looking up price for product: {product}")
    prices ={"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    
    return prices.get(product.lower(), 0.0)

def apply_discount(price: float, discount_tier: str) -> float:
    """
    Tool to apply a discount tier to a given price and return the final price.
    Available tiers: "bronze", "silver", "gold"
    """
    
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    print(f"Applying discount: Original price {price}, Discount {discount_tier}%, Discounted price {price * (1 - discount_percentages.get(discount_tier.lower(), 0) / 100)}")
    
    return round(price * (1 - discount / 100), 2)


tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}

def get_tool_descriptions(tools_dict):
    """
    Extracts the description of a tool function from its docstring.
    """
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        signature = inspect.signature(tool_function)
        docstring = " ".join((inspect.getdoc(tool_function) or "").split())
        descriptions.append(f"{tool_name}{signature}: {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""You are a helpful assistant that can look up product prices and apply discounts.
STRICTLY use the provided tools to answer user queries. Do not make up prices or discounts.
You must call the get_product_price and apply_discount tools to get accurate information and
never assume prices if they are not from the catalog. Only call apply_discount after you have
obtained the product price using get_product_price. Never calculate a price without using the
tools. If you don't know the answer, say 'I don't know'. If the user asks for a discount, you
must first get the product price using get_product_price and then apply the discount using
apply_discount. Always use the tools provided to get accurate information.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values in the order shown by the tool signature
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

## Run log

_SEPARATOR = "=" * 70

# Matches a JSON escape sequence in already-dumped text. A single left-to-right
# pass matters: "\\n" (escaped backslash + n) must not become a line break.
_JSON_ESCAPE = re.compile(r"\\(n|t|\\)")
_UNESCAPED = {"n": "\n", "t": "\t", "\\": "\\"}


def render_json(data) -> str:
    """Pretty-print JSON on multiple lines with real line breaks instead of escaped \\n."""
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return _JSON_ESCAPE.sub(lambda match: _UNESCAPED[match.group(1)], text)


class RunLog:
    """Writes the prompt, every request body, and every response to a .txt file.

    The file intentionally mirrors the console output instead of replacing it.
    Files are opened with UTF-8 so long prompts stay readable in Notepad.
    """

    def __init__(self, path, question, prompt):
        self.path = path
        self._file = open(path, "w", encoding="utf-8")
        self._write(
            f"{_SEPARATOR}\n"
            f"REACT AGENT RUN LOG\n"
            f"{_SEPARATOR}\n"
            f"started        : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"model          : {MODEL}\n"
            f"endpoint       : POST {BASE_URL}{CHAT_COMPLETIONS_PATH}\n"
            f"max iterations : {MAX_ITERATIONS}\n"
            f"question       : {question}\n"
            f"{_SEPARATOR}\n"
            f"FULL PROMPT SENT (iteration 1, before any observation)\n"
            f"{_SEPARATOR}\n"
            f"{prompt}\n"
        )
        self._file.flush()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def start_iteration(self, iteration):
        self._write(f"\n{_SEPARATOR}\nITERATION {iteration}\n{_SEPARATOR}\n")

    def request(self, method, url, payload):
        self._write(
            f"REST CALL    : {method}\n"
            f"URL          : {url}\n"
            f"REQUEST BODY (JSON)\n"
            f"{render_json(payload)}\n"
        )

    def response(self, response):
        self._write(f"\nRESPONSE (HTTP {response.status_code}) (JSON)\n")
        try:
            body = render_json(response.json())
        except ValueError:
            body = response.text
        self._write(f"{body}\n")
        self._file.flush()

    def _write(self, text):
        self._file.write(text)


## Raw REST call to DeepSeek


def call_deepseek(prompt: str, log=None) -> str:
    """Send one prompt to DeepSeek over plain HTTP and return the completion text."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to your .env file.")

    url = f"{BASE_URL}{CHAT_COMPLETIONS_PATH}"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        # Stop the completion before the model can invent its own Observation.
        "stop": ["\nObservation:"],
    }

    if log is not None:
        log.request("POST", url, payload)

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    if log is not None:
        log.response(response)

    if not response.ok:
        raise RuntimeError(f"DeepSeek request failed ({response.status_code}): {response.text}")
    return response.json()["choices"][0]["message"]["content"] or ""


## ReAct output parsing


def parse_final_answer(model_output: str):
    """Return the text after 'Final Answer:', or None while the model is still reasoning."""
    match = re.search(r"Final Answer\s*:\s*(.*)", model_output, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_action(model_output: str):
    """Return (tool_name, raw_action_input) parsed out of a ReAct response."""
    action_match = re.search(r"^\s*Action\s*:\s*(.+)$", model_output, re.MULTILINE)
    if action_match is None:
        return None, None

    tool_name = re.split(r"[\s(]", action_match.group(1).strip())[0].strip("\"'`")

    remaining = model_output[action_match.end():]
    input_match = re.search(r"Action\s*Input\s*:\s*(.*)", remaining, re.DOTALL)
    if input_match is None:
        return tool_name, ""

    raw_action_input = re.split(r"\n\s*(?:Thought|Observation|Action)\s*:", input_match.group(1))[0]
    return tool_name, raw_action_input.strip().strip("\"'`").strip()


def coerce_argument(value, annotation):
    """Cast an Action Input value to the type declared in the tool signature."""
    if annotation is float:
        return float(value)
    if annotation is int:
        return int(float(value))
    if annotation is bool:
        return str(value).strip().lower() in {"true", "1", "yes"}
    return str(value)


def build_tool_kwargs(tool_function, raw_action_input: str) -> dict:
    """Turn the raw Action Input into keyword arguments for the tool.

    The prompt asks for comma separated values in signature order, but a JSON
    object such as {"price": 29.99, "discount_tier": "bronze"} also works.
    """
    if not raw_action_input:
        raise ValueError("No Action Input provided.")

    parameters = list(inspect.signature(tool_function).parameters.values())

    if raw_action_input.startswith("{"):
        provided = json.loads(raw_action_input)
        missing = [parameter.name for parameter in parameters if parameter.name not in provided]
        if missing:
            raise ValueError(f"Missing arguments: {', '.join(missing)}")
        values = [provided[parameter.name] for parameter in parameters]
    else:
        if len(parameters) == 1:
            values = [raw_action_input]
        else:
            values = [part.strip().strip("\"'`") for part in raw_action_input.split(",")]
        if len(values) != len(parameters):
            raise ValueError(
                f"Expected {len(parameters)} comma separated value(s) "
                f"({', '.join(parameter.name for parameter in parameters)}), got {len(values)}"
            )

    return {
        parameter.name: coerce_argument(value, parameter.annotation)
        for parameter, value in zip(parameters, values)
    }


## Agent Loop Implementation


def run_agent(question: str):
    print(f"Starting agent loop for question: {question}")
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    with RunLog(LOG_PATH, question, prompt) as log:
        print(f"Writing run log to: {LOG_PATH}")

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\nIteration {iteration}:")
            log.start_iteration(iteration)
            model_output = call_deepseek(prompt + scratchpad, log)
            print(model_output)

            final_answer = parse_final_answer(model_output)
            if final_answer is not None:
                print("Final answer from AI:")
                print(final_answer)
                return final_answer

            tool_name, raw_action_input = parse_action(model_output)
            if tool_name is None:
                print("The model produced neither an Action nor a Final Answer. Stopping.")
                return None

            print(f"Tool call detected: {tool_name} with input {raw_action_input!r}")
            tool_to_use = tools.get(tool_name)

            try:
                if tool_to_use is None:
                    raise ValueError(f"Tool {tool_name} not found. Available tools: {tool_names}")
                observation = tool_to_use(**build_tool_kwargs(tool_to_use, raw_action_input))
            except (ValueError, TypeError) as error:
                # Feed the failure back as an Observation so the model can correct itself.
                observation = f"Error: {error}"

            print(f"Observation from tool {tool_name}: {observation}")
            scratchpad += f"{model_output.strip()}\nObservation: {observation}\nThought:"

        print("Maximum iterations reached without a final answer.")
        return None


if __name__ == "__main__":
    result = run_agent("Mouse prices while apply a bronze discount to it?")
    print(result)