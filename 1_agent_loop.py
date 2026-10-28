from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable

MAX_ITERATIONS = 10  # Maximum number of iterations for the agent loop
MODEL = "deepseek-flash"  # Model name for the ChatDeepSeek model

## Tool Definitions

@tool
def get_product_price(product: str) -> float:
    """
    Tool to look up the price of a product in the product catalog.
    """

    print(f"Looking up price for product: {product}")
    prices ={"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    
    return prices.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Tool to apply a discount tier to a given price and return the final price.
    Available tiers: "bronze", "silver", "gold"
    """
    
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    print(f"Applying discount: Original price {price}, Discount {discount_tier}%, Discounted price {price * (1 - discount_percentages.get(discount_tier.lower(), 0) / 100)}")
    
    return round(price * (1 - discount / 100), 2)


## Agent Loop Implementation

@traceable(name="agent_loop", description="Agent loop that processes user queries and interacts with tools.", tags=["agent", "loop", "tools"])
def run_agent(question: str):

    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"deepseek:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Starting agent loop for question: {question}")

    messages = [
        SystemMessage(content="""You are a helpful assistant that can look up product prices 
        and apply discounts. STRICTLY use the provided tools to answer user queries. 
        Do not make up prices or discounts. You must call the get_product_price and apply_discount 
        tools to get accurate information and never assume prices if they are not from the catalog. 
        Only call apply_discount after you have obtained the product price using get_product_price. 
        Do not make up any made-up number. Never calculate the price without using the tools. 
        If you don't know the answer, say 'I don't know'. If the user asks for a discount, 
        you must first get the product price using get_product_price and then apply the discount using apply_discount. 
        Do not make up any prices or discounts. Always use the tools provided to get accurate information."""),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration}:")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If no tool calls were made, return the AI's final response
        if not tool_calls:
            print("No tool calls made. Final response from AI:")
            print(ai_message.content)
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"Tool call detected: {tool_name} with arguments {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        
        if tool_to_use is None:
            print(f"Tool {tool_name} not found. Skipping this tool call.")
            raise ValueError(f"Tool {tool_name} not found.")

        observation = tool_to_use.invoke(tool_args)
        print(f"Observation from tool {tool_name}: {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id)) 
    
    print("Maximum iterations reached. Final response from AI:")
    print(ai_message.content)
    return None
        

if __name__ == "__main__":
    result = run_agent("Mouse prices while apply a bronze discount to it?")