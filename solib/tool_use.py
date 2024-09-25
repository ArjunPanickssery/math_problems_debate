from typing import Callable, Dict, List, Literal, Tuple, Union
import torch
torch.set_grad_enabled(False)
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import re
import ast, operator
import json

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, BaseMessage
from langchain.tools import tool, StructuredTool
from langchain_core.tools import render_text_description

from solib.utils import apply, apply_async


def math_eval(expr: str) -> Union[str, float]:
    """
    Evaluate a simple mathematical expression. The supported operators are * (multiply), - (subtract),
    + (add), / (divide), and ** (power). If your expression was unable to be evaluated,
    the output will instead be "Invalid expression".

    Args:
    expr: str, the expression to evaluate

    Returns:
    float, the result of the evaluation, or str "Invalid expression" if the expression is invalid
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.Pow: operator.pow
    }
    # Parse the expression into an AST
    node = ast.parse(expr, mode='eval').body

    def eval_node(node):
        if isinstance(node, ast.BinOp):
            if type(node.op) in operators:
                left = eval_node(node.left)
                right = eval_node(node.right)
                return operators[type(node.op)](left, right)
            else:
                raise ValueError(f"Unsupported operation {node.op}")
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            raise ValueError(f"Unsupported type {type(node)}")

    try:
        return eval_node(node)
    except ValueError as e:
        return "Invalid expression"


class ToolStopper(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
        self.tokenizer = tokenizer

        self.last_char = end_tag[-1]
        self.tokens_with_end_tag = self.token_with_char(self.last_char)
        self.start_tag = start_tag
        self.end_tag = end_tag

    def token_with_char(self, char: str):
        tokens_containing_char = []
        for tok_id in self.tokenizer.vocab.values():
            str_tok = self.tokenizer.decode(tok_id)
            if char in str_tok:
                tokens_containing_char.append(tok_id)
        return tokens_containing_char

    def find_tool_call(self, text: str):
        # Start from the last closing brace in the string
        end_pos = text.rfind(self.last_char)
        if end_pos == -1:
            return None  # No valid JSON block found
        end_tag = text[end_pos-len(self.end_tag)+1 : end_pos+1]
        if end_tag != self.end_tag:
            return None

        # Loop backward to find an opening brace and check the content in between
        start_pos = text.rfind(self.start_tag, 0, end_pos)
        if start_pos == -1:
            return None  # No more opening braces

        # Extract the candidate substring
        candidate = text[start_pos+len(self.start_tag) : end_pos + 1 - len(self.end_tag)]

        # Try parsing the candidate as JSON
        try:
            json_data = json.loads(candidate)

            return json_data  # Return the valid JSON block

        except json.JSONDecodeError:
            pass  # Invalid JSON block, continue searching

        return None  # No valid JSON block found



    def __call__(self, ids, scores) -> bool:
        ids = ids[0]  # assume unbatched
        if ids[-1] in self.tokens_with_end_tag:
            decoded = self.tokenizer.decode(ids)
            tool_call = self.find_tool_call(decoded)
            if tool_call is not None:
                return True
        return False

def get_structured_tools(tools: List[Callable | StructuredTool]) -> Tuple[Dict[str, Callable], List[StructuredTool]]:
    tool_map = {tool.__name__: tool for tool in tools}
    structured_tools = [(t if isinstance(t, StructuredTool) else tool(t)) for t in tools]
    return tool_map, structured_tools


def render_tools(tools: List[Union[Callable, StructuredTool]]) -> str:
    tool_text = render_text_description(get_structured_tools(tools)[1])
    return tool_text


def get_tool_prompt(tools: List[Union[Callable, StructuredTool]]) -> str:
    TOOLS_PROMPT = """
You are a tool calling LLM that has access to the following set of tools.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query.
Don\'t make assumptions about what values to plug into functions. Here are the available tools:

<tools>
{rendered_tools}
</tools>

Every tool call should be surrounded by <tool_call></tool_call> tags. And should take the form of a JSON object with the following format:
<tool_call>
{{"name": <function-name>, "arguments": <args-dict>}}
</tool_call>

The result will then be automatically supplied in the form of a JSON object with the following format:
<tool_result>
{{"name": <function-name>, "result": <result>, "arguments": <args-dict>}}
</tool_result>
"""
    rendered_tools = render_tools(tools)
    return TOOLS_PROMPT.format(rendered_tools=rendered_tools)


class HuggingFaceToolCaller:
    TOOL_RESULT_PROMPT = """
<tool_result>
{{
    "name": {name},
    "result": {result},
    "arguments": {arguments}
}}
</tool_result>
"""

    def __init__(self, tokenizer, model, tools=None):
        self.tools = tools
        self.tool_map = get_structured_tools(tools)[0]
        self.stopper = ToolStopper(tokenizer)

        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

    def natively_supports_tools(self):
        return isinstance(self.tokenizer.chat_template, dict) and 'tool_use' in self.tokenizer.chat_template

    def generate(self, input_ids, max_length, **kwargs):
        last_length = len(input_ids[0])
        while True:
            output = self.model.generate(input_ids, max_length=max_length, stopping_criteria=[self.stopper], **kwargs)
            generated_text = self.tokenizer.decode(output[0, last_length:])
            last_length = len(output[0])
            tool_call = self.stopper.find_tool_call(generated_text)
            if tool_call:
                tool_name = tool_call.get('name', 'Must supply the `name` field for valid tool calls')
                args = tool_call.get('arguments', {})
                try:
                    tool_result = self.tool_map.get(tool_name, lambda: 'Invalid tool name')(**args)
                except Exception as e:
                    tool_result = f"Error: {e}"
                # should add handling here to turn the args into the right types
                result = self.TOOL_RESULT_PROMPT.format(name=tool_name, result=tool_result, arguments=args)
                result_ids = self.tokenizer.encode(result, return_tensors='pt', add_special_tokens=False).to(self.device)
                input_ids = torch.cat([output, result_ids], dim=-1)
                last_length = len(input_ids[0])
            else:
                return output


def tool_use_loop_generate(input: List[BaseMessage],
                           get_response: Callable[[List[BaseMessage]], BaseMessage],
                           tool_map: Dict[str, Callable],
                           **kwargs):
    start_len = len(input)
    while True:
        response = apply(get_response,
                         **{'input': input,
                          **kwargs})
        input.append(response)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_result = tool_map[tool_name](**tool_args)
                input.append(ToolMessage(tool_result, tool_call_id=tool_call['id']))
        else:
            return input[start_len:]
