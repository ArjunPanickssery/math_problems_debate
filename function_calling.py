from typing import Callable, Dict, List, Union
import torch
torch.set_grad_enabled(False)
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import re
import ast, operator
import json
from langchain.tools import tool
from langchain_core.tools import render_text_description

from model_wrappers import HuggingFaceWrapper, ModelWrapper


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
    

def supports_tools(model: HuggingFaceWrapper) -> bool:
    return 'tool_use' in model.tokenizer.chat_template


class FunctionCallingLLM(ModelWrapper):
    TOOLS_PROMPT = """
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

The `arguments` should be a dictionary, with keys corresponding 
to the argument names and the values corresponding to the requested values."""
    def __init__(self, model, tools=None):
        self.model = model
        if tools is None:
            return model
    
        self.tools = tools
        self.tool_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        self.tool_map = {tool.__name__: tool for tool in tools}


    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}]
        if supports_tools(self):   # add switches for OpenAI vs. CLaude, vs HuggingFaceWrappers
            input_ids = self.tokenizer.apply_chat_template(messages,
                                                return_tensors="pt",
                                                add_generation_prompt=False, 
                                                chat_template='tool_use',
                                                tools=self.tools).to('cuda')
        else:
            messages.insert(0, {'role': 'system', 
                                'content': self.TOOLS_PROMPT.format(
                                    rendered_tools=render_text_description(self.tools)
                                    )})
            input_ids = self.tokenizer.apply_chat_template(messages,
                                                    return_tensors="pt",
                                                    add_generation_prompt=True).to('cuda')
        output = input_ids
        while True:
            output = self.model.model.generate(   # need to implement .generate(messages) -> str for underlying classes
                input_ids, 
            )

            # check if there are any <tool_call> tags. If no, then assume the argument is finished. Otherwise,
            # compute the result of the tool and append it to the output
            # strip off last character, assuming that assistant message ends with a single token (eos)
            assistant_toks = output[0][input_ids.shape[1]:-1]
            assistant_text = self.tokenizer.decode(assistant_toks)
            # print("Response finished, assistant text:", assistant_text)
            # maybe this should be a findall
            func_match = self.tool_pattern.search(assistant_text)#.group(1)
            if func_match is None:   # no tool call found, so we are done
                # print('no match found, done')
                break
            func_call = func_match.group(1) if func_match else None
            try:
                func_info = json.loads(func_call)
                func_name = func_info['name']
                args = func_info['arguments']
                # should add handling here to turn the args into the right types
                result = {'value': self.tool_map[func_name](**args),
                            'func_name': func_name, 
                            'args': args}
            except KeyError:
                result = "Invalid tool use"
            result_str = f"<tool_response>{result}</tool_response>"
            messages.append({'role': 'assistant', 'content': assistant_text})
            messages.append({'role': 'tool_call', 'content': result_str})
            input_ids = self.tokenizer.apply_chat_template(messages,
                                                            return_tensors="pt").to('cuda')
        return self.tokenizer.decode(output[0])
    

    def __getattr__(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return getattr(self.model, name)