from typing import Callable, Dict, List, Any
import torch
from transformers import AutoTokenizer, StoppingCriteria
import json
import instructor
from instructor import Mode
from openai import OpenAI, AsyncOpenAI
from solib.globals import TOOL_CALL_START_TAG, TOOL_CALL_END_TAG
from solib.tool_use.tool_rendering import get_structured_tools, render_tool_call_result
from pydantic import BaseModel, Field

torch.set_grad_enabled(False)


class ToolCall(BaseModel):
    name: str = Field(..., description="Name of the tool to call")
    args: Dict[str, Any] = Field(..., description="Arguments to pass to the tool")


class ToolStopper(StoppingCriteria):
    """A stopping criteria that detects tool calls in the output of a model and returns the json if found.
    Tool calls are expected to be embedded between a start and end tag (XML-style), with a json block
    in between that corresponds to the arguments and name of the tool call."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        start_tag: str = TOOL_CALL_START_TAG,
        end_tag: str = TOOL_CALL_END_TAG,
    ):
        self.tokenizer = tokenizer

        self.last_char = end_tag[-1]
        self.tokens_with_end_tag = self.token_with_char(self.last_char)
        self.start_tag = start_tag
        self.end_tag = end_tag

    def token_with_char(self, char: str):
        """Due to tokenization weirdness, if the model wants to output a closing brace, it may use a token
        that contains the closing brace as a substring. This function returns all tokens that contain the
        given character, so that no matter what particular token it chooses, we can detect it.
        """
        tokens_containing_char = []
        for tok_id in self.tokenizer.vocab.values():
            str_tok = self.tokenizer.decode(tok_id)
            if char in str_tok:
                tokens_containing_char.append(tok_id)
        return tokens_containing_char

    def find_tool_call(self, text: str):
        """Takes the last character of the end tag, and checks to see if that character is part of the
        full end tag string. If so, then it will search backwards for the nearest opening tag. If it finds one,
        and the content in between is valid JSON, it will return the JSON object. Otherwise, it will return None.
        """
        end_pos = text.rfind(self.last_char)
        if (
            end_pos == -1
        ):  # if string doesn't contain closing brace, there are no tool calls.
            return None

        # check that the last closing brace is part of the end tag
        end_tag = text[end_pos - len(self.end_tag) + 1 : end_pos + 1]
        if end_tag != self.end_tag:
            return None

        # find the nearest opening tag, starting from the end
        start_pos = text.rfind(self.start_tag, 0, end_pos)
        if start_pos == -1:  # if no opening tag, then not a valid tool call
            return None

        # Extract the candidate substring
        candidate = text[
            start_pos + len(self.start_tag) : end_pos + 1 - len(self.end_tag)
        ]

        # Try parsing the candidate as JSON
        try:
            json_data = json.loads(candidate)

            return json_data  # Return the valid JSON block

        except json.JSONDecodeError:
            pass  # Invalid JSON block => not a valid tool call

        return None  # No valid JSON block found

    def __call__(self, ids, scores) -> bool:
        ids = ids[0]  # assume unbatched
        if (
            ids[-1] in self.tokens_with_end_tag
        ):  # check if the last token emitted contained the last end tag character
            decoded = self.tokenizer.decode(
                ids
            )  # if so, then a full check for a tool call is needed
            tool_call = self.find_tool_call(decoded)
            if tool_call is not None:
                return True
        return False


class HuggingFaceToolCaller:
    """Wraps a HuggingFace AutoModelForCausalLM and AutoTokenizer in a context that allows for tool results
    to be inserted mid-generation. The generate function will repeatedly call the underlying model.generate
    function until a tool call is no longer detected in the output. When a tool call is detected, the tool is
    invoked and the result is inserted into the input_ids, and generation continues from there.
    """

    def __init__(self, tokenizer, model, tools=None):
        self.tools = tools
        self.tool_map = get_structured_tools(tools)[0]
        self.stopper = ToolStopper(tokenizer)

        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

    def natively_supports_tools(self):
        return (
            isinstance(self.tokenizer.chat_template, dict)
            and "tool_use" in self.tokenizer.chat_template
        )

    def generate(self, input_ids, max_length, **kwargs):
        last_length = len(input_ids[0])
        while True:
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                stopping_criteria=[self.stopper],
                **kwargs,
            )
            generated_text = self.tokenizer.decode(output[0, last_length:])
            last_length = len(output[0])
            tool_call = self.stopper.find_tool_call(generated_text)
            if tool_call:
                tool_name = tool_call.get(
                    "name", "Must supply the `name` field for valid tool calls"
                )
                args = tool_call.get("args", {})
                try:
                    tool_result = self.tool_map.get(
                        tool_name, lambda: "Invalid tool name"
                    )(**args)
                except Exception as e:
                    tool_result = f"Error: {e}"
                # should add handling here to turn the args into the right types
                result = render_tool_call_result(tool_name, tool_result, args)
                result_ids = self.tokenizer.encode(
                    result, return_tensors="pt", add_special_tokens=False
                ).to(self.device)
                input_ids = torch.cat([output, result_ids], dim=-1)
                last_length = len(input_ids[0])
            else:
                return output


async def tool_use_loop_generate_async(
    messages: List[Dict[str, Any]],
    client: OpenAI,
    tool_map: Dict[str, callable],
    response_model: BaseModel = None
):
    while True:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tool_map.values(),
            tool_choice="auto"
        )
        
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                tool = tool_map[tool_call.function.name]
                result = tool(**tool_call.function.arguments)
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call.id
                })
        else:
            if response_model:
                return instructor.validate(response_model, response.choices[0].message.content)
            return response.choices[0].message.content
