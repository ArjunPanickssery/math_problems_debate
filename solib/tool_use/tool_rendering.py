from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


from langchain.tools import StructuredTool, tool


from typing import Callable, Dict, List, Tuple, Union


from langchain_core.tools import render_text_description

TOOL_CALL_START_TAG = "<tool_call>"
TOOL_CALL_END_TAG = "</tool_call>"
TOOL_RESULT_START_TAG = "<tool_result>"
TOOL_RESULT_END_TAG = "</tool_result>"

TOOL_CALL_TEMPLATE = """
{call_start_tag}
{{
    "name": {name},
    "args": {arguments}
}}
{call_end_tag}
"""

TOOL_RESULT_TEMPLATE = """
{result_start_tag}
{{
    "name": {name},
    "result": {result},
    "args": {arguments}
}}
{result_end_tag}
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = """
You are a tool calling LLM that has access to the following set of tools.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query.
Don\'t make assumptions about what values to plug into functions. Here are the available tools:

<tools>
{rendered_tools}
</tools>

Every tool call should be surrounded by {start_call_tag}{end_call_tag} tags, and should take the form of a JSON object with the following format:
{call_template}

The result will then be automatically supplied in the form of a JSON object with the following format:
{result_template}
"""


def get_structured_tools(tools: List[Callable | StructuredTool]) -> Tuple[Dict[str, Callable], List[StructuredTool]]:
    """Given a list of functions, return a dictionary mapping the function names to the functions, and a list of
    StructuredTool objects that wrap the functions. If the function is already a StructuredTool, it will be included
    in the list as is."""
    tool_map = {tool.__name__: tool for tool in tools}
    structured_tools = [(t if isinstance(t, StructuredTool) else tool(t)) for t in tools]
    return tool_map, structured_tools


def render_tools(tools: List[Union[Callable, StructuredTool]]) -> str:
    """Turn a list of StructuredTools into a string representation of the tools."""
    tool_text = render_text_description(get_structured_tools(tools)[1])
    return tool_text


def get_tool_prompt(tools: List[Union[Callable, StructuredTool]]) -> str:
    """Get the tool prompt for the given tools."""
    # prompt format based off of https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/tree/main
    rendered_tools = render_tools(tools)
    return DEFAULT_TOOL_PROMPT_TEMPLATE.format(
        start_call_tag=TOOL_CALL_START_TAG,
        end_call_tag=TOOL_CALL_END_TAG,

        call_template=TOOL_CALL_TEMPLATE.format(
            call_start_tag=TOOL_CALL_START_TAG,
            call_end_tag=TOOL_CALL_END_TAG,
            name="<function-name>",
            arguments="<args-dict>"
        ),

        result_template=TOOL_RESULT_TEMPLATE.format(
            result_start_tag=TOOL_RESULT_START_TAG,
            result_end_tag=TOOL_RESULT_END_TAG,
            name="<function-name>",
            result="<result>",
            arguments="<args-dict>"),

        rendered_tools=rendered_tools)

def render_tool_call(name: str, args: Dict[str, str]) -> str:
    """Render a tool call with the given name and arguments. This is used to represent tool calls
    happening in API-based models, as these models typically will not give you the actual tokens
    that are generated. Also, this way we can make transcripts of API-based models and
    hugging face models more similar."""
    return TOOL_CALL_TEMPLATE.format(name=name, arguments=args,
                                     call_start_tag=TOOL_CALL_START_TAG,
                                     call_end_tag=TOOL_CALL_END_TAG)


def render_tool_call_result(name: str, result: str, args: Dict[str, str]) -> str:
    """Render a tool call result with the given name, result, and arguments. This is used to represent
    the outputs of tool calls in both API-based models and hugging face models."""
    return TOOL_RESULT_TEMPLATE.format(name=name, result=result, arguments=args,
                                       result_start_tag=TOOL_RESULT_START_TAG,
                                       result_end_tag=TOOL_RESULT_END_TAG)


def render_tool_call_conversation(response: List[BaseMessage]) -> str:
    """Given a list of BaseMessages, turn the conversation into one unified string representation that includes
    model responses, tool calls, and tool call results."""
    raw_response = ""
    tool_calls = {}   # map tool call ids to their associated messages
    for r in response:
        if isinstance(r, HumanMessage):  # ignore input prompt
            continue
        elif isinstance(r, AIMessage):
            if r.tool_calls:
                for t in r.tool_calls:
                    tool_calls[t['id']] = t
                    raw_response += render_tool_call(t['name'], t['args'])
            else:
                raw_response += r.content
        elif isinstance(r, ToolMessage):  # tool response
            tool_call = tool_calls[r.tool_call_id]
            raw_response += render_tool_call_result(tool_call['name'], r.content, tool_call['args'])
    return raw_response