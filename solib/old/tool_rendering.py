from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain.tools import StructuredTool, tool
from typing import Callable, Dict, List, Tuple, Union
from langchain_core.tools import render_text_description
from solib.utils.llm_utils import (
    jinja_env,
)

TOOL_CALL_TEMPLATE = jinja_env.get_template("tool_use/tool_call.jinja")
TOOL_RESULT_TEMPLATE = jinja_env.get_template("tool_use/tool_result.jinja")


def get_structured_tools(
    tools: List[Callable | StructuredTool],
) -> Tuple[Dict[str, Callable], List[StructuredTool]]:
    """Given a list of functions, return a dictionary mapping the function names to the functions, and a list of
    StructuredTool objects that wrap the functions. If the function is already a StructuredTool, it will be included
    in the list as is."""
    # why not use json representation of the tool?
    # https://python.langchain.com/docs/how_to/tools_prompting/ suggests using natural language
    tool_map = {tul.__name__: tul for tul in tools}
    structured_tools = [
        (t if isinstance(t, StructuredTool) else tool(t)) for t in tools
    ]
    return tool_map, structured_tools


def render_tools(tools: List[Union[Callable, StructuredTool]]) -> str:
    """Turn a list of StructuredTools into a string representation of the tools."""
    tool_text = render_text_description(get_structured_tools(tools)[1])
    return tool_text


def get_tool_prompt_for_nonnative(tools: List[Union[Callable, StructuredTool]]) -> str:
    """Get the tool prompt for the given tools.

    System message template that gets added when an llm has access to tools but
    doesn't natively support them."""

    rendered_tools = render_tools(tools)
    return jinja_env.get_template("tool_use/tool_prompt_nonnative.jinja").render(
        rendered_tools=rendered_tools,
        name="<function-name>",
        arguments="<args-dict>",
        result="<result>",
    )


def get_tool_prompt_for_native() -> str:
    """Get the tool prompt for a native tool-calling LLM.

    Will be added to system messages when tools are available."""

    return jinja_env.get_template("tool_use/tool_prompt.jinja").render()


def get_tool_prompt(
    tools: List[Union[Callable, StructuredTool]], is_native: bool
) -> str:
    """Get the tool prompt for the given tools."""
    if is_native:
        return get_tool_prompt_for_native()
    else:
        return get_tool_prompt_for_nonnative(tools)


def render_tool_call(name: str, args: Dict[str, str]) -> str:
    """Render a tool call with the given name and arguments. This is used to represent tool calls
    happening in API-based models, as these models typically will not give you the actual tokens
    that are generated. Also, this way we can make transcripts of API-based models and
    hugging face models more similar."""
    return TOOL_CALL_TEMPLATE.render(name=name, arguments=args)


def render_tool_call_result(name: str, result: str, args: Dict[str, str]) -> str:
    """Render a tool call result with the given name, result, and arguments. This is used to represent
    the outputs of tool calls in both API-based models and hugging face models."""
    return TOOL_RESULT_TEMPLATE.render(name=name, result=result, arguments=args)


def render_tool_call_conversation(response: List[BaseMessage]) -> str:
    """Given a list of BaseMessages, turn the conversation into one unified string representation that includes
    model responses, tool calls, and tool call results."""
    raw_response = ""
    tool_calls = {}  # map tool call ids to their associated messages
    for r in response:
        if isinstance(r, HumanMessage):  # ignore input prompt
            continue
        elif isinstance(r, AIMessage):
            if r.tool_calls:
                for t in r.tool_calls:
                    tool_calls[t["id"]] = t
                    raw_response += render_tool_call(t["name"], t["args"])
            else:
                # sometimes r.content can be an empty list, so this catches that??
                if r.content:
                    raw_response += r.content
        elif isinstance(r, ToolMessage):  # tool response
            tool_call = tool_calls[r.tool_call_id]
            raw_response += render_tool_call_result(
                tool_call["name"], r.content, tool_call["args"]
            )
    return raw_response
