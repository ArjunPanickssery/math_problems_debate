from typing import Callable, Dict, List, Any
import inspect
from pydantic import create_model, BaseModel
from solib.globals import (
    jinja_env,
)

TOOL_CALL_TEMPLATE = jinja_env.get_template("tool_use/tool_call.jinja")
TOOL_RESULT_TEMPLATE = jinja_env.get_template("tool_use/tool_result.jinja")


def get_structured_tools(
    tools: List[Callable],
) -> tuple[Dict[str, Callable], List[Dict[str, Any]]]:
    """Given a list of functions, return a dictionary mapping the function names to the functions, and a list of
    OpenAI function definitions that describe the functions."""
    # why not use json representation of the tool?
    # https://python.langchain.com/docs/how_to/tools_prompting/ suggests using natural language
    tool_map = {tul.__name__: tul for tul in tools}
    
    structured_tools = []
    for func in tools:
        sig = inspect.signature(func)
        params = {
            name: (param.annotation, ... if param.default == param.empty else param.default)
            for name, param in sig.parameters.items()
        }
        
        # Create a Pydantic model for the function parameters
        model = create_model(f"{func.__name__}Parameters", **params)
        
        tool_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": model.model_json_schema()
            }
        }
        structured_tools.append(tool_schema)
    
    return tool_map, structured_tools


def render_tools(tools: List[Callable]) -> str:
    """Turn a list of functions into a string representation of the tools."""
    tool_text = ""
    for tool in tools:
        tool_text += f"\n{tool.__name__}:\n"
        tool_text += f"  Description: {tool.__doc__ or ''}\n"
        sig = inspect.signature(tool)
        tool_text += "  Parameters:\n"
        for name, param in sig.parameters.items():
            tool_text += f"    - {name}: {param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)}\n"
    return tool_text


def get_tool_prompt_for_nonnative(tools: List[Callable]) -> str:
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
    tools: List[Callable], is_native: bool
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


def render_tool_call_conversation(response: List[Dict[str, Any]]) -> str:
    """Given a list of message dicts, turn the conversation into one unified string representation that includes
    model responses, tool calls, and tool call results."""
    raw_response = ""
    tool_calls = {}  # map tool call ids to their associated messages
    for r in response:
        if r["role"] == "user":  # ignore input prompt
            continue
        elif r["role"] == "assistant":
            if "tool_calls" in r:
                for t in r["tool_calls"]:
                    tool_calls[t["id"]] = t
                    raw_response += render_tool_call(t["name"], t["args"])
            else:
                # sometimes r.content can be an empty list, so this catches that??
                if r["content"]:
                    raw_response += r["content"]
        elif r["role"] == "tool":  # tool response
            tool_call = tool_calls[r["tool_call_id"]]
            raw_response += render_tool_call_result(
                tool_call["name"],
                r["content"],
                tool_call["args"]
            )
    return raw_response
