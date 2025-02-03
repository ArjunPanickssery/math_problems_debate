import functools
import logging
# from typing import TYPE_CHECKING
import torch
from solib.utils.globals import HF_TOKEN
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)

@functools.cache
def load_hf_model(model: str, hf_quantization_config=True):

    has_cuda = torch.cuda.is_available()
    LOGGER.info(f"Loading Hugging Face model (has_cuda={has_cuda})", model, hf_quantization_config)


    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = (
        BitsAndBytesConfig(load_in_8bit=True) 
        if has_cuda and hf_quantization_config 
        else None
    )

    model = model.split("localhf://")[1]
    tokenizer = AutoTokenizer.from_pretrained(model)
    device_map = "cuda" if has_cuda and hf_quantization_config else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map=device_map,
        token=HF_TOKEN,
        quantization_config=quant_config,
    )

    return (tokenizer, model)

def get_input_string(
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    tokenizer: AutoTokenizer | None = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    # tools: list[callable] = None,
    # natively_supports_tools: bool = False,
) -> str:
    """
    Args:
        prompt: str: Prompt to convert. Either prompt or messages must be provided
            to calculate input_string.
        messages: list[dict[str, str] | BaseMessage]: Messages to convert. Either prompt or
            messages must be provided to calculate input_string.
        tokenizer: AutoTokenizer: Tokenizer to use for the conversion.
        system_message: str | None: System message to add to the messages. Will be
            ignored if messages is provided.
        words_in_mouth: str | None: Words to append to the prompt. Will be ignored
            if input_string is provided.
        tools: list[callable]: If tool use is enabled, this will be a list of python functions.
        natively_supports_tools: bool: If True, the model natively supports tools and we can pass
            them in the `tools` parameter in apply_chat_template if tokenizer is supplied. If False,
            we will use the default tool prompt in tool_use.HuggingFaceToolCaller.TOOLS_PROMPT, and
            we will append it as the first system message to the messages.
    Returns:
        str: The input string to use for the model.
    """
    if input_string is None:
        if messages is None:
            assert prompt is not None

            messages = []
            if system_message is not None:
                messages.append({"role": "system", "content": system_message})
                
            messages.append({"role": "user", "content": prompt})

        # if tools:
        #     tool_msg = solib.tool_use.tool_rendering.get_tool_prompt(
        #         tools, natively_supports_tools
        #     )
        #     if msg_type == "langchain":
        #         messages.insert(0, SystemMessage(content=tool_msg))
        #     elif (
        #         not natively_supports_tools
        #     ):  # local models (where msg_type="dict") that natively support tools will have the tool prompt added by apply_chat_template
        #         messages.insert(  # otherwise, we manually create one
        #             0,
        #             {
        #                 "role": "system",
        #                 "content": tool_msg,
        #             },
        #         )

        if tokenizer is not None:
            # if tools and natively_supports_tools:
            #     input_string = tokenizer.apply_chat_template(
            #         messages, tokenize=False, tools=tools
            #     )
            # else:
            #     input_string = tokenizer.apply_chat_template(messages, tokenize=False)
            input_string = tokenizer.apply_chat_template(messages, tokenize=False)

            if words_in_mouth is not None:
                input_string += words_in_mouth

    return input_string

def get_hf_llm(model: str, hf_quantization_config=True):

    client = load_hf_model(model, hf_quantization_config)
    tokenizer, model = client

    # @costly(
    #     simulator=LLM_Simulator.simulate_llm_call,
    #     messages=lambda kwargs: format_prompt(
    #         prompt=kwargs.get("prompt"),
    #         messages=kwargs.get("messages"),
    #         input_string=kwargs.get("input_string"),
    #         system_message=kwargs.get("system_message"),
    #         msg_type=msg_type,
    #     )["messages"],
    #     disable_costly=DISABLE_COSTLY,
    # )
    def generate(
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        # tools: list[callable] = None,
        max_tokens: int = 2048,
        **kwargs,
    ):
        # natively_supports_tools = False
        # if tools:
        #     bmodel = HuggingFaceToolCaller(tokenizer, model, tools)
        #     natively_supports_tools = bmodel.natively_supports_tools()
        # else:
        #     bmodel = model
        bmodel = model

        input_string = get_input_string(
            prompt=prompt,
            messages=messages,
            input_string=input_string,
            tokenizer=tokenizer,
            system_message=system_message,
            words_in_mouth=words_in_mouth,
            # tools=tools,
            # natively_supports_tools=natively_supports_tools,
        )

        input_ids = tokenizer.encode(
            input_string,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(bmodel.device)

        input_length = input_ids.shape[1]
        output = bmodel.generate(
            input_ids,
            max_length=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )[0]

        decoded = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        return decoded

    # @costly(
    #     simulator=LLM_Simulator.simulate_llm_call,
    #     messages=lambda kwargs: format_prompt(
    #         prompt=kwargs.get("prompt"),
    #         messages=kwargs.get("messages"),
    #         input_string=kwargs.get("input_string"),
    #         system_message=kwargs.get("system_message"),
    #         msg_type=msg_type,
    #     )["messages"],
    #     disable_costly=DISABLE_COSTLY,
    # )
    def return_probs(
        return_probs_for: list[str],
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        **kwargs,
    ):
        input_string = get_input_string(  # don't pass tools to judges
            prompt=prompt,
            messages=messages,
            input_string=input_string,
            tokenizer=tokenizer,
            system_message=system_message,
            words_in_mouth=words_in_mouth,
            # msg_type=msg_type,
        )
        input_ids = tokenizer.encode(
            input_string, return_tensors="pt"
        ).to(model.device)
        output = model(input_ids).logits[0, -1, :]
        output_probs = output.softmax(dim=0)
        probs = {token: 0 for token in return_probs_for}
        for token in probs:
            # workaround for weird difference between word as a continuation vs standalone
            token_enc = tokenizer.encode(f"({token}", add_special_tokens=False)[-1]
            probs[token] = output_probs[token_enc].item()
        total_prob = sum(probs.values())
        try:
            probs_relative = {token: prob / total_prob for token, prob in probs.items()}
        except ZeroDivisionError:
            import pdb

            pdb.set_trace()
        return probs_relative

    return client, generate, return_probs