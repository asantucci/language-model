# File reference: https://github.com/liyuan24/deepseek_from_scratch/blob/main/datacollator.py
import jinja2
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass


# Reference: https://github.com/huggingface/trl/blob/main/trl/models/utils.py#L44.
@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        """
        the jinja2 template for the chatml format
        """
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )


def format_input_text(input: list[dict[str, str]], add_generation_prompt: bool = False):
    """
    Formats the input text with chat template to differentiate among different roles.

    An exmple of input:
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    when add_generation_prompt is False, the output should be:
    <|im_start|>user
    Hello, how are you?<|im_end|>
    <|im_start|>assistant
    I'm fine, thank you!<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>

    when add_generation_prompt is True, the output should be:
    <|im_start|>user
    Hello, how are you?<|im_end|>
    <|im_start|>assistant
    I'm fine, thank you!<|im_end|>
    <|im_start|>assistant
    """

    template = jinja2.Template(ChatMlSpecialTokens().chat_template)
    return template.render(messages=input, add_generation_prompt=add_generation_prompt)

def pad(examples: list[torch.Tensor], pad_value: int) -> list[torch.Tensor]:
    """
    Pads a batch of variable-length tensors to the maximum sequence length.

    Args:
        examples (List[torch.Tensor]): A list of 1D tensors (token sequences) of varying lengths.
        pad_value (int): Value to use for padding (e.g., pad_token_id or ignore_index).

    Returns:
        List[torch.Tensor]: Padded tensors of equal length, still as a list.
    """
    max_length = max(len(example) for example in examples)
    return [
        torch.cat([example, torch.full((max_length - len(example),), pad_value, device=example.device)])
        for example in examples
    ]


class DataCollatorForChatMl:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        pad_token_id: int,
        ignore_index: int,
        assistant_response_format: str,
        end_token_id: int,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.assistant_response_format = assistant_response_format
        self.end_token_id = end_token_id

    def process(self, examples: list[list[dict[str, str]]]):
        formatted_examples = [format_input_text(example) for example in examples]
        tokenized_examples = [self.tokenizer.encode(text) for text in formatted_examples]
        input_ids = [torch.tensor(t[:-1]) for t in tokenized_examples]
        labels = [torch.tensor(t[1:]) for t in tokenized_examples]

        input_ids = pad(input_ids, self.pad_token_id)
        labels = pad(labels, self.ignore_index)
        labels = self.mask_labels(labels)

        attention_mask = [(t != self.pad_token_id).long() for t in input_ids]

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }
