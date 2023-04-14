import os
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding
import json
from typing import List, Sized, Iterator
from rich.progress import Progress, TimeElapsedColumn, SpinnerColumn
import math
from multiprocessing import Pool
from functools import partial
import logging
from itertools import chain


def batch(iterable: Sized, n=1) -> Iterator:
    """
    Yield successive n-sized chunks from iterable.

    Args:
        iterable (`Sized`):
            The iterable to split.
        n (`int`, optional):
            The size of the chunks. Defaults to `1`.

    Yields:
        `Iterator`:
            An iterator with the chunks.
    """
    l: int = len(iterable)
    p: int = math.ceil(l / n)
    for ndx in range(0, l, p):
        yield iterable[ndx : min(ndx + p, l)]


def prepare_data(
    example: str,
    tokenizer: PreTrainedTokenizerBase,
    is_encoder_decoder: bool = False,
    max_length: int = 2048,
    inference: bool = False,
) -> BatchEncoding:
    """
    Prepare data for training or inference.

    Args:
        example (`str`):
            The example to prepare.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to use.
        is_encoder_decoder (`bool`, optional):
            Whether the model is an encoder-decoder model. Defaults to `False`.
        max_length (`int`, optional):
            The maximum length of the input. Defaults to `2048`.
        inference (`bool`, optional):
            Whether to prepare the data for inference. During inference labels
            are not included in model inputs. Defaults to `False`.

    Returns:
        `BatchEncoding`: `BatchEncoding` with the prepared data.
    """

    if is_encoder_decoder:
        prompt, result = example.split("result = [\n")
        prompt = prompt + "result = [\n"
        prompt = prompt.strip()
        result = result.strip()

        model_inputs = tokenizer(
            text=prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        if not inference:
            model_inputs["labels"] = tokenizer(
                text_target=result,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )["input_ids"]

    else:
        if inference:
            prompt = example.split("result = [\n")[0] + "result = [\n"
            model_inputs = tokenizer(
                text=prompt,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )

            # Remove the last token if it is an eos token
            if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
                model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
                model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]

        else:
            model_inputs = tokenizer(
                text=example,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )

            model_inputs["labels"] = model_inputs["input_ids"].copy()

    if "token_type_ids" in model_inputs:
        # LLaMa tokenizer adds token type ids, but we don't need them
        model_inputs.pop("token_type_ids")

    return model_inputs


def batch_tokenization(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    is_encoder_decoder: bool,
    max_length: int,
    inference: bool,
    examples: List[str],
    process_no: int,
) -> List[BatchEncoding]:
    """
    Batch tokenization function.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to use.
        dataset_name (`str`):
            The name of the dataset.
        is_encoder_decoder (`bool`):
            Whether the model is an encoder-decoder model.
        max_length (`int`):
            The maximum length of the input.
        inference (`bool`):
            Whether to prepare the data for inference. If model
            `is_encoder_decoder=False`, inputs ids will be truncated to don't include the
            results section of the example. Labels will still include the full correct
            example. If model `is_encoder_decoder=True`, this parameter is ignored.
        examples (`List[str]`):
            The examples to tokenize.
        process_no (`int`):
            The process number.

    Returns:
        `List[BatchEncoding]`:
            List of BatchEncoding with the prepared data.
    """
    tokenized_examples: List[BatchEncoding] = []
    if process_no == 0:
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"Tokenizing {dataset_name}", total=len(examples))

            for example in examples:
                tokenized_examples.append(
                    prepare_data(
                        example,
                        tokenizer,
                        is_encoder_decoder,
                        max_length,
                        inference,
                    )
                )
                progress.update(task, advance=1)
    else:
        tokenized_examples = [
            prepare_data(
                example,
                tokenizer,
                is_encoder_decoder,
                max_length,
                inference,
            )
            for example in examples
        ]

    return tokenized_examples


class CollieDataset(Dataset):
    """
    Dataset for Collie.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to use.
        dataset_path (`str`):
            The path to the jsonl file containing the dataset.
        is_encoder_decoder (`bool`, optional):
            Whether the model is an encoder-decoder model. Defaults to `False`.
        max_length (`int`, optional):
            The maximum length of the input. Defaults to `2048`.
        inference (`bool`, optional):
            Whether to prepare the data for inference. If model
            `is_encoder_decoder=False`, inputs ids will be truncated to don't include
            the results section of the example. Labels will still include the full
            correct example. If model `is_encoder_decoder=True`, this parameter is
            ignored. Defaults to `False`.
        num_workers (`int`, optional):
            The number of workers to use for tokenization. Defaults to
            `min(os.cpu_count(), 16)`.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str,
        is_encoder_decoder: bool = False,
        max_length: int = 2048,
        inference: bool = False,
        num_workers: int = min(os.cpu_count(), 16),
    ):
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

        with open(dataset_path, "r", encoding="utf8") as f:
            examples = f.readlines()

        examples = [json.loads(example.strip())["text"] for example in examples]

        if num_workers <= 1:
            self.tokenized_examples = batch_tokenization(
                tokenizer,
                self.dataset_name,
                is_encoder_decoder,
                max_length,
                inference,
                examples,
                0,
            )
        else:
            tokenizer_fn = partial(
                batch_tokenization,
                tokenizer,
                self.dataset_name,
                is_encoder_decoder,
                max_length,
                inference,
            )
            with Pool(num_workers) as p:
                self.tokenized_examples = p.starmap(
                    tokenizer_fn,
                    zip(batch(examples, num_workers), range(num_workers)),
                )
                self.tokenized_examples = list(
                    chain.from_iterable(self.tokenized_examples)
                )

        logging.info(
            f"Loaded {len(self.tokenized_examples)} examples from {self.dataset_name}"
        )

    def __len__(self) -> int:
        return len(self.tokenized_examples)

    def __getitem__(self, idx) -> List[BatchEncoding]:
        return self.tokenized_examples[idx].copy()