from datasets import load_dataset
from transformers import AutoTokenizer
import os
import argparse
import math

from itertools import chain
from functools import partial
from tqdm import tqdm


def parse_args():
    """
    Parse the command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse the arguments.")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/opt/ml/output/data/",
        help="The directory where the data is stored.",
    )

    parser.add_argument(
        "--input_data_dir",
        type=str,
        default="/opt/ml/input/data/train",
        help="The directory where the data is stored.",
    )

    parser.add_argument(
        "--job_name",
        type=str,
        default="test",
        help="The directory where the data is stored.",
    )

    parser.add_argument(
        "--split_range",
        type=str,
        default="",
        help="The range of data splits to use (e.g., '0-99,100-199').",
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=2,
        help="The number of processes to use for data processing.",
    )

    parser.add_argument(
        "--model_id", type=str, default="", help="The ID of the model to use."
    )

    args = parser.parse_known_args()
    return args



def create_batch(start_of_split, end_of_split, token, split_name):
    worker_split_batch=[]
    
    for split_number in range(int(start_of_split), int(end_of_split)):    
        file_name = split_name.replace(token, str(split_number).zfill(5))
        worker_split_batch.append(file_name)
    
    return worker_split_batch




def download_data(split_range, data_dir, parallel_proc, cache_dir_name):
    """
    Download a subset of the C4 dataset based on the provided arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        str: The name of the column containing the text data.
    """
    # Extract the split_range argument
    #list_of_splitfiles = (split_range).split(",")

    #print(f"list of files being processed:{list_of_splitfiles}")

    # Load the C4 dataset subset
    c4_subset = load_dataset(
        "allenai/c4", data_files=split_range, num_proc=parallel_proc, cache_dir=cache_dir_name, trust_remote_code=True
    )

    print(f"dataset printing:{c4_subset}")
    # Determine the text column name
    column_names = c4_subset["train"].column_names

    print(f"column names: {column_names}")

    text_column_name = "text" if "text" in column_names else column_names[0]

    return c4_subset, text_column_name


def group_texts(examples, block_size=2048):
    """
    Group the text in the provided examples into blocks of a specified size.

    Args:
        examples (dict): A dictionary of examples, where the values are lists of text.
        block_size (int, optional): The size of the text blocks. Defaults to 2048.

    Returns:
        dict: A dictionary containing the grouped text blocks and their labels.
    """
    # Concatenate the examples for each key
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    # Determine the total length of the concatenated examples
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Ensure the total length is a multiple of the block size
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split the concatenated examples into blocks
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # Set the labels to be the same as the input IDs
    result["labels"] = result["input_ids"].copy()

    return result


def preprocess_dataset(dataset, text_column_name, parallel_proc, model_id):
    """
    Preprocess the dataset by tokenizing the text and grouping it into blocks.

    Args:
        dataset (datasets.Dataset): The dataset to preprocess.
        text_column_name (str): The name of the column containing the text data.
        num_proc (int): The number of processes to use for preprocessing.

    Returns:
        datasets.Dataset: The preprocessed dataset.
    """

    print(f"repo id: {model_id}")
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f'cloumn names: {dataset["train"].column_names}')

    print(f'sample train row again:{dataset["train"][0]}')

    print(f"parallel_proc:{parallel_proc}")

    lm_dataset = dataset.map(
        lambda sample: tokenizer(sample[text_column_name]),
        batched=True,
        remove_columns=list(dataset["train"].column_names),
        desc="Running tokenizer on dataset",
        num_proc=parallel_proc
    ).map(partial(group_texts, block_size=2048), batched=True, num_proc=parallel_proc)

    return lm_dataset


def main():
    # Parse command-line arguments
    args, _ = parse_args()

    os.environ['HF_HOME'] = f"{args.input_data_dir}/{args.job_name}/tmp"
    os.environ['TRANSFORMERS_CACHE'] = f"{args.input_data_dir}/{args.job_name}/checkpoints"
    os.environ['HF_DATASETS_CACHE'] = f"{args.input_data_dir}/{args.job_name}/cache"

    token = "split_number"
    split_name = f"en/c4-train.{token}-of-01024.json.gz"

    start_of_split,end_of_split=(args.split_range).split(",")

    worker_split_batch = create_batch(
        start_of_split, end_of_split, token, split_name
    )
    
    print(f"split batch:{worker_split_batch}")

    # Download and process the data
    c4_subset, text_column_name = download_data(
        worker_split_batch, args.data_dir, args.num_proc, f"{args.input_data_dir}/{args.job_name}/data_cache"
    )

    print(f"text cloumn name identified: {text_column_name}")

    print(f'sample train rown:{c4_subset["train"][0]}')

    # Tokenize the dataset and group the text into blocks
    lm_dataset = preprocess_dataset(
        c4_subset, text_column_name, args.num_proc, args.model_id
    )

    print("Saving data to FSx now .... ")
    # Use the preprocessed dataset for further processing or training
    lm_dataset.save_to_disk(f'{args.input_data_dir}/{args.job_name}/')

    print(f"Saved data to: {args.input_data_dir}/{args.job_name}/")


if __name__ == "__main__":
    main()
