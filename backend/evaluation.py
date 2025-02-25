import csv
import logging
from functools import partial
from pathlib import Path
from typing import Annotated, Dict, List, Tuple, TypedDict

from langsmith import Client
from langsmith.schemas import Dataset

from backend.chat import resume_chat_workflow
from backend.configuration import Configuration, config_cache
from backend.logging_config import LOGGER_CONFIG
from backend.pdf_ingestion import create_or_load_chunks
from backend.vector_store import create_or_load_vector_store

config = Configuration()
config_cache()

logging.config.dictConfig(LOGGER_CONFIG)

logger = logging.getLogger("evaluation")
openai_client = config.llm

description_template = "Resume: `{}`, job descr: ```{}```"


def chat_with_resume(resume_file_path, job_description, query):
    messages = []
    chunks = create_or_load_chunks(resume_file_path)
    # Create a vector store from the resume chunks
    vector_store = create_or_load_vector_store(
        chunks=chunks,
        vector_index_name=resume_file_path.name.removesuffix(".pdf"),
        config=config,
    )
    # Chat with the resume
    answer_text = resume_chat_workflow(
        vector_store=vector_store,
        job_description=job_description,
        query=query,
        messages=messages,
    )

    return answer_text


def create_or_load_dataset(
    langsmith_client: Client, dataset_name: str, description: str = ""
) -> Dataset:
    # Check if dataset already exist
    if langsmith_client.has_dataset(dataset_name=dataset_name):
        logger.debug("Loading dataset from LangSmith")
        dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
    else:
        logger.info("Uploading dataset to LangSmith")
        dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name, description=description
        )
    return dataset


def load_test_cases_from_csv(
    csv_file: Path,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    inputs = []
    outputs = []
    with open(csv_file, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            inputs.append({"question": row["Input"]})
            outputs.append({"answer": row["Reference Output"]})
    logger.info(
        "Loaded %i inputs and %i outputs from %s",
        len(inputs),
        len(outputs),
        csv_file,
    )
    return inputs, outputs


def upload_test_cases(
    langsmith_client: Client,
    dataset_name: str,
    inputs: List,
    outputs: List,
) -> Dataset:
    if not langsmith_client.has_dataset(dataset_name=dataset_name):
        raise ValueError(f"{dataset_name}: Dataset does not exist")
    else:
        logger.debug("%s: Loading dataset from LangSmith", dataset_name)
        dataset = langsmith_client.read_dataset(dataset_name=dataset_name)

    # Add examples only if they do not already exist
    existing_examples = langsmith_client.list_examples(dataset_id=dataset.id)
    existing_inputs = [example.inputs for example in existing_examples]

    counter = 0
    for i, input in enumerate(inputs):
        if input not in existing_inputs:
            logger.info("Adding example number %d", counter)
            counter += 1
            output = outputs[i]
            langsmith_client.create_example(
                inputs=input,
                outputs=output,
                dataset_id=dataset.id,
            )
    logger.info("Added %d examples to the dataset", counter)
    return dataset


class CorrectnessGrade(TypedDict):
    """Grade output schema"""

    # NB: place explanation first to force the model to reason about the score
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for answer accuracy"""
    eval_instructions = "You are an expert recruiter specialized in matching jobs with resumes and answering questions about them."
    user_content = (
        f"QUESTION: {inputs['question']}\n"
        f"GROUND TRUTH ANSWER: {reference_outputs['answer']}\n"
        f"ANSWER: {outputs['response']}"
    )
    # Run evaluator
    grader_llm = config.llm.with_structured_output(
        CorrectnessGrade, method="json_schema", strict=True
    )
    evaluation = grader_llm.invoke(
        [
            {"role": "system", "content": eval_instructions},
            {"role": "user", "content": user_content},
        ]
    )
    return evaluation["correct"]  # type: ignore


def ls_target(inputs: Dict[str, str]) -> Dict[str, str]:
    """Map the dataset's input keys to the function we want to call, and
    map the output of the function to the expected output key"""
    part = partial(
        chat_with_resume,
        resume_file_path=config.cv_path,
        job_description=config.job_description,
    )
    return {"response": part(query=inputs["question"])}


class Color:
    BOLD = "\033[1m"
    END = "\033[0m"


def main():
    # Get the dataset name and dataset
    ls_client = Client()

    dataset_name = config.langsmith_dataset
    if config.langsmith_dataset == "":
        ds = list(ls_client.list_datasets())
        print(f"{Color.BOLD}Number of datasets: {len(ds)}{Color.END}\n")
        print(
            "\n".join(
                [
                    f"{Color.BOLD}* `{d.name}` containing {d.example_count} examples{Color.END}\n  "
                    f"Description: {repr(d.description[:200])}.."
                    for d in ds
                ]
            )
        )
        user_ds = input(
            "Type in the LangSmith dataset you would like to use or create. "
            "or 'q' to quit."
        ).strip()
        if user_ds.lower() == "q":
            return None
        elif user_ds in [d.name for d in ds]:
            dataset_name = user_ds
            dataset = create_or_load_dataset(
                langsmith_client=ls_client, dataset_name=dataset_name
            )
        else:
            # Create a new dataset
            user_start_experiments = input(
                f"{Color.BOLD}Creating a new dataset on LangSmith. Continue? [y/n]{Color.END}"
            )
            if not user_start_experiments.lower() == "y":
                print("Exiting")
                return
            else:
                logger.info("Uploading dataset to LangSmith")
                dataset = ls_client.create_dataset(
                    dataset_name=dataset_name,
                    description=description_template.format(
                        config.cv_path, config.job_description
                    ),
                )

    # Add test cases to dataset
    langsmith_tests_csv = config.langsmith_tests_csv
    if langsmith_tests_csv is None:
        user_csv = input(
            f"{Color.BOLD}Type an existing csv file if you want to upload test cases, "
            f"otherwise press enter to continue{Color.END}"
        )
        if user_csv != "" and Path(user_csv).is_file():
            langsmith_tests_csv = Path(user_csv)
    if langsmith_tests_csv is not None:
        if not langsmith_tests_csv.exists():
            raise ValueError(f"{langsmith_tests_csv}: csv file does not exist")
        inputs, outputs = load_test_cases_from_csv(langsmith_tests_csv)

        # Add any new experiment to the dataset
        dataset = upload_test_cases(
            langsmith_client=ls_client,
            dataset_name=dataset_name,
            inputs=inputs,
            outputs=outputs,
        )

    # Run experiments
    user_start_experiments = input(
        f"{Color.BOLD}Do you want to run {dataset.example_count} experiments "
        f"in the {dataset.name} dataset? [y/n]{Color.END}"
    )
    if not user_start_experiments.lower() == "y":
        print("Exiting")
        return
    logger.info(
        "Running %s experiments in the `%s` dataset",
        dataset.example_count,
        dataset.name,
    )
    experiment_results = ls_client.evaluate(
        ls_target,
        data=dataset_name,
        evaluators=[correctness],
        experiment_prefix="QA Example",
    )
    # Print the experiment results
    df = experiment_results.to_pandas()
    # Iterate over each row in the DataFrame
    fields = (
        "inputs.question",
        "outputs.response",
        "reference.answer",
        "feedback.correctness",
    )
    for index, row in df.loc[:, fields].iterrows():
        print(f"Index: {index}")
        print(f"Question: {row['inputs.question']}")
        print(f"Response: {row['outputs.response']}")
        print(f"Reference Answer: {row['reference.answer']}")
        print(f"Correctness: {row['feedback.correctness']}")
        print("-" * 40)  # Separator between rows


if __name__ == "__main__":
    main()
