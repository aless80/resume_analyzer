from functools import partial
from typing import Dict

from langsmith import Client

from backend.chat import resume_chat_workflow
from backend.configuration import Configuration, config_cache
from backend.pdf_ingestion import create_or_load_chunks
from backend.vector_store import create_or_load_vector_store

config = Configuration()
config_cache()

openai_client = config.llm

# Define dataset with test cases
inputs = [
    {"question": "What is the name of the candidate in the resume?"},
    {"question": "Does he know python? Answer with Yes or No"},
]
outputs = [
    {"answer": "Alessandro Marin"},
    {"answer": "Yes"},
]


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


def add_test_cases(
    langsmith_client,
    dataset_name: str,
    inputs: list,
    outputs: list,
    description: str = "",
):
    # Check if dataset already exist
    if langsmith_client.has_dataset(dataset_name=dataset_name):
        print("Loading dataset from LangSmith")
        dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
    else:
        print("Uploading dataset to LangSmith")
        dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name, description=description
        )

    # Add examples only if they do not already exist
    existing_examples = langsmith_client.list_examples(dataset_id=dataset.id)
    existing_inputs = [example.inputs for example in existing_examples]

    counter = 0
    for i, input in enumerate(inputs):
        if input not in existing_inputs:
            print(f"Adding example number: {counter}")
            counter += 1
            output = outputs[i]
            langsmith_client.create_example(
                inputs=input,
                outputs=output,
                dataset_id=dataset.id,
            )

    return dataset


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    eval_instructions = "You are an expert recruiter specialized in matching jobs with resumes and answering questions about them."
    user_content = (
        f"You are grading the following question:\n{inputs['question']})\n"
        f"Here is the real answer:\n{reference_outputs['answer']}\n"
        f"You are grading the following predicted answer:\n{outputs['response']}\n"
        "Respond with CORRECT or INCORRECT:\n"
    )
    evaluation = openai_client.invoke(
        input=[
            {"role": "system", "content": eval_instructions},
            {"role": "user", "content": user_content},
        ],
    ).content

    return evaluation == "CORRECT"


def ls_target(inputs: str) -> Dict[str, str]:
    """Map the dataset's input keys to the function we want to call, and
    map the output of the function to the expected output key"""
    part = partial(
        chat_with_resume,
        resume_file_path=config.cv_path,
        job_description=config.job_description,
    )
    return {"response": part(query=inputs["question"])}


def main():
    ls_client = Client()
    dataset_name = "QA Example"
    # Add any experiment to the dataset
    dataset = add_test_cases(
        langsmith_client=ls_client,
        dataset_name=dataset_name,
        inputs=inputs,
        outputs=outputs,
        description=f"Test dataset. Resume: `{config.cv_path}`, job descr: ```{config.job_description}```",
    )
    # Run experiments
    experiment_results = ls_client.evaluate(
        ls_target,
        data=dataset_name,
        evaluators=[correctness],
        experiment_prefix="QA Example",
    )
    print(experiment_results)


if __name__ == "__main__":
    main()
