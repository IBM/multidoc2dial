import argparse
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, AutoTokenizer

# https://huggingface.co/facebook/rag-token-base


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
    )

    parser.add_argument(
        "--out_path",
        type=str,
    )

    parser.add_argument(
        "--index_name",
        type=str,
        default="exact",
    )

    args = parser.parse_args()

    model = RagTokenForGeneration.from_pretrained_question_encoder_generator(args.model_path, "facebook/bart-large")
    model.config.use_dummy_dataset = True
    model.config.index_name = args.index_name

    question_encoder_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    tokenizer = RagTokenizer(question_encoder_tokenizer, generator_tokenizer)
    retriever = RagRetriever(model.config, question_encoder_tokenizer, generator_tokenizer)

    model.save_pretrained(args.out_path)
    tokenizer.save_pretrained(args.out_path)
    retriever.save_pretrained(args.out_path)


if __name__ == "__main__":
    main()