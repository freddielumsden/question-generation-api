# https://colab.research.google.com/drive/1z-Zl2hftMrFXabYfmz8o9YZpgYx6sGeW?usp=sharing#scrollTo=p6H3FZjMM8e2
# https://huggingface.co/ThomasSimonini/t5-end2end-question-generation?text=generate+questions%3A+Python+is+an+interpreted%2C+high-level%2C+general-purpose+programming+language.+Created+by+Guido+van+Rossum+and+first+released+in+1991%2C+Python%27s+design+philosophy+emphasizes+code+readability+with+its+notable+use+of+significant+whitespace.+%3Cs%3E
from transformers import T5ForConditionalGeneration, T5TokenizerFast

hfmodel = T5ForConditionalGeneration.from_pretrained(
    'generate_question/t5-end2end-question-generation')

tokenizer = T5TokenizerFast.from_pretrained('t5-base', model_max_length=1000)
tokenizer.sep_token = '<sep>'
tokenizer.add_tokens(['<sep>'])


def hf_run_model(input_string, **generator_args):
    input_string = str(input_string)
    generator_args = {
        "max_length": 256,
        "num_beams": 4,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }
    input_string = "generate questions: " + input_string + " </s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = hfmodel.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    output = [item.split("<sep>") for item in output]
    # Formatting
    output = [None if i == "" else i for i in output[0]]
    return output


# Example
# text = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace."
# print(hf_run_model(text))
