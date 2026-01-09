from unsloth import FastLanguageModel
from dataset_helper import prepare_input


def inference(template, samples, model, tokenizer, batch_size=64):

    # Prepare model
    model = FastLanguageModel.for_inference(model)

    # Prepare inputs
    model_inputs = prepare_input(template, samples, tokenizer)

    #Generate
    outputs = [
        model.generate(
            model_inputs[i:i+batch_size],
            do_sample=True,
            temperature=0.1,
            top_k=50,
            top_p=0.1,
            repetition_penalty=1.05,
            max_new_tokens=512,
        )
        for i in range(model_inputs.shape[0]//batch_size)
    ]

    decoded_outputs = [
        [
            tokenizer(outputs[i][len(model_inputs[i]):], skip_special_tokens=True)
            for i in range(output.shape[0])
        ]
        for output in outputs
    ]
    decoded_outputs = sum(decoded_outputs, [])

    return decoded_outputs

if __name__ == '__main__':

    templ = "{prompt}"
    samplez = [{"prompt" : "Give me a short introduction to large language model."}]

    from model_helper import load_and_prepare_model

    m, t = load_and_prepare_model()

    responses = inference(templ, samplez, m, t)

    print(responses[:5])
