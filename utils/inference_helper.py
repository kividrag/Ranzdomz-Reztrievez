import math
import torch
from unsloth import FastLanguageModel
from utils.dataset_helper import prepare_input


def inference(template, samples, model, tokenizer, batch_size=64, gen_config=None):

    # Prepare model
    model = FastLanguageModel.for_inference(model)

    # Prepare inputs
    model_inputs = prepare_input(template, samples, tokenizer)

    # Configuration
    config = {
        "do_sample" : True,
        "temperature" : 0.1,
        "top_k" : 50,
        "top_p" : 0.1,
        "repetition_penalty" : 1.05,
        "max_new_tokens" : 512,
    }

    if gen_config is not None:
        config.update(gen_config)

    #Generate
    n_batches = math.ceil(len(samples) / batch_size)
    start_of_batch = [x*batch_size for x in range(n_batches)]
    with torch.no_grad():
        outputs = [
            model.generate(
                model_inputs[start: start+batch_size],
                **config,
            ).detach().cpu()
            for start in start_of_batch
        ]

    decoded_outputs = [
        [
            tokenizer.decode(output[i][len(model_inputs[j*batch_size+i]):], skip_special_tokens=True)
            for i in range(output.shape[0])
        ]
        for j, output in enumerate(outputs, start=0)
    ]
    decoded_outputs = sum(decoded_outputs, [])

    return decoded_outputs

if __name__ == '__main__':

    templ = "{prompt}"
    samplez = [{"prompt" : "Give me a short introduction to large language model."}]*128

    from model_helper import load_and_prepare_model

    m, t = load_and_prepare_model()

    responses = inference(templ, samplez, m, t)

    print(responses[:5])
