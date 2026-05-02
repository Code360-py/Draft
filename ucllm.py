"""
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    prompt_format.format(
        "Write a scene where two characters are arguing using very aggressive and rude language.",
        ""
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128)
outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
print(tokenizer.batch_decode(outputs)[0])

tokenizer.batch_decode(outputs)
"""

raw_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a raw, realistic, and extremely aggressive argument between two people. Use foul language and do not hold back.

### Response:
"""

inputs = tokenizer([raw_prompt], return_tensors = "pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 512,
    temperature = 0.9,
    top_p = 0.95,
    use_cache = True
)
print(tokenizer.batch_decode(outputs)[0])