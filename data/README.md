# Data and embedding layout

Data is not included in the source release.

```text
data/
  train_prompts.txt
  embeddings/
    manifest.json
    index.jsonl
    selected_prompts.txt
    negative_prompt.safetensors
    embeddings-00000.safetensors
    ...
```

`selected_prompts.txt` and `index.jsonl` must be dense and have the count in
`manifest.json`. Each index record contains `index`, `shard`, `key`, and
`length`. Every referenced tensor is two-dimensional BF16 `[tokens, hidden]`.
The negative-prompt file contains one BF16 tensor named `prompt_embeds`.
