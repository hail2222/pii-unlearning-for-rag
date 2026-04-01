"""
Model loading and token-by-token generation with logits + hidden state capture.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME, DEVICE, MAX_NEW_TOKENS, PROBE_LAYER


class ModelProbe:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        probe_layer: int = PROBE_LAYER,
    ):
        self.device = device
        self.probe_layer = probe_layer

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect if this is an Instruct/Chat model (has chat template)
        self.is_instruct = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )

        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device)
        self.model.eval()

        total_layers = self.model.config.num_hidden_layers
        print(f"Model loaded. Layers: {total_layers}, probe layer: {probe_layer}")
        assert probe_layer < total_layers, (
            f"probe_layer={probe_layer} out of range for model with {total_layers} layers"
        )

    @torch.no_grad()
    def generate_with_probes(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> dict:
        """
        Run greedy generation and collect per-token logits and hidden states.

        Returns dict with:
          - "prompt":          original prompt string
          - "generated_text":  model output (excluding prompt)
          - "generated_ids":   list[int] of new token ids
          - "tokens":          list[str] decoded tokens
          - "logits":          list[Tensor(vocab_size)] one per step
          - "hidden_states":   list[Tensor(hidden_size)] at probe_layer, one per step
        """
        # For instruct models, wrap prompt in chat template
        if self.is_instruct:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        all_logits: list[torch.Tensor] = []
        all_hidden: list[torch.Tensor] = []
        generated_ids: list[int] = []

        # Manually decode token-by-token for fine-grained capture
        current_ids = inputs["input_ids"]

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_ids,
                output_hidden_states=True,
            )

            # logits for the last token position: shape (vocab_size,)
            step_logits = outputs.logits[0, -1, :]
            all_logits.append(step_logits.cpu())

            # hidden state at probe_layer for last token: shape (hidden_size,)
            # outputs.hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, hidden)
            step_hidden = outputs.hidden_states[self.probe_layer + 1][0, -1, :]
            all_hidden.append(step_hidden.cpu())

            # Greedy next token
            next_token_id = int(step_logits.argmax())
            generated_ids.append(next_token_id)

            # Stop at EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token_id]], device=self.device)],
                dim=1,
            )

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]

        return {
            "prompt":         prompt,
            "generated_text": generated_text,
            "generated_ids":  generated_ids,
            "tokens":         tokens,
            "logits":         all_logits,       # list of (vocab_size,) tensors
            "hidden_states":  all_hidden,        # list of (hidden_size,) tensors
        }


if __name__ == "__main__":
    probe = ModelProbe()
    result = probe.generate_with_probes("The capital of France is")
    print("Generated:", result["generated_text"])
    print("Tokens:", result["tokens"])
    print("Num steps:", len(result["logits"]))
    print("Hidden state shape:", result["hidden_states"][0].shape)
