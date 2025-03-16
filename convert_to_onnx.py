import os
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel

class CustomAttention(torch.nn.Module):
    def __init__(self, original_attention):
        super().__init__()
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.out_proj = original_attention.out_proj
        self.embed_dim = original_attention.embed_dim
        self.num_heads = original_attention.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if causal_attention_mask is not None:
            causal_attention_mask = causal_attention_mask.view(batch_size, 1, seq_len, seq_len)
            attention_scores = torch.where(causal_attention_mask == 0, float("-inf"), attention_scores)

        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_states)
        
        # Reshape output
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_dim * self.num_heads,)
        context_layer = context_layer.view(new_context_layer_shape)

        # Project output
        attention_output = self.out_proj(context_layer)
        
        if output_attentions:
            return attention_output, attention_probs
        return attention_output, None

def replace_attention_layers(model):
    for name, module in model.named_modules():
        if "self_attn" in name and hasattr(module, "q_proj"):
            parent_name = ".".join(name.split(".")[:-1])
            parent = model.get_submodule(parent_name)
            setattr(parent, name.split(".")[-1], CustomAttention(module))
    return model

def convert_clip_to_onnx():
    print("Loading CLIP vision model...")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    print("Replacing attention layers...")
    model = replace_attention_layers(model)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Converting model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        "models/clip_vision.onnx",
        input_names=["pixel_values"],
        output_names=["image_embeds", "last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"}
        },
        opset_version=12,
        do_constant_folding=True
    )
    print("Model conversion complete!")

if __name__ == "__main__":
    convert_clip_to_onnx() 