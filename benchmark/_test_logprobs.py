"""测试 Qwen3.5-4B 能否加载并输出 logprobs。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/Qwen/Qwen3.5-4B"
print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    print("model loaded:", type(model).__name__)
    model.eval()
except Exception as e:
    print(f"FAILED to load as CausalLM: {e}")
    print("trying AutoModel...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    print("loaded as:", type(model).__name__)

# 测试一次前向传播，拿 logits
text = "用户问: 我爸提醒之下，我发现钢琴有些荒废了。回答: 这一刻..."
inputs = tok(text, return_tensors="pt").to(model.device)
print(f"\ninput tokens: {inputs['input_ids'].shape}")

with torch.no_grad():
    outputs = model(**inputs)

print(f"outputs type: {type(outputs).__name__}")
print(f"outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'no keys'}")
if hasattr(outputs, 'logits'):
    print(f"logits shape: {outputs.logits.shape}")
    # 算第 10 个 token 的 surprisal
    if outputs.logits.shape[1] > 10:
        logits = outputs.logits[0]  # [seq, vocab]
        target_id = inputs['input_ids'][0, 10]
        log_probs = torch.log_softmax(logits[9], dim=-1)  # 用位置9预测位置10
        surprisal = -log_probs[target_id].item()
        token_text = tok.decode([target_id.item()])
        print(f"token 10 = '{token_text}' (id={target_id.item()}), surprisal={surprisal:.4f}")
        print("✓ logprobs 可用！")
else:
    print("✗ 没有 logits 输出")
    print("outputs:", outputs)
