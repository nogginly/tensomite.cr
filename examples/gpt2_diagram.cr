# Prints a Mermaid flowchart of GPT-2's graph for an 8-token prompt: a summary of
# its scopes at DEPTH (default 2), or every op when DEPTH is 0.
#
#     scripts/fetch_gpt2.sh models/gpt2
#     crystal run examples/gpt2_diagram.cr -- models/gpt2/model.safetensors 2
require "../src/tensomite"

include Tensomite

path = ARGV[0]? || abort "usage: gpt2_diagram MODEL.safetensors [DEPTH]"
depth = (ARGV[1]? || "2").to_i
model = GPT2.load(path)
logits = model.graph([464, 2068, 7586, 21831, 18045, 625, 262, 16931])
puts depth > 0 ? logits.to_mermaid(depth: depth, params: model.params) : logits.to_mermaid(max_nodes: 400)
