require "./spec_helper"

private def linear_gelu
  x = Tensor.from(random_values(8, 1), [2, 4]).named("x")
  w = Tensor.from(random_values(12, 1), [4, 3]).named("w")
  b = Tensor.from(random_values(3, 1), [3]).named("b")
  {(x.matmul(w) + b).gelu.sum, b}
end

# Returns the output of three scoped blocks, `h + mlp(h)`, and their weights;
# the last block's weight is *last_width* wide.
private def scoped_blocks(last_width = 4)
  h = Tensor.from(random_values(8, 1), [2, 4]).named("x")
  weights = [4, 4, last_width].map { |width| Tensor.from(random_values(4 * width, 1), [4, width]) }
  weights.each do |weight|
    h = Tensomite.scope("block") do
      inner = Tensomite.scope("mlp") { h.matmul(weight).gelu }
      inner.shape == h.shape ? h + inner : inner
    end
  end
  {Tensomite.scope("head") { h.sum }, weights}
end

describe Tensomite::Scope do
  it "nests, names its path and restores the outer scope" do
    Tensomite::Scope.current.should be_nil
    Tensomite.scope("block") do
      Tensomite.scope("mlp") { Tensomite::Scope.current.to_s.should eq "block/mlp" }
      Tensomite::Scope.current.try(&.name).should eq "block"
    end
    Tensomite::Scope.current.should be_nil
  end

  it "restores the outer scope when the block raises" do
    expect_raises(Exception) { Tensomite.scope("block") { raise "inside" } }
    Tensomite::Scope.current.should be_nil
  end

  it "is recorded by the nodes built inside it" do
    a = Tensor.from(f32(1, 2), [2])
    b = Tensomite.scope("double") { a * 2 }
    b.node.scope.to_s.should eq "double"
    a.node.scope.should be_nil
  end

  it "is given to the gradient nodes of the nodes built inside it" do
    a = Tensor.from(f32(1, 2), [2])
    loss = Tensomite.scope("square") { (a * a) }.sum
    grad = Tensomite.grad(loss, [a]).first
    Diagram.graph_order(grad.node).compact_map(&.scope.try(&.name)).uniq!.should eq ["square"]
  end
end

describe "Tensor#to_mermaid" do
  it "draws every op, with views on the edges and leaves and root styled" do
    loss, _ = linear_gelu
    loss.to_mermaid.should eq <<-MERMAID
      flowchart TB
          n0["x · leaf [2,4]"]
          n1["w · leaf [4,3]"]
          n2["MatMul [2,3]"]
          n3["b · leaf [3]"]
          n4["Add [2,3]"]
          n5["Kernel gelu [2,3]"]
          n6["Sum []"]
          n0 --> n2
          n1 --> n2
          n2 --> n4
          n3 -- "expand [2,3]" --> n4
          n4 --> n5
          n5 --> n6
          classDef leaf stroke:#2b8a3e,stroke-width:2px
          class n0,n1,n3 leaf
          classDef root stroke:#c92a2a,stroke-width:2px
          class n6 root

      MERMAID
  end

  it "labels each kind of view, axes and non-F32 dtypes" do
    a = Tensor.from(random_values(6, 1), [2, 3])
    a.slice(1, 0, 2).sum(1).to_mermaid.should contain %(-- "slice [2,2]" -->)
    (a.reshape([3, 2]) * 2).to_mermaid.should contain %(-- "reshape [3,2]" -->)
    a.transpose.contiguous.to_mermaid.should contain %(-- "permute [3,2]" -->)
    a.sum(1).to_mermaid.should contain %(n1["Sum axis 1 [2]"])
    Tensor.from([1, 2], [2]).to_mermaid.should contain %(n0["leaf [2] I32"])
  end

  it "escapes quotes in names" do
    Tensor.from(f32(1), [1]).named(%(say "hi")).to_mermaid.should contain %(say #quot;hi#quot;)
  end

  it "draws the nodes nearest the root and one box for the rest" do
    loss, _ = linear_gelu
    text = loss.to_mermaid(max_nodes: 3)
    text.should contain %(more["… 4 more nodes"])
    text.should contain "more --> n0"
    text.lines.count(&.includes?("-->")).should eq 3
  end

  it "summarises scopes, folding matching siblings and counting params per instance" do
    loss, weights = scoped_blocks
    loss.to_mermaid(depth: 1, params: weights).should eq <<-MERMAID
      flowchart TB
          b0["x · leaf [2,4]"]
          b1["block ×3<br/>3 ops · 16 params each"]
          b2["head<br/>1 op"]
          b0 -- "[2,4]" --> b1
          b1 -. "next" .-> b1
          b1 -- "[2,4]" --> b2
          classDef leaf stroke:#2b8a3e,stroke-width:2px
          class b0 leaf
          classDef root stroke:#c92a2a,stroke-width:2px
          class b2 root

      MERMAID
  end

  it "draws outer scopes as subgraphs at greater depths" do
    loss, weights = scoped_blocks
    text = loss.to_mermaid(depth: 2, params: weights)
    text.should contain %(subgraph sub1 ["block ×3"])
    text.should contain %(["mlp<br/>2 ops · 16 params each"])
    text.should contain %(["own ops<br/>1 op each"])
  end

  it "keeps siblings apart when their graphs differ" do
    loss, weights = scoped_blocks(last_width: 5)
    text = loss.to_mermaid(depth: 1, params: weights)
    text.should_not contain "×"
    text.lines.count(&.includes?(%(["block<br/>))).should eq 3
  end

  it "gives the same text for the same graph" do
    loss, weights = scoped_blocks
    loss.to_mermaid(depth: 2, params: weights).should eq loss.to_mermaid(depth: 2, params: weights)
  end

  it "rejects a depth or node limit below 1" do
    loss, _ = linear_gelu
    expect_raises(ArgumentError) { loss.to_mermaid(depth: 0) }
    expect_raises(ArgumentError) { loss.to_mermaid(max_nodes: 0) }
  end
end
