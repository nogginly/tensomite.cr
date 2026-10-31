# Understanding FAQ

The questions labelled Q-A, Q-B and so on that came up while building tensomite, each in its own page with an answer, an analogy, a worked example and a diagram. They are grouped by theme rather than by the order they were asked.

**How to read a page.** Each opens with the question and a hint. The answer sits in a folded section: try the question first, then open it. Pages within a theme build on each other's examples, so reading a theme in order pays off.

```mermaid
flowchart LR
    graphs["Lazy graphs and views"] --> fusion["Fusion and kernels"]
    graphs --> autograd["Autograd"]
    memory["Memory and devices"] --> runtime["GPU runtime and lifetimes"]
    fusion --> order["Order of addition and determinism"]
    order --> gpuspeed["GPU speed"]
    cpuspeed["CPU speed"] --> gpuspeed
    autograd --> modules["Modules"]
    order --> decode["Decoding and the KV cache"]
    runtime --> gpuspeed
    style order stroke:#1971c2,stroke-width:3px
```

## Order of addition and determinism

One running example threads through this theme: three float32 numbers, `1`, `1e8` and `−1e8`, whose sum depends on the order they are added in.

Question                           |In one line                                                                                                                      
-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------
[Q-H](q-h-tile-order.md)           |Does the order of tiles matter to online softmax? Not to the maths, but to the last bits.                                        
[Q-AG](q-ag-fused-multiply-add.md) |Why does a more accurate fused multiply-add break bit-matching? Because the oracle rounds twice.                                 
[Q-S](q-s-fused-gelu.md)           |Why can a fused CPU `gelu` match bit for bit, but Metal's not? `tanh` comes from a different maths library.                      
[Q-I](q-i-exact-prefill-decode.md) |Why test prefill against decode exactly, not with a tolerance? Same device, same operations, same order: any difference is a bug.
[Q-R](q-r-split-depends-on-k.md)   |Why must split-K's split depend only on `k`? So that a row is summed the same way whatever the batch.                            
[Q-Z](q-z-modules-match-exactly.md)|Why should a refactor match exactly rather than roughly? Same operations, same order, same draws.                                
[Q-AR](q-ar-splitting-a-sum.md)    |Why not split one long sum across 32 threads? It regroups the additions, and the bits change.                                    
[Q-AT](q-at-chunk-sums.md)         |Why write every chunk sum rather than one per run? Adding a run first regroups the sum, as in Q-AR.                              

## Themes to come

Theme                    |Questions                              
-------------------------|---------------------------------------
Lazy graphs and views    |Q-A, Q-J, Q-K, Q-L, Q-M                
Memory and devices       |Q-B, Q-AE, Q-AF, Q-AJ                  
Fusion and kernels       |Q-D, Q-E, Q-F, Q-G, Q-T                
CPU speed                |Q-N, Q-O                               
Autograd                 |Q-Q, Q-W                               
Modules                  |Q-Y, Q-AA, Q-AB, Q-AC                  
GPU runtime and lifetimes|Q-U, Q-V, Q-AH, Q-AI, Q-AK             
GPU speed                |Q-X, Q-AL, Q-AM, Q-AN, Q-AP, Q-AQ, Q-AS
Decoding and the KV cache|Q-AD, Q-AO                             

Q-C and Q-P were labels on a decision and a list, not questions, so they have no page.
