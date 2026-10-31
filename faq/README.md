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

## Lazy graphs and views

The running example: `a = [[1, 2, 3], [4, 5, 6]]` and a bias `b = [10, 20, 30]`, added as `a + b`.

Question                      |In one line                                                                                                  
------------------------------|-------------------------------------------------------------------------------------------------------------
[Q-A](q-a-lazy-errors.md)     |Why does an error in a lazy graph surface far from its cause? Building and running happen at different times.
[Q-K](q-k-stride-zero.md)     |What does a stride of 0 mean? Move nowhere: every row reads the same storage, so broadcasting copies nothing.
[Q-J](q-j-reduce-axis-last.md)|Why does `sum(axis)` move the summed axis to the end? So each output's inputs are one run of neighbours.     
[Q-L](q-l-cat-slice-views.md) |Why does `Cat` write through slice views? A slice is a view, so writing is just a copy through two views.    
[Q-M](q-m-max-start.md)       |Why does `max` start from the first element? Zero is not neutral for maximum; all-negative rows would give 0.

## Memory and devices

The running example: the bias `b = [10, 20, 30]` from the previous theme, followed onto each kind of device.

Question                                         |In one line                                                                                                                                 
-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------
[Q-B](q-b-unified-as-special-case.md)            |Why is unified memory a special case of discrete? Code that always copies works on both; code that reads directly breaks on discrete.       
[Q-AE](q-ae-mapped-memory-on-a-discrete-card.md) |Why can device-local, host-visible memory fail on a discrete card? On a card it is a small window, and allocations past it cannot be mapped.
[Q-AF](q-af-compute-and-graphics-capabilities.md)|Why does Vulkan need the `graphics` capability? Its driver is in the graphics library, which a container gets only from the host.           
[Q-AJ](q-aj-host-copy-after-upload.md)           |Why may a host copy stay valid after an upload? It equals the GPU's memory until something writes that memory.                              

## Fusion and kernels

Two running examples: GPT-2's `gelu` on a `[1024, 3072]` activation, and the softmax row `[1, 2, 3, 4]`.

Question                        |In one line                                                                                                           
--------------------------------|----------------------------------------------------------------------------------------------------------------------
[Q-D](q-d-spir-v-writer.md)     |Why would a SPIR-V writer need no compiler? It writes the bytecode itself; the driver always does the rest.           
[Q-E](q-e-matmul-fusion.md)     |Why is matmul not fused like elementwise ops? It reuses every input many times, so tiling rules; only its edges fuse. 
[Q-F](q-f-epilogue-fusion.md)   |Which ops ride along at the end of a matmul? Those that need only their own element; not softmax or layernorm.        
[Q-G](q-g-online-softmax.md)    |How can softmax work a tile at a time? Rescale the running sum by `exp(m − m')` when the maximum grows.               
[Q-T](q-t-softmax-two-passes.md)|Why can softmax's `exp` pass vectorise and keep the bits? The `exp`s are independent; only the sum must stay in order.

## CPU speed

The running example: a 512 × 512 matmul, measured one change at a time.

Question                                 |In one line                                                                                                                                            
-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------
[Q-N](q-n-where-blas-speed-comes-from.md)|Why does a naive matmul lose to BLAS by thousands of times? Several factors of 5 to 20, multiplied: bookkeeping, loop order, reuse, wider instructions.
[Q-O](q-o-which-loop-vectorises.md)      |Why does `c[j] += …` vectorise when `acc += …` cannot? Independent outputs can run side by side; one running sum cannot without regrouping.            

## Themes to come

Theme                    |Questions                              
-------------------------|---------------------------------------
Autograd                 |Q-Q, Q-W                               
Modules                  |Q-Y, Q-AA, Q-AB, Q-AC                  
GPU runtime and lifetimes|Q-U, Q-V, Q-AH, Q-AI, Q-AK             
GPU speed                |Q-X, Q-AL, Q-AM, Q-AN, Q-AP, Q-AQ, Q-AS
Decoding and the KV cache|Q-AD, Q-AO                             

Q-C and Q-P were labels on a decision and a list, not questions, so they have no page.
