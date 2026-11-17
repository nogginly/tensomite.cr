# Q-DC: Why do fewer, wider ops help when the host is idle?

*[Batched decoding](README.md#batched-decoding) · previous: [Q-DB](q-db-where-a-step-waits.md) · next: [Q-DD](q-dd-a-ragged-attention-kernel.md)*

## The question

Fewer, wider ops helped the M5 Pro, though its host was idle most of the step. Why?

> **Hint:** what runs one after another on the GPU?

<details>
<summary><strong>Answer</strong></summary>

### In short

Dispatches in a command buffer run one after another, and each with 8 threadgroups uses a sliver of a GPU of thousands of threads, so each one's latency adds up however idle the hardware. Packing the rotations took 1,024 ops to 16.

### An analogy

A single-lane road: cars are few, but each waits for the one in front.

### Worked example

Stage 1, 64 rows:

Quantity   |Before |After  
-----------|-------|-------
Step, Metal|22.8 ms|18.4 ms
Step, CUDA |17.5 ms|12.5 ms
Dispatches |3,180  |2,174  
Allocations|1,686  |681    

```mermaid
flowchart LR
    a["1,024 rotations"] --> b["16 packed"]
    b --> c["Fewer one-after-another latencies"]
    style a stroke:#d9480f,stroke-width:3px
    style b stroke:#2b8a3e,stroke-width:3px
```

### Where it shows up in tensomite

`RotaryEmbedding#angles`; Decision DV stage 1.

</details>

## See also

- [Q-DA](q-da-the-cost-of-each-row.md): where the per-row ops came from.
- [Q-DD](q-dd-a-ragged-attention-kernel.md): the next stage.
