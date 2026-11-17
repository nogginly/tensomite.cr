# Q-DD: Why does a shared cache need a new attention kernel?

*[Batched decoding](README.md#batched-decoding) · previous: [Q-DC](q-dc-fewer-wider-ops.md) · next: [Q-DE](q-de-one-dispatch-for-all-rows.md)*

## The question

With every row's keys and values in one shared buffer, why can't attention read it the way it reads one row's cache, without a new kernel?

> **Hint:** what does today's kernel assume is the same for every query?

<details>
<summary><strong>Answer</strong></summary>

### In short

Today's kernel takes one query block, one key range and one length for every query. The rows have different lengths and starts, so a ragged kernel must look up each token's row, the last position it may see, and its row's length from a table. Stage 2 (one shared cache, `Op::WriteRows`) took writes from 1,024 to 16.

### An analogy

A library with one shelf per reader against one shared shelf: the shared shelf needs an index saying whose books are where.

### Worked example

Stage 2, 64 rows:

Quantity   |Before |After  
-----------|-------|-------
Writes     |1,024  |16     
Step, Metal|18.4 ms|16.2 ms
Step, CUDA |12.5 ms|10.8 ms

```mermaid
flowchart LR
    s["Shared cache"] --> t["Table: row, last position, length per token"]
    t --> k["Ragged attention kernel"]
    style s stroke:#d9480f,stroke-width:3px
    style t stroke:#2b8a3e,stroke-width:3px
```

### Where it shows up in tensomite

`CausalSelfAttention::SharedCache`; Decision DV stage 2.

</details>

## See also

- [Q-DC](q-dc-fewer-wider-ops.md): stage 1.
- [Q-DE](q-de-one-dispatch-for-all-rows.md): the kernel that followed.
