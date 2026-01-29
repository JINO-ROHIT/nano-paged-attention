# nano-paged-attention

a minimal paged attention implementation to understand the main concepts of vllm engine.


why paged attention?

normal kv cache results in -
1. fragmented memory access - sequence lengths wont be equal all the time. frequent allocation and deallocations results in memory being allocated weirdly and leaves gaps in between, those gaps basically become unusable.
2. over allocation - when you pre-allocate a certain GB VRAM, theyre often underutilized and results in wastage.


paged attention borrrows concepts from OS and implements a paging based attention where you breaks the sequences into smaller pages and store them in blocks. during attention you iterate over all the pages and get the result.


### components

1. **page** - the page is the smallest unit in the physical memory. 
    - it is defined by `page_size` and stores a store portion of the kv cache.
    it also has a `ref_count` that tells you how many sequences are using this particular page. for example, the same system prompt will point to the same page always.

2. **page_table** - the page table keeps a mapping of the logical pages to the actual physical page in the GPU