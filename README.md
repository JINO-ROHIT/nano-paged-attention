# nano-paged-attention

a minimal paged attention implementation to understand the main concepts of vllm engine.


why paged attention?

normal kv cache results in -
1. fragmented memory access - sequence lengths wont be equal all the time. frequent allocation and deallocations results in memory being allocated weirdly and leaves gaps in between, those gaps basically become unusable.
2. over allocation - when you pre-allocate a certain GB VRAM, theyre often underutilized and results in wastage.


paged attention borrrows concepts from OS and implements a paging based attention where you breaks the sequences into smaller pages and store them in blocks. during attention you iterate over all the pages and get the result.


### components

1. **page** - the page(often called as block) is the smallest allocation unit for the kv cache.
    - it stores the kv for a fixed number of tokens defined by the `page_size` (this is not bytes)
    - it also has a `ref_count` that tells you how many sequences are using this particular page. this enables two things -
        1. prefix sharing - a lot of the requests start with the same system prompt. it makes sense to not store duplicated kv cache for this each time. prefix sharing enables you to point to the same page if they share the same prefix tokens.
        2. decoding - for some of the decoding strategies like beam search etc requires starting from the same tokens but diverges as the generation moves forward in time. in this case, multiple sequences share the same initial kv pages but diverge in the future.
    - it lives in the physical GPU memory. 

2. **page_table** - the page table keeps a mapping of the logical pages to the actual physical page in the GPU. 
    - each request maintains its own page table.
    - the page table gives the illusion of the pages being contiguous in memory, because the logical pages seem ordered and continuous.

3. **sequence** - the sequence represents the user's decoding request. it has -
    - token ids
    - status (WAITING/RUNNING/FINISHED)
    - page table
    - current position

4. **block_manager** - the block manager is the whole heart which handles and maintains the allocation and deallocation of pages for every sequence.
    - handles allocation for the prefill stage
    - does incremental allocation for the decoding phase.
    - also does the reference counting for each page
    - frees the pages when the sequence is finished.