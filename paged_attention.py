import uuid
import math

import torch
import torch.nn.functional as F

from dataclasses import dataclass, field
from collections import deque

from typing import Optional, List

@dataclass
class Config:
    num_head: int = 8
    head_dim: int = 64

    number_of_pages: int = 10
    page_size: int = 16


@dataclass
class Page:
    page_size: int = 10
    ref_count: int = 0
    hash: str = field(default_factory = lambda: uuid.uuid4().hex)

    def __repr__(self):
        return f"{self.hash} the page size is {self.page_size} and referenced {self.ref_count} times."

    def __post_init__(self):
        self.kv = torch.zeros(2, self.page_size, Config.num_head, Config.head_dim)
    
    def __hash__(self):
        return hash(self.hash)

@dataclass
class PageTable:
    map: dict = field(default_factory = dict)

    def __repr__(self):
        return f"the page table has {len(self.map)} entries"

    def map_page(self, logical_id: int, physical_page: Page):
        """map the logical page id to physical page"""
        self.map[logical_id] = physical_page
    
    def get_page(self, logical_id: int) -> Optional[Page]:
        """given logical id, fetch the physical page"""
        return self.map.get(logical_id)

@dataclass
class Sequence:
    """a sequence is a generation request"""
    def __init__(self, seq_id: int, prompt_tokens: List[int], page_size: int):
        self.seq_id = seq_id
        self.tokens = prompt_tokens.copy() # this is actually for (prompt + generated tokens)
        self.page_size = page_size
        self.page_table = PageTable() # each sequence maintains a page table
        self.logical_pages = []
    
    def __repr__(self):
        return f"sequence {self.seq_id} has {len(self.tokens)} tokens and {len(self.logical_pages)} pages"
    
    def get_num_tokens(self) -> int:
        return len(self.tokens)
    
    def get_num_pages_needed(self) -> int:
        return (len(self.tokens) + self.page_size -1 ) // self.page_size
    
    def append_token(self, token_id: int):
        self.tokens.append(token_id)


#to-do implement sequence allocation and deallocation
class BlockManager:
    def __init__(self, num_pages: int, page_size: int):
        self.pages = [Page(page_size) for _ in range(num_pages)]
        self.free = deque(self.pages) # at first, all the pages are free
        self.allocated = set()
    
    def _allocate(self) -> Optional[Page]:
        if not self.free:
            print("all the pages are occupied")
            return None
        
        page_to_be_allocated = self.free.popleft() # remove the left side page
        page_to_be_allocated.ref_count += 1
        self.allocated.add(page_to_be_allocated)
        return page_to_be_allocated
    
    def _deallocate(self, page: Page):
        if page not in self.allocated:
            return  
        
        page.ref_count -= 1
        
        if page.ref_count == 0:
            self.allocated.remove(page)
            self.free.append(page)
            page.kv.zero_() # clear the kv cache
    
    def allocate_for_sequence(self, sequence: Sequence) -> bool:
        num_pages_needed = sequence.get_num_pages_needed()
        current_pages = len(sequence.logical_pages)
        pages_to_allocate = num_pages_needed - current_pages

        if len(self.free) < pages_to_allocate:
            return False
        
        for i in range(pages_to_allocate):
            page = self._allocate()
            if page is None:
                return False
            
            logical_id = current_pages + i
            sequence.logical_pages.append(logical_id)
            sequence.page_table.map_page(logical_id, page)
        
        return True
    
    def free_sequence(self, sequence: Sequence):
        for logical_id in sequence.logical_pages:
            page = sequence.page_table.get_page(logical_id)
            if page:
                self._deallocate(page)
        
        sequence.logical_pages.clear()
        sequence.page_table.map.clear()
    
    def get_num_free_pages(self):
        return len(self.free)

class PagedAttention:
    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, query: torch.Tensor, sequence: Sequence):
        k_blocks = []
        v_blocks = []

        for logical_id in sequence.logical_pages:
            page = sequence.page_table.get_page(logical_id) # get the actual physical memory page
            k_blocks.append(page.kv[0])
            v_blocks.append(page.kv[1])
        
        keys = torch.cat(k_blocks, dim = 0) # becomes (tokens, num_head, head_dim)
        values = torch.cat(v_blocks, dim = 0)

        num_tokens = sequence.get_num_tokens()
        keys = keys[:num_tokens].transpose(0, 1)    # (num_head, tokens, head_dim)
        values = values[:num_tokens].transpose(0, 1)

        q = query.transpose(0, 1)
        #print(q.shape, keys.shape)
        attn_weights = torch.matmul(q, keys.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, values)
        
        return output.transpose(0, 1)


if __name__ == '__main__':
    config = Config(num_head = 8, head_dim = 64, page_size = 16)
    block_manager = BlockManager(num_pages = 10, page_size = config.page_size)
    
    print(f"\ninitialized block manager with {len(block_manager.pages)} pages")
    print(f"page size: {config.page_size} tokens per page")

    prompt_tokens = list(range(50))
    seq1 = Sequence(seq_id = 1, prompt_tokens = prompt_tokens, page_size = config.page_size)

    print(f"\n{seq1}")
    print(f"tokens: {seq1.get_num_tokens()}")
    print(f"pages needed: {seq1.get_num_pages_needed()}")

    print("\n====== prefill phase =========")
    success = block_manager.allocate_for_sequence(seq1)
    print(f"allocation successful: {success}")
    print(f"logical pages: {seq1.logical_pages}")
    print(f"page table: {seq1.page_table}")
    print(f"free pages remaining: {block_manager.get_num_free_pages()}")

    print("\n====== decode phase =========")
    for i in range(20):
        new_token = 100 + i
        seq1.append_token(new_token)

        pages_needed = seq1.get_num_pages_needed()
        if pages_needed > len(seq1.logical_pages):
            print(f"\ntoken {i+1}: need new page (total tokens: {seq1.get_num_tokens()})")
            success = block_manager.allocate_for_sequence(seq1)
            print(f"allocated page {seq1.logical_pages[-1]}")
        else:
            print(f"token {i+1}: using existing pages (total tokens: {seq1.get_num_tokens()})")
    
    print(f"\nfinal sequence state: {seq1}")
    print(f"free pages: {block_manager.get_num_free_pages()}")

    pa = PagedAttention(config.num_head, config.head_dim)
    
    mock_query = torch.randn(1, config.num_head, config.head_dim)
    for logical_id in seq1.logical_pages:
        page = seq1.page_table.get_page(logical_id)
        page.kv.uniform_(-1, 1) # simulate some KV values
        
    attn_output = pa.forward(mock_query, seq1)
    print(f"\npaged attention output: {attn_output.shape}")
    
    



