import torch
import uuid

from dataclasses import dataclass, field
from collections import deque

from typing import Optional

@dataclass
class Config:
    num_head: int = 10
    head_dim: int = 128

    number_of_pages: int = 10
    page_size: int = 10


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
        return (self.tokens + self.page_size -1 ) // self.page_size
    
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
            return "all the pages are occupied"
        
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
