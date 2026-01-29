import torch
import uuid

from dataclasses import dataclass, field
from collections import deque

@dataclass
class Config:
    num_head: int = 10
    head_dim: int = 128

    number_of_pages: int = 10


@dataclass
class Page:
    page_size: int = 10
    ref_count: int = 0
    hash: str = uuid.uuid4().hex


    def __repr__(self):
        return f"{self.hash} the page size is {self.page_size} and referenced {self.ref_count} times."

    def __post__init__(self):
        self.kv = torch.zeros(2, page_size, Config.num_head, Config.head_dim)

@dataclass
class PageTable:
    map: dict = field(default_factory = dict)

    def __repr__(self):
        return f"the page table has {self.map} entries"

    def map_page(self, logical_id: int, physical_id: int):
        """map the logical and physical id"""
        self.map[logical_id] = physical_id

#to-do    
class BlockManager:
    def __init__(self, num_pages: int, page_size: int):
        self.pages = [Page(page_size) for _ in range(num_pages)]
        self.allocated = deque()
        self.free = deque(self.pages) # at first, all the pages are free
    
    def allocate(self):
        if not self.free:
            return "all the pages are occupied"
        page_to_be_allocated = self.free.popleft()
        page_to_be_allocated.ref_count += 1
        self.allocated.append(page_to_be_allocated) # remove the left side page
    
    def deallocate(self, page_id):
        if page_id in self.allocated_pages:
            self.pages[page_id].ref_count -= 1
            if self.pages[page_id].ref_count == 0:
                self.allocated_pages.popleft(page_id)
                self.free.append(page_id)

if __name__ == '__main__':
    pg = Page(page_size = 2, ref_count = 0)
    print(pg)

    pg_table = PageTable()
    pg_table.map_page(0, 100)
    pg_table.map_page(1, 200)
    print(pg_table)

