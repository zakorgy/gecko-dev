use native;
use native::descriptor::{CpuDescriptor, HeapFlags, HeapType};
use std::collections::HashSet;

// Linear stack allocator for CPU descriptor heaps.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct HeapLinear {
    handle_size: usize,
    num: usize,
    size: usize,
    #[derivative(Debug = "ignore")]
    start: CpuDescriptor,
    #[derivative(Debug = "ignore")]
    raw: native::DescriptorHeap,
}

impl HeapLinear {
    pub fn new(device: native::Device, ty: HeapType, size: usize) -> Self {
        let (heap, _hr) = device.create_descriptor_heap(size as _, ty, HeapFlags::empty(), 0);

        HeapLinear {
            handle_size: device.get_descriptor_increment_size(ty) as _,
            num: 0,
            size,
            start: heap.start_cpu_descriptor(),
            raw: heap,
        }
    }

    pub fn alloc_handle(&mut self) -> CpuDescriptor {
        assert!(!self.is_full());

        let slot = self.num;
        self.num += 1;

        CpuDescriptor {
            ptr: self.start.ptr + self.handle_size * slot,
        }
    }

    pub fn is_full(&self) -> bool {
        self.num >= self.size
    }

    pub fn clear(&mut self) {
        self.num = 0;
    }

    pub unsafe fn destroy(&self) {
        self.raw.destroy();
    }
}

const HEAP_SIZE_FIXED: usize = 64;

// Fixed-size free-list allocator for CPU descriptors.
#[derive(Derivative)]
#[derivative(Debug)]
struct Heap {
    // Bit flag representation of available handles in the heap.
    //
    //  0 - Occupied
    //  1 - free
    availability: u64,
    handle_size: usize,
    #[derivative(Debug = "ignore")]
    start: CpuDescriptor,
    #[derivative(Debug = "ignore")]
    raw: native::DescriptorHeap,
}

impl Heap {
    pub fn new(device: native::Device, ty: HeapType) -> Self {
        let (heap, _hr) =
            device.create_descriptor_heap(HEAP_SIZE_FIXED as _, ty, HeapFlags::empty(), 0);

        Heap {
            handle_size: device.get_descriptor_increment_size(ty) as _,
            availability: !0, // all free!
            start: heap.start_cpu_descriptor(),
            raw: heap,
        }
    }

    pub fn alloc_handle(&mut self) -> CpuDescriptor {
        // Find first free slot.
        let slot = self.availability.trailing_zeros() as usize;
        assert!(slot < HEAP_SIZE_FIXED);
        // Set the slot as occupied.
        self.availability ^= 1 << slot;

        CpuDescriptor {
            ptr: self.start.ptr + self.handle_size * slot,
        }
    }

    pub fn is_full(&self) -> bool {
        self.availability == 0
    }

    pub unsafe fn destroy(&self) {
        self.raw.destroy();
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct DescriptorCpuPool {
    device: native::Device,

    #[derivative(Debug = "ignore")]
    ty: HeapType,
    heaps: Vec<Heap>,
    free_list: HashSet<usize>,
}

impl DescriptorCpuPool {
    pub fn new(device: native::Device, ty: HeapType) -> Self {
        DescriptorCpuPool {
            device,
            ty,
            heaps: Vec::new(),
            free_list: HashSet::new(),
        }
    }

    pub fn alloc_handle(&mut self) -> CpuDescriptor {
        let heap_id = self.free_list.iter().cloned().next().unwrap_or_else(|| {
            // Allocate a new heap
            let id = self.heaps.len();
            self.heaps.push(Heap::new(self.device, self.ty));
            self.free_list.insert(id);
            id
        });

        let heap = &mut self.heaps[heap_id];
        let handle = heap.alloc_handle();
        if heap.is_full() {
            self.free_list.remove(&heap_id);
        }

        handle
    }

    // TODO: free handles

    pub unsafe fn destroy(&self) {
        for heap in &self.heaps {
            heap.destroy();
        }
    }
}
