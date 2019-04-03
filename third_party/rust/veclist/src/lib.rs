use std::mem::replace;
use std::ops::{Index, IndexMut};

/// Node of the list
#[derive(Clone, Debug)]
enum Node<T> {
    Vacant(Option<usize>),
    Occupied(T),
}

impl<T> Node<T> {
    fn expect_vacant(&self) -> Option<usize> {
        match *self {
            Node::Vacant(next) => next,
            Node::Occupied(_) => panic!("Node is occupied"),
        }
    }

    fn expect_occupied(self) -> T {
        match self {
            Node::Vacant(_) => panic!("Node is vacant"),
            Node::Occupied(value) => value,
        }
    }

    fn free(&mut self, next: Option<usize>) -> Option<T> {
        match *self {
            Node::Vacant(_) => return None,
            _ => {}
        };
        Some(replace(self, Node::Vacant(next)).expect_occupied())
    }
}

/// `Vec` with slots which allow to `pop` values from index
/// which will be reused by later `push`.
#[derive(Clone, Debug)]
pub struct VecList<T> {
    // next free slot
    free: Option<usize>,
    // slots
    data: Vec<Node<T>>,
}

impl<T> Default for VecList<T> {
    fn default() -> Self {
        VecList::new()
    }
}

impl<T> VecList<T> {
    /// Create new empty `VecList`
    pub fn new() -> Self {
        VecList {
            free: None,
            data: Vec::new(),
        }
    }

    /// Create new `VecList` with specified capacity
    pub fn with_capacity(cap: usize) -> Self {
        VecList {
            free: None,
            data: Vec::with_capacity(cap),
        }
    }

    /// Push new value into `VecList` returning index
    /// where value is placed.
    pub fn push(&mut self, value: T) -> usize {
        if let Some(free) = self.free {
            debug_assert!(free < self.data.len());
            let old = replace(&mut self.data[free], Node::Occupied(value));
            replace(&mut self.free, old.expect_vacant()).unwrap()
        } else {
            // No free nodes available
            self.data.push(Node::Occupied(value));
            self.data.len() - 1
        }
    }

    /// Pop value from specified index.
    /// Returns `None` if index is unused.
    pub fn pop(&mut self, index: usize) -> Option<T> {
        if index > self.data.len() {
            None
        } else {
            self.data[index].free(self.free).map(|value| {
                self.free = Some(index);
                value
            })
        }
    }

    /// Returns a reference to the value of given index or `None` if there is no value yet.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index).and_then(|node| match *node {
            Node::Vacant(_) => None,
            Node::Occupied(ref value) => Some(value),
        })
    }

    /// Returns a mutable reference to the value of given index or `None` if there is no value yet.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index).and_then(|node| match *node {
            Node::Vacant(_) => None,
            Node::Occupied(ref mut value) => Some(value),
        })
    }

    /// Get upper bound (exclusive) of occupied incides
    pub fn upper_bound(&self) -> usize {
        self.data.len()
    }
}

impl<T> Index<usize> for VecList<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.get(index).expect("Expect occupied")
    }
}

impl<T> IndexMut<usize> for VecList<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).expect("Expect occupied")
    }
}

#[cfg(test)]
mod tests {
    use VecList;

    #[test]
    fn test_push() {
        let mut veclist = VecList::new();

        for i in 0..10 {
            veclist.push(i);
        }

        for i in 0..10 {
            assert_eq!(veclist.get(i), Some(&i));
        }
    }

    #[test]
    fn test_pop() {
        let mut veclist = VecList::new();

        for i in 0..10 {
            veclist.push(i);
        }

        for i in 0..5 {
            assert_eq!(veclist.pop(i), Some(i));
        }

        for i in 0..5 {
            assert_eq!(veclist.get(i), None);
        }

        for i in 6..10 {
            assert_eq!(veclist[i], i);
        }
    }

    #[test]
    fn test_reuse() {
        let mut veclist = VecList::new();

        for i in 0..10 {
            veclist.push(i);
        }

        for i in 0..5 {
            assert_eq!(veclist.pop(i), Some(i));
        }

        for i in 0..5 {
            veclist.push(i + 10);
        }

        for i in 0..5 {
            // reused in LIFO manner
            assert_eq!(veclist[i], 14 - i);
        }
    }
}
