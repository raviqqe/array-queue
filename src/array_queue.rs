use crate::error::CapacityError;
use core::mem::{MaybeUninit, replace, transmute};

/// A queue backed by a fixed-size array.
#[derive(Debug)]
pub struct ArrayQueue<T, const N: usize> {
    array: [MaybeUninit<T>; N],
    start: usize,
    length: usize,
}

impl<T, const N: usize> ArrayQueue<T, N> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            array: [const { MaybeUninit::uninit() }; N],
            start: 0,
            length: 0,
        }
    }

    /// Returns a reference to the first element of the queue, or `None` if it is empty.
    pub fn first(&self) -> Option<&T> {
        self.element(0)
    }

    /// Returns a mutable reference to the first element of the queue, or `None` if it is empty.
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.element_mut(0)
    }

    /// Returns a reference to the last element of the queue, or `None` if it is empty.
    pub fn last(&self) -> Option<&T> {
        self.element(N + self.length - 1)
    }

    /// Returns a mutable reference to the last element of the queue, or `None` if it is empty.
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.element_mut(N + self.length - 1)
    }

    fn element(&self, index: usize) -> Option<&T> {
        if index < self.length {
            let x = &self.array[self.index(index)];

            Some(unsafe { x.assume_init_ref() })
        } else {
            None
        }
    }

    fn element_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.length {
            let x = &mut self.array[self.index(index)];

            Some(unsafe { x.assume_init_mut() })
        } else {
            None
        }
    }

    /// Pushes an element to the back of the queue.
    pub fn push_back(&mut self, x: T) -> Result<(), CapacityError> {
        if self.is_full() {
            return Err(CapacityError);
        }

        self.array[self.index(self.length)] = MaybeUninit::new(x);
        self.length += 1;

        Ok(())
    }

    /// Pushes an element to the front of the queue.
    pub fn push_front(&mut self, x: T) -> Result<(), CapacityError> {
        if self.is_full() {
            return Err(CapacityError);
        }

        self.start = self.index(N - 1);
        self.array[self.start] = MaybeUninit::new(x);
        self.length += 1;

        Ok(())
    }

    /// Pops an element from the back of the queue.
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let x = replace(&mut self.array[self.length - 1], MaybeUninit::uninit());
            self.length -= 1;

            Some(unsafe { x.assume_init() })
        }
    }

    /// Pops an element from the front of the queue.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let x = replace(&mut self.array[self.start], MaybeUninit::uninit());
            self.start = self.index(1);
            self.length -= 1;

            Some(unsafe { x.assume_init() })
        }
    }

    /// Returns the number of elements in the queue.
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns `true` if the queue is empty.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the queue is full.
    pub fn is_full(&self) -> bool {
        self.len() == N
    }

    fn index(&self, index: usize) -> usize {
        (self.start + index) % N
    }
}

impl<T: Clone, const N: usize> Clone for ArrayQueue<T, N> {
    fn clone(&self) -> Self {
        let mut queue = Self::new();

        for x in self {
            queue.push_back(x.clone()).unwrap();
        }

        queue
    }
}

impl<T, const N: usize> Default for ArrayQueue<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for ArrayQueue<T, N> {
    fn drop(&mut self) {
        for _ in self {
            // TODO
        }
    }
}

#[derive(Debug)]
pub struct ArrayQueueIterator<'a, T, const N: usize> {
    queue: &'a ArrayQueue<T, N>,
    first: usize,
    last: usize,
}

impl<'a, T, const N: usize> ArrayQueueIterator<'a, T, N> {
    const fn is_exhausted(&self) -> bool {
        self.first >= self.last
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a ArrayQueue<T, N> {
    type Item = &'a T;
    type IntoIter = ArrayQueueIterator<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        ArrayQueueIterator {
            queue: self,
            first: 0,
            last: self.len(),
        }
    }
}

impl<'a, T, const N: usize> Iterator for ArrayQueueIterator<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_exhausted() {
            return None;
        }

        let x = self.queue.element(self.first);
        self.first += x.is_some() as usize;
        x
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for ArrayQueueIterator<'a, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.is_exhausted() {
            return None;
        }

        let x = self.queue.element(self.last - 1);
        self.last -= x.is_some() as usize;
        x
    }
}

#[derive(Debug)]
pub struct ArrayQueueMutIterator<'a, T, const N: usize> {
    queue: &'a mut ArrayQueue<T, N>,
    first: usize,
    last: usize,
}

impl<'a, T, const N: usize> ArrayQueueMutIterator<'a, T, N> {
    const fn is_exhausted(&self) -> bool {
        self.first >= self.last
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut ArrayQueue<T, N> {
    type Item = &'a mut T;
    type IntoIter = ArrayQueueMutIterator<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let last = self.len();

        ArrayQueueMutIterator {
            queue: self,
            first: 0,
            last,
        }
    }
}

impl<'a, T, const N: usize> Iterator for ArrayQueueMutIterator<'a, T, N> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_exhausted() {
            return None;
        }

        let x = self.queue.element_mut(self.first);
        self.first += x.is_some() as usize;
        x.map(|x| unsafe { transmute(x) })
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for ArrayQueueMutIterator<'a, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.is_exhausted() {
            return None;
        }

        let x = self.queue.element_mut(self.last - 1);
        self.last -= x.is_some() as usize;
        x.map(|x| unsafe { transmute(x) })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use alloc::boxed::Box;

    #[test]
    fn new() {
        ArrayQueue::<usize, 1>::new();
        ArrayQueue::<usize, 2>::new();
    }

    #[test]
    fn first_and_last() {
        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert_eq!(a.first(), None);
        assert_eq!(a.first_mut(), None);
        assert_eq!(a.last(), None);
        assert_eq!(a.last_mut(), None);

        assert!(a.push_back(1).is_ok());

        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.first_mut(), Some(&mut 1));
        assert_eq!(a.last(), Some(&1));
        assert_eq!(a.last_mut(), Some(&mut 1));

        assert!(a.push_back(2).is_ok());

        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.first_mut(), Some(&mut 1));
        assert_eq!(a.last(), Some(&2));
        assert_eq!(a.last_mut(), Some(&mut 2));
    }

    #[test]
    fn push_back() {
        let mut a: ArrayQueue<usize, 1> = ArrayQueue::new();

        assert_eq!(a.len(), 0);
        assert!(a.push_back(42).is_ok());
        assert_eq!(a.len(), 1);
        assert_eq!(a.push_back(42), Err(CapacityError));
        assert_eq!(a.len(), 1);

        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert_eq!(a.len(), 0);
        assert!(a.push_back(42).is_ok());
        assert_eq!(a.len(), 1);
        assert!(a.push_back(42).is_ok());
        assert_eq!(a.len(), 2);
        assert_eq!(a.push_back(42), Err(CapacityError));
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn push_front() {
        let mut a: ArrayQueue<usize, 1> = ArrayQueue::new();

        assert_eq!(a.len(), 0);
        assert!(a.push_front(42).is_ok());
        assert_eq!(a.len(), 1);
        assert_eq!(a.push_front(42), Err(CapacityError));
        assert_eq!(a.len(), 1);

        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert_eq!(a.len(), 0);
        assert!(a.push_front(1).is_ok());
        assert_eq!(a.first(), Some(&1));
        assert_eq!(a.last(), Some(&1));
        assert_eq!(a.len(), 1);
        assert!(a.push_front(2).is_ok());
        assert_eq!(a.first(), Some(&2));
        assert_eq!(a.last(), Some(&1));
        assert_eq!(a.len(), 2);
        assert_eq!(a.push_front(3), Err(CapacityError));
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn pop_back() {
        let mut a: ArrayQueue<usize, 1> = ArrayQueue::new();

        assert!(a.push_back(42).is_ok());

        assert_eq!(a.pop_back(), Some(42));
        assert_eq!(a.len(), 0);

        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert!(a.push_back(123).is_ok());
        assert!(a.push_back(42).is_ok());

        assert_eq!(a.pop_back(), Some(42));
        assert_eq!(a.first(), Some(&123));
        assert_eq!(a.last(), Some(&123));
        assert_eq!(a.len(), 1);
        assert_eq!(a.pop_back(), Some(123));
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn pop_front() {
        let mut a: ArrayQueue<usize, 1> = ArrayQueue::new();

        assert!(a.push_back(42).is_ok());

        assert_eq!(a.pop_front(), Some(42));
        assert_eq!(a.len(), 0);

        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert!(a.push_back(123).is_ok());
        assert!(a.push_back(42).is_ok());

        assert_eq!(a.pop_front(), Some(123));
        assert_eq!(a.first(), Some(&42));
        assert_eq!(a.last(), Some(&42));
        assert_eq!(a.len(), 1);
        assert_eq!(a.pop_front(), Some(42));
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn push_and_pop_across_edges() {
        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert!(a.push_back(1).is_ok());
        assert!(a.push_back(2).is_ok());

        for i in 3..64 {
            assert_eq!(a.pop_front(), Some(i - 2));
            assert_eq!(a.len(), 1);
            assert!(a.push_back(i).is_ok());
            assert_eq!(a.len(), 2);
        }
    }

    #[test]
    fn is_empty() {
        let a: ArrayQueue<usize, 1> = ArrayQueue::new();
        assert!(a.is_empty());

        let a: ArrayQueue<usize, 2> = ArrayQueue::new();
        assert!(a.is_empty());
    }

    #[test]
    fn is_full() {
        let mut a: ArrayQueue<usize, 1> = ArrayQueue::new();
        assert!(a.push_back(0).is_ok());
        assert!(a.is_full());

        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();
        assert!(a.push_back(0).is_ok());
        assert!(a.push_back(0).is_ok());
        assert!(a.is_full());
    }

    #[test]
    fn iterator() {
        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert!(a.push_back(0).is_ok());
        assert!(a.push_back(1).is_ok());

        for (i, e) in a.into_iter().enumerate() {
            assert_eq!(*e, i);
        }
    }

    #[test]
    fn iterator_across_edges() {
        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert!(a.push_back(42).is_ok());
        a.pop_front();
        assert!(a.push_back(0).is_ok());
        assert!(a.push_back(1).is_ok());

        for (i, e) in a.into_iter().enumerate() {
            assert_eq!(*e, i);
        }
    }

    #[test]
    fn iterate_forward_and_backward() {
        let mut a = ArrayQueue::<usize, 2>::new();

        assert!(a.push_back(0).is_ok());
        assert!(a.push_back(1).is_ok());

        let mut i = a.into_iter();

        assert_eq!(i.next(), Some(&0));
        assert_eq!(i.next_back(), Some(&1));
        assert_eq!(i.next(), None);
        assert_eq!(i.next_back(), None);
    }

    #[test]
    fn iterate_forward_and_backward_mutable() {
        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert!(a.push_back(0).is_ok());
        assert!(a.push_back(1).is_ok());

        let mut i = (&mut a).into_iter();

        assert_eq!(i.next(), Some(&mut 0));
        assert_eq!(i.next_back(), Some(&mut 1));
        assert_eq!(i.next(), None);
        assert_eq!(i.next_back(), None);
    }

    #[test]
    fn iterate_empty_queue() {
        let a = ArrayQueue::<usize, 0>::new();

        for _ in a.into_iter() {}
    }

    #[test]
    fn iterator_mut() {
        let mut a: ArrayQueue<usize, 2> = ArrayQueue::new();

        assert!(a.push_back(0).is_ok());
        assert!(a.push_back(1).is_ok());

        for (i, e) in (&mut a).into_iter().enumerate() {
            assert_eq!(*e, i);
            *e = 42;
        }
    }

    #[test]
    fn reference_elements() {
        let mut a: ArrayQueue<Box<usize>, 2> = ArrayQueue::new();
        assert!(a.push_back(Box::new(42)).is_ok());
        assert!(a.push_front(Box::new(42)).is_ok());
    }

    #[test]
    fn clone() {
        let mut a: ArrayQueue<Box<usize>, 32> = ArrayQueue::new();

        for _ in 0..32 {
            assert!(a.push_back(Box::new(42)).is_ok());
        }

        let _ = a.clone();
    }

    static mut FOO_SUM: usize = 0;

    #[derive(Clone)]
    struct Foo;

    impl Drop for Foo {
        fn drop(&mut self) {
            unsafe {
                FOO_SUM += 1;
            }
        }
    }

    #[test]
    fn no_drops_of_elements_on_push_back() {
        assert_eq!(unsafe { FOO_SUM }, 0);

        let mut a: ArrayQueue<Foo, 32> = ArrayQueue::new();

        for _ in 0..32 {
            assert!(a.push_back(Foo).is_ok());
        }

        assert_eq!(unsafe { FOO_SUM }, 32); // drops of arguments `&Foo`

        drop(a);

        assert_eq!(unsafe { FOO_SUM }, 64); // drops of elements
    }

    static mut BAR_SUM: usize = 0;

    #[derive(Clone)]
    struct Bar;

    impl Drop for Bar {
        fn drop(&mut self) {
            unsafe {
                BAR_SUM += 1;
            }
        }
    }

    #[test]
    fn drops_of_elements_on_pop_back() {
        assert_eq!(unsafe { BAR_SUM }, 0);

        let mut a: ArrayQueue<Bar, 32> = ArrayQueue::new();

        for _ in 0..32 {
            assert!(a.push_back(Bar).is_ok());
        }

        assert_eq!(unsafe { BAR_SUM }, 0);

        for _ in 0..32 {
            assert!(a.pop_back().is_some());
        }

        assert_eq!(unsafe { BAR_SUM }, 32);

        drop(a);

        assert_eq!(unsafe { BAR_SUM }, 32);
    }
}
