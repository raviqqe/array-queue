//! A queue backed by a fixed-size array.

#![no_std]

extern crate alloc;

mod array_queue;
mod error;

pub use array_queue::ArrayQueue;
