use core::error::Error;
use core::fmt::{Display, Formatter, Result};

const MESSAGE: &str = "queue is full";

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CapacityError;

impl Display for CapacityError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{MESSAGE}")
    }
}

impl Error for CapacityError {
    fn description(&self) -> &str {
        MESSAGE
    }
}
