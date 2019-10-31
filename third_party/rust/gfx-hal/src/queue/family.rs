//! Queue family and groups.

use crate::backend::RawQueueGroup;
use crate::queue::capability::{Capability, Compute, Graphics, Transfer};
use crate::queue::{CommandQueue, QueueType};
use crate::Backend;

use std::any::Any;
use std::fmt::Debug;

/// General information about a queue family, available upon adapter discovery.
///
/// Note that a backend can expose multiple queue families with the same properties.
pub trait QueueFamily: Debug + Any + Send + Sync {
    /// Returns the type of queues.
    fn queue_type(&self) -> QueueType;
    /// Returns maximum number of queues created from this family.
    fn max_queues(&self) -> usize;
    /// Returns true if the queue supports graphics operations.
    fn supports_graphics(&self) -> bool {
        Graphics::supported_by(self.queue_type())
    }
    /// Returns true if the queue supports compute operations.
    fn supports_compute(&self) -> bool {
        Compute::supported_by(self.queue_type())
    }
    /// Returns true if the queue supports transfer operations.
    fn supports_transfer(&self) -> bool {
        Transfer::supported_by(self.queue_type())
    }
    /// Returns the queue family ID.
    fn id(&self) -> QueueFamilyId;
}

/// Identifier for a queue family of a physical device.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QueueFamilyId(pub usize);

/// Strong-typed group of queues of the same queue family.
#[derive(Debug)]
pub struct QueueGroup<B: Backend, C> {
    family: QueueFamilyId,
    /// Command queues created in this family.
    pub queues: Vec<CommandQueue<B, C>>,
}

impl<B: Backend, C> QueueGroup<B, C> {
    /// Return the associated queue family id.
    pub fn family(&self) -> QueueFamilyId {
        self.family
    }
}

impl<B: Backend, C: Capability> QueueGroup<B, C> {
    /// Create a new strongly typed queue group from a raw one.
    ///
    /// # Panics
    ///
    /// Panics if the family doesn't expose required queue capabilities.
    fn new(raw: RawQueueGroup<B>) -> Self {
        assert!(C::supported_by(raw.family.queue_type()));
        QueueGroup {
            family: raw.family.id(),
            queues: raw
                .queues
                .into_iter()
                .map(|q| unsafe { CommandQueue::new(q) })
                .collect(),
        }
    }
}

/// Contains a list of all instantiated queues. Conceptually structured as a collection of
/// `QueueGroup`s, one for each queue family.
#[derive(Debug)]
pub struct Queues<B: Backend>(pub(crate) Vec<RawQueueGroup<B>>);

impl<B: Backend> Queues<B> {
    /// Removes the queue family with the passed id from the queue list and
    /// returns the queue group.
    ///
    /// # Panics
    ///
    /// Panics if the family doesn't expose required queue capabilities.
    pub fn take<C: Capability>(&mut self, id: QueueFamilyId) -> Option<QueueGroup<B, C>> {
        self.0
            .iter()
            .position(|raw| raw.family.id() == id)
            .map(|index| QueueGroup::new(self.0.swap_remove(index)))
    }

    /// Removes the queue family with the passed id from the queue list and
    /// returns the command queues.
    pub fn take_raw(&mut self, id: QueueFamilyId) -> Option<Vec<B::CommandQueue>> {
        self.0
            .iter()
            .position(|raw| raw.family.id() == id)
            .map(|index| self.0.swap_remove(index).queues)
    }
}
