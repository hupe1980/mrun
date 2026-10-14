pub mod model;
pub mod qualification;

use crate::budget::ledger::Lease;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendError {
    Bounds,
    Cancelled,
    State,
    Fault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Evaluation {
    Prefill,
    Decode,
}

#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn observed(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

pub trait Adapter: 'static {
    fn inspect(&mut self) -> Result<u32, BackendError>;
    fn load(&mut self, leases: Vec<Lease>) -> Result<(), BackendError>;
    fn render(&mut self, input: &[u8]) -> Result<Vec<u32>, BackendError>;
    fn reset_request(&mut self) -> Result<(), BackendError>;
    fn attach(&mut self, retained: u32, end: u32) -> Result<(), BackendError>;
    fn evaluate(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        kind: Evaluation,
    ) -> Result<(), BackendError>;
    fn logits(&self) -> Result<&[f32], BackendError>;
    fn publish(&mut self, retained: u32, end: u32) -> Result<(), BackendError>;
    fn sample_parse(&mut self) -> Result<Vec<u8>, BackendError>;
    fn synchronize(&mut self) -> Result<(), BackendError>;
    fn release(&mut self) -> Result<(), BackendError>;
    fn clear(&mut self) -> Result<(), BackendError>;
    fn unload(&mut self) -> Result<(), BackendError>;
}

pub struct Owner<Backend: Adapter> {
    backend: Backend,
    cancellation: Cancellation,
    submitted: bool,
    loaded: bool,
    failed: bool,
    prompt_boundary: bool,
    decoding: bool,
    active: bool,
    position: u32,
    vocabulary: u32,
    thread_bound: PhantomData<Rc<()>>,
}

pub fn spawn<Backend, Factory, Work, Output>(
    factory: Factory,
    work: Work,
) -> std::io::Result<std::thread::JoinHandle<Result<Output, BackendError>>>
where
    Backend: Adapter,
    Factory: FnOnce() -> Result<Backend, BackendError> + Send + 'static,
    Work: FnOnce(&mut Owner<Backend>) -> Result<Output, BackendError> + Send + 'static,
    Output: Send + 'static,
{
    std::thread::Builder::new()
        .name("mrun-native-owner".into())
        .spawn(move || {
            let mut owner = Owner {
                backend: factory()?,
                cancellation: Cancellation::default(),
                submitted: false,
                loaded: false,
                failed: false,
                prompt_boundary: false,
                decoding: false,
                active: false,
                position: 0,
                vocabulary: 0,
                thread_bound: PhantomData,
            };
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&mut owner)))
                    .unwrap_or(Err(BackendError::Fault));
            let cleanup = owner.backend.synchronize();
            if cleanup.is_err() {
                std::mem::forget(owner);
                return Err(BackendError::Fault);
            }
            if owner.active {
                owner.backend.release()?;
            }
            owner.backend.unload()?;
            result
        })
}

impl<Backend: Adapter> Owner<Backend> {
    fn ready(&self) -> Result<(), BackendError> {
        if self.failed || !self.loaded || self.submitted {
            return Err(BackendError::State);
        }
        if self.cancellation.observed() {
            return Err(BackendError::Cancelled);
        }
        Ok(())
    }

    pub fn load(&mut self, leases: Vec<Lease>) -> Result<(), BackendError> {
        if self.loaded || leases.is_empty() || leases.len() > 16 {
            return Err(BackendError::State);
        }
        let vocabulary = self.backend.inspect()?;
        if vocabulary == 0 || vocabulary > 1_048_576 {
            return Err(BackendError::Bounds);
        }
        self.backend.load(leases)?;
        self.loaded = true;
        self.vocabulary = vocabulary;
        Ok(())
    }

    pub fn begin_request(&mut self, cancellation: Cancellation) -> Result<(), BackendError> {
        if self.submitted || self.failed || !self.loaded {
            return Err(BackendError::State);
        }
        if self.active {
            self.release()?;
        }
        self.backend.reset_request()?;
        self.cancellation = cancellation;
        self.position = 0;
        self.prompt_boundary = false;
        self.decoding = false;
        Ok(())
    }

    pub fn render(&mut self, input: &[u8]) -> Result<Vec<u32>, BackendError> {
        self.ready()?;
        if input.len() > 64 * 1024 * 1024 {
            return Err(BackendError::Bounds);
        }
        let tokens = self.backend.render(input)?;
        if tokens.is_empty()
            || tokens.len() > 16_384
            || tokens.iter().any(|token| *token >= self.vocabulary)
        {
            return Err(BackendError::Bounds);
        }
        Ok(tokens)
    }

    pub fn attach(&mut self, retained: u32, end: u32) -> Result<(), BackendError> {
        self.ready()?;
        if !(1..=4).contains(&retained) || end > 16_384 || self.position != 0 {
            return Err(BackendError::Bounds);
        }
        self.backend.attach(retained, end)?;
        self.position = end;
        self.active = true;
        Ok(())
    }

    pub fn evaluate(&mut self, tokens: &[u32], kind: Evaluation) -> Result<(), BackendError> {
        self.ready()?;
        if kind == Evaluation::Prefill && self.decoding {
            return Err(BackendError::State);
        }
        let limit = if kind == Evaluation::Prefill { 128 } else { 1 };
        let end = self
            .position
            .checked_add(tokens.len().try_into().map_err(|_| BackendError::Bounds)?)
            .ok_or(BackendError::Bounds)?;
        if tokens.is_empty()
            || tokens.len() > limit
            || end > 16_384
            || tokens.iter().any(|token| *token >= self.vocabulary)
        {
            return Err(BackendError::Bounds);
        }
        let positions: Vec<_> = (self.position..end).collect();
        self.submitted = true;
        self.active = true;
        self.decoding |= kind == Evaluation::Decode;
        match self.backend.evaluate(tokens, &positions, kind) {
            Ok(()) => {
                self.position = end;
                self.prompt_boundary = kind == Evaluation::Prefill;
                Ok(())
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    pub fn synchronize(&mut self) -> Result<(), BackendError> {
        self.backend.synchronize().inspect_err(|_| {
            self.failed = true;
        })?;
        self.submitted = false;
        Ok(())
    }

    pub fn copy_logits(&self) -> Result<Vec<f32>, BackendError> {
        self.ready()?;
        let logits = self.backend.logits()?;
        if logits.len() != self.vocabulary as usize || logits.iter().any(|value| !value.is_finite())
        {
            return Err(BackendError::Bounds);
        }
        Ok(logits.to_vec())
    }

    pub fn publish(&mut self, retained: u32) -> Result<(), BackendError> {
        self.ready()?;
        if !self.prompt_boundary || !(1..=4).contains(&retained) {
            return Err(BackendError::State);
        }
        self.backend.publish(retained, self.position)
    }

    pub fn sample_parse(&mut self) -> Result<Vec<u8>, BackendError> {
        self.ready()?;
        let bytes = self.backend.sample_parse()?;
        if bytes.len() > 256 * 1024 {
            return Err(BackendError::Bounds);
        }
        Ok(bytes)
    }

    pub fn release(&mut self) -> Result<(), BackendError> {
        self.synchronize()?;
        if self.active {
            self.backend.release()?;
            self.active = false;
        }
        self.position = 0;
        self.prompt_boundary = false;
        Ok(())
    }

    pub fn clear(&mut self) -> Result<(), BackendError> {
        self.synchronize()?;
        self.backend.clear()
    }
}
