use crate::backend::{self, Adapter, BackendError, Cancellation, Evaluation};
use crate::budget::ledger::{Lease, Ledger, Pin};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

struct Fake {
    events: Arc<Mutex<Vec<&'static str>>>,
    leases: Vec<Lease>,
    pins: Vec<Pin>,
    logits: Vec<f32>,
    submitted: bool,
    thread_local: Rc<Cell<u32>>,
}

impl Fake {
    fn event(&self, event: &'static str) {
        let mut events = self.events.lock().unwrap();
        assert!(events.len() < 128);
        events.push(event);
    }
}

impl Adapter for Fake {
    fn inspect(&mut self) -> Result<u32, BackendError> {
        self.event("inspect");
        Ok(32)
    }
    fn load(&mut self, leases: Vec<Lease>) -> Result<(), BackendError> {
        self.event("load");
        self.leases = leases;
        Ok(())
    }
    fn render(&mut self, _input: &[u8]) -> Result<Vec<u32>, BackendError> {
        self.event("render");
        Ok(vec![1, 2])
    }
    fn reset_request(&mut self) -> Result<(), BackendError> {
        self.event("reset");
        self.thread_local.set(0);
        Ok(())
    }
    fn attach(&mut self, _retained: u32, _end: u32) -> Result<(), BackendError> {
        assert!(!self.submitted);
        self.event("attach");
        Ok(())
    }
    fn evaluate(
        &mut self,
        _tokens: &[u32],
        _positions: &[u32],
        _kind: Evaluation,
    ) -> Result<(), BackendError> {
        assert!(!self.submitted);
        self.event("evaluate");
        self.submitted = true;
        self.pins = self
            .leases
            .iter()
            .map(|lease| lease.pin().unwrap())
            .collect();
        self.logits.fill(1.0);
        Ok(())
    }
    fn logits(&self) -> Result<&[f32], BackendError> {
        assert!(!self.submitted);
        Ok(&self.logits)
    }
    fn publish(&mut self, _retained: u32, _end: u32) -> Result<(), BackendError> {
        assert!(!self.submitted);
        self.event("publish");
        Ok(())
    }
    fn sample_parse(&mut self) -> Result<Vec<u8>, BackendError> {
        self.event("sample");
        Ok(b"fixture".to_vec())
    }
    fn synchronize(&mut self) -> Result<(), BackendError> {
        self.event("synchronize");
        self.submitted = false;
        self.pins.clear();
        Ok(())
    }
    fn release(&mut self) -> Result<(), BackendError> {
        assert!(!self.submitted);
        self.event("release");
        Ok(())
    }
    fn clear(&mut self) -> Result<(), BackendError> {
        assert!(!self.submitted);
        self.event("clear");
        Ok(())
    }
    fn unload(&mut self) -> Result<(), BackendError> {
        assert!(!self.submitted);
        self.event("unload");
        self.leases.clear();
        Ok(())
    }
}

#[test]
fn native_owner_is_a_dedicated_thread_and_copies_only_synchronized_logits() {
    let events = Arc::new(Mutex::new(vec![]));
    let logged = events.clone();
    let thread = std::thread::current().id();
    let ledger = Ledger::new(1024);
    let leases = vec![ledger.reserve(1024).unwrap()];
    let task = backend::spawn(
        move || {
            Ok(Fake {
                events: logged,
                leases: vec![],
                pins: vec![],
                logits: vec![0.0; 32],
                submitted: false,
                thread_local: Rc::new(Cell::new(0)),
            })
        },
        move |owner| {
            assert_ne!(std::thread::current().id(), thread);
            owner.load(leases)?;
            owner.begin_request(Cancellation::default())?;
            assert_eq!(owner.render(b"public fixture")?, vec![1, 2]);
            assert_eq!(
                owner.evaluate(&[1; 129], Evaluation::Prefill),
                Err(BackendError::Bounds)
            );
            owner.attach(1, 0)?;
            owner.evaluate(&[1, 2], Evaluation::Prefill)?;
            assert_eq!(owner.copy_logits(), Err(BackendError::State));
            assert_eq!(owner.publish(1), Err(BackendError::State));
            owner.synchronize()?;
            let owned = owner.copy_logits()?;
            owner.publish(1)?;
            assert_eq!(owner.sample_parse()?, b"fixture");
            owner.evaluate(&[3], Evaluation::Decode)?;
            owner.synchronize()?;
            assert_eq!(owned, vec![1.0; 32]);
            assert_eq!(owner.publish(2), Err(BackendError::State));
            assert_eq!(
                owner.evaluate(&[1], Evaluation::Prefill),
                Err(BackendError::State)
            );
            owner.release()?;
            owner.release()?;
            owner.clear()?;
            Ok(())
        },
    )
    .unwrap();
    task.join().unwrap().unwrap();
    assert_eq!(ledger.totals().reserved, 0);
    let recorded = events.lock().unwrap();
    assert_eq!(
        recorded
            .iter()
            .filter(|event| **event == "evaluate")
            .count(),
        2
    );
    assert_eq!(
        recorded.iter().filter(|event| **event == "release").count(),
        1
    );
    for (index, event) in recorded.iter().enumerate() {
        if *event == "release" || *event == "unload" || *event == "clear" {
            assert_eq!(recorded[index - 1], "synchronize");
        }
    }
}

#[test]
fn observed_cancellation_prevents_further_native_calls_but_allows_safe_release() {
    let events = Arc::new(Mutex::new(vec![]));
    let logged = events.clone();
    let ledger = Ledger::new(1024);
    let lease = ledger.reserve(1024).unwrap();
    backend::spawn(
        move || {
            Ok(Fake {
                events: logged,
                leases: vec![],
                pins: vec![],
                logits: vec![0.0; 32],
                submitted: false,
                thread_local: Rc::new(Cell::new(0)),
            })
        },
        move |owner| {
            owner.load(vec![lease])?;
            let cancel = Cancellation::default();
            owner.begin_request(cancel.clone())?;
            owner.evaluate(&[1], Evaluation::Prefill)?;
            cancel.cancel();
            owner.synchronize()?;
            assert_eq!(
                owner.evaluate(&[2], Evaluation::Decode),
                Err(BackendError::Cancelled)
            );
            assert_eq!(owner.publish(1), Err(BackendError::Cancelled));
            owner.release()?;
            Ok(())
        },
    )
    .unwrap()
    .join()
    .unwrap()
    .unwrap();
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| **event == "evaluate")
            .count(),
        1
    );
    assert_eq!(ledger.totals().reserved, 0);
}

#[test]
fn owner_panic_synchronizes_submitted_work_before_release_and_unload() {
    let events = Arc::new(Mutex::new(vec![]));
    let logged = events.clone();
    let ledger = Ledger::new(1024);
    let lease = ledger.reserve(1024).unwrap();
    let task = backend::spawn(
        move || {
            Ok(Fake {
                events: logged,
                leases: vec![],
                pins: vec![],
                logits: vec![0.0; 32],
                submitted: false,
                thread_local: Rc::new(Cell::new(0)),
            })
        },
        move |owner| -> Result<(), BackendError> {
            owner.load(vec![lease])?;
            owner.evaluate(&[1], Evaluation::Prefill)?;
            panic!("public fake owner fault");
        },
    )
    .unwrap();
    assert_eq!(task.join().unwrap(), Err(BackendError::Fault));
    assert_eq!(ledger.totals().reserved, 0);
    assert_eq!(
        &events.lock().unwrap()[2..],
        &["evaluate", "synchronize", "release", "unload"]
    );
}
