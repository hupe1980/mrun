//! The per-request decode hook (002 FR-036) running on the engine thread: token masks,
//! grammar-forced tokens and `ignore_eos`.

use crate::runtime::generate::DecodeHook;
use mrun_engine::TokenMask;

/// What a request asks of the decode loop beyond sampling.
#[derive(Debug, Default)]
pub struct RequestHook {
    /// FR-031: EOS tokens are ordinary output; only the output limit ends generation.
    pub ignore_eos: bool,
}

impl RequestHook {
    /// `None` when the request needs nothing from the hook, so decode stays unhooked.
    pub fn boxed(self) -> Option<Box<dyn DecodeHook>> {
        self.ignore_eos
            .then(|| Box::new(self) as Box<dyn DecodeHook>)
    }
}

impl DecodeHook for RequestHook {
    fn on_token(&mut self, _id: u32) -> Option<TokenMask> {
        None
    }

    fn ignore_eos(&self) -> bool {
        self.ignore_eos
    }
}
