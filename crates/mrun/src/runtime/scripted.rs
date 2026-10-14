//! `ScriptedEngine`: a deterministic [`Engine`] for tests on any OS (no MLX).

use mrun_engine::{
    Architecture, CancelToken, Engine, EngineError, KvSegmentId, MemoryBudget, ModelInfo, Sampling,
    Segment, TokenMask,
};
use std::collections::HashMap;
use std::ops::Range;

/// Emits a fixed token script, records exactly which tokens are in its "cache", and tracks
/// which token ranges each snapshot holds, so tests can check reuse structurally.
#[derive(Debug)]
pub struct ScriptedEngine {
    info: ModelInfo,
    budget: MemoryBudget,
    script: Vec<u32>,
    next: usize,
    active: Vec<u32>,
    segments: HashMap<KvSegmentId, Vec<u32>>,
    next_segment: u64,
    /// Tokens processed per `prefill` call between cancel checks (models a layer boundary).
    pub cancel_granularity: usize,
    /// Every token passed to `prefill` or fed by `decode`, in order.
    pub processed: Vec<u32>,
}

/// End-of-sequence token used by [`ScriptedEngine`].
pub const SCRIPT_EOS: u32 = u32::MAX - 1;

impl ScriptedEngine {
    /// `script` is returned by successive `decode` calls, then [`SCRIPT_EOS`] forever.
    pub fn new(script: Vec<u32>, capacity_tokens: u32) -> Self {
        Self {
            info: ModelInfo {
                identity: "scripted".into(),
                architecture: Architecture::Qwen3,
                layers: 1,
                kv_heads: 1,
                head_dim: 1,
                vocab: u32::MAX,
                max_position: capacity_tokens,
                kv_bytes_per_token: 1,
                eos_tokens: vec![SCRIPT_EOS],
            },
            budget: MemoryBudget {
                kv_capacity_tokens: capacity_tokens,
                ..MemoryBudget::default()
            },
            script,
            next: 0,
            active: Vec::new(),
            segments: HashMap::new(),
            next_segment: 0,
            cancel_granularity: 64,
            processed: Vec::new(),
        }
    }

    pub fn active_tokens(&self) -> &[u32] {
        &self.active
    }

    pub fn segment_tokens(&self, id: KvSegmentId) -> Option<&[u32]> {
        self.segments.get(&id).map(Vec::as_slice)
    }
}

impl Engine for ScriptedEngine {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    fn active_len(&self) -> u32 {
        self.active.len() as u32
    }

    fn reserve(&mut self, total_tokens: u32) -> Result<(), EngineError> {
        let capacity = self.budget.kv_capacity_tokens;
        if total_tokens > capacity {
            return Err(EngineError::Capacity {
                requested: total_tokens,
                available: capacity,
            });
        }
        Ok(())
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        cancel: &CancelToken,
        progress: &mut dyn FnMut(u32),
    ) -> Result<u32, EngineError> {
        let mut done = 0;
        for chunk in tokens.chunks(self.cancel_granularity.max(1)) {
            if cancel.is_cancelled() {
                break;
            }
            self.active.extend_from_slice(chunk);
            self.processed.extend_from_slice(chunk);
            done += chunk.len() as u32;
            progress(done);
        }
        Ok(done)
    }

    fn decode(
        &mut self,
        last: u32,
        _sampling: &Sampling,
        mask: Option<&TokenMask>,
    ) -> Result<u32, EngineError> {
        if self.active.len() as u32 >= self.budget.kv_capacity_tokens {
            return Err(EngineError::Backend(
                "scripted engine out of capacity".into(),
            ));
        }
        self.active.push(last);
        self.processed.push(last);
        let token = self.script.get(self.next).copied().unwrap_or(SCRIPT_EOS);
        self.next += 1;
        if let Some(mask) = mask
            && token != SCRIPT_EOS
            && !mask.is_allowed(token)
        {
            return Err(EngineError::Backend(format!(
                "script token {token} is masked"
            )));
        }
        Ok(token)
    }

    fn trim(&mut self, len: u32) {
        self.active.truncate(len as usize);
    }

    fn snapshot(&mut self, range: Range<u32>) -> Result<Segment, EngineError> {
        let len = self.active.len() as u32;
        if range.start > range.end || range.end > len {
            return Err(EngineError::InvalidRange {
                start: range.start,
                end: range.end,
                len,
            });
        }
        let id = KvSegmentId(self.next_segment);
        self.next_segment += 1;
        let tokens = self.active[range.start as usize..range.end as usize].to_vec();
        let bytes = tokens.len() as u64 * self.info.kv_bytes_per_token;
        self.segments.insert(id, tokens);
        Ok(Segment { id, bytes })
    }

    fn restore(&mut self, segments: &[KvSegmentId], keep: u32) -> Result<(), EngineError> {
        let mut restored = Vec::new();
        for id in segments {
            restored.extend_from_slice(
                self.segments
                    .get(id)
                    .ok_or(EngineError::UnknownSegment(*id))?,
            );
        }
        restored.truncate(keep as usize);
        self.active = restored;
        Ok(())
    }

    fn drop_segment(&mut self, id: KvSegmentId) {
        self.segments.remove(&id);
    }

    fn reset(&mut self) {
        self.active.clear();
    }
}
