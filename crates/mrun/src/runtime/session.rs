//! `Session`: the reuse policy of `mrun run`, one conversation kept in the live engine state.

use crate::chat::prompt::RenderedPrompt;
use crate::runtime::generate::{ChunkPlan, Prepared, Pressure, ReusePolicy};
use mrun_engine::{Checkpoint, Engine, EngineError};

/// Checkpoints a hybrid conversation keeps (planned at each turn's generation start).
const KEPT_CHECKPOINTS: usize = 2;
/// Output positions the first growth of a request covers.
const FIRST_GROWTH: u32 = 4096;

#[derive(Debug, Default)]
pub struct Session {
    /// Prompt plus emitted output of the last request: the conversation.
    committed: Vec<u32>,
    /// Tokens the engine state holds: a prefix of `committed` after every finish.
    tokens: Vec<u32>,
    /// Hybrid only: checkpoints at positions within `tokens`, oldest first.
    checkpoints: Vec<Checkpoint>,
}

impl Session {
    pub fn committed(&self) -> &[u32] {
        &self.committed
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Length of the state prefix `prompt` can reuse; its last token is always processed.
    fn common(&self, prompt: &RenderedPrompt) -> u32 {
        self.tokens
            .iter()
            .zip(&prompt.tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.tokens.len().saturating_sub(1)) as u32
    }

    /// The generation-start checkpoint to plan when prefill starts at `resume`.
    fn planned_checkpoint(
        engine: &dyn Engine,
        prompt: &RenderedPrompt,
        resume: u32,
    ) -> Option<u32> {
        let at = prompt.generation_start;
        (engine.info().is_hybrid() && at > resume && at < prompt.len()).then_some(at)
    }

    /// Cuts the state to `len`; where recurrent state cannot be cut, resumes from the deepest
    /// checkpoint at or below `len`, else from empty. Returns the length reached.
    fn cut(&mut self, engine: &mut dyn Engine, len: u32) -> u32 {
        if engine.trim(len).is_ok() {
            return len;
        }
        let deepest = self
            .checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.position <= len)
            .max_by_key(|checkpoint| checkpoint.position);
        if let Some(checkpoint) = deepest
            && engine.restore_checkpoint(checkpoint.id).is_ok()
        {
            return checkpoint.position;
        }
        engine.reset();
        0
    }

    /// Drops checkpoints above `position`, then all but the newest [`KEPT_CHECKPOINTS`].
    fn retain_checkpoints(&mut self, engine: &mut dyn Engine, position: u32) {
        self.checkpoints
            .sort_by_key(|checkpoint| checkpoint.position);
        let valid = self
            .checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.position <= position)
            .count();
        let first_kept = valid.saturating_sub(KEPT_CHECKPOINTS);
        let mut dropped: Vec<_> = self.checkpoints.drain(valid..).collect();
        dropped.extend(self.checkpoints.drain(..first_kept));
        for checkpoint in dropped {
            engine.drop_checkpoint(checkpoint.id);
        }
    }

    /// Drops the `count` oldest checkpoints; returns their bytes.
    fn drop_oldest(&mut self, engine: &mut dyn Engine, count: usize) -> u64 {
        self.checkpoints
            .drain(..count.min(self.checkpoints.len()))
            .map(|checkpoint| {
                engine.drop_checkpoint(checkpoint.id);
                checkpoint.bytes
            })
            .sum()
    }
}

impl ReusePolicy for Session {
    fn initial_need(&self, engine: &dyn Engine, prompt: &RenderedPrompt, total: u32) -> u64 {
        let info = engine.info();
        let positions = total.min(prompt.len().saturating_add(FIRST_GROWTH));
        let checkpoints = Session::planned_checkpoint(engine, prompt, self.common(prompt))
            .map_or(0, |_| info.recurrent_bytes);
        // The engine holds only the live state and this session's checkpoints.
        let held: u64 = self
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.bytes)
            .sum();
        let live = engine.allocated().saturating_sub(held);
        let target = u64::from(positions) * info.kv_bytes_per_token + info.recurrent_bytes;
        target.saturating_sub(live) + checkpoints
    }

    /// Oldest checkpoint first. `protect` is not consulted: losing a checkpoint costs only
    /// re-processing, and the session holds nothing else.
    fn evict(
        &mut self,
        engine: &mut dyn Engine,
        bytes: u64,
        _protect: Option<&RenderedPrompt>,
    ) -> u64 {
        let mut freed = 0;
        while freed < bytes && !self.checkpoints.is_empty() {
            freed += self.drop_oldest(engine, 1);
        }
        freed
    }

    fn prepare(
        &mut self,
        engine: &mut dyn Engine,
        prompt: &RenderedPrompt,
        total: u32,
    ) -> Result<Prepared, EngineError> {
        let common = self.common(prompt);
        self.retain_checkpoints(engine, common);
        let resume = self.cut(engine, common);
        self.tokens.truncate(resume as usize);
        engine.reserve(resume, total)?;
        let checkpoints = Session::planned_checkpoint(engine, prompt, resume)
            .into_iter()
            .collect();
        Ok(Prepared {
            resume,
            chunks: ChunkPlan {
                boundaries: Vec::new(),
                checkpoints,
            },
            miss: None,
        })
    }

    /// Warning halves the checkpoints (oldest first); Critical drops them all and, when idle,
    /// the live state too.
    fn on_pressure(&mut self, engine: &mut dyn Engine, level: Pressure, idle: bool) {
        let count = match level {
            Pressure::Warning => self.checkpoints.len().div_ceil(2),
            Pressure::Critical => self.checkpoints.len(),
        };
        self.drop_oldest(engine, count);
        if idle && level == Pressure::Critical {
            engine.reset();
            self.tokens.clear();
        }
    }

    fn commit(
        &mut self,
        engine: &mut dyn Engine,
        _prompt: &RenderedPrompt,
        committed: &[u32],
        target: u32,
        captured: Vec<Checkpoint>,
    ) -> u32 {
        self.checkpoints.extend(captured);
        let reached = self.cut(engine, target);
        self.retain_checkpoints(engine, reached);
        self.tokens = committed[..reached as usize].to_vec();
        self.committed = committed.to_vec();
        reached
    }

    fn clear(&mut self, engine: &mut dyn Engine) {
        engine.reset();
        for checkpoint in self.checkpoints.drain(..) {
            engine.drop_checkpoint(checkpoint.id);
        }
        self.tokens.clear();
        self.committed.clear();
    }
}
