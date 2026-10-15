//! `Session`: the reuse policy of `mrun run`, one conversation kept in the live engine state.

use crate::runtime::generate::{ChunkPlan, GenerateRequest, ReusePolicy};
use mrun_engine::{Checkpoint, Engine, EngineError};

/// Checkpoints a hybrid conversation keeps (planned at each turn's generation start).
const KEPT_CHECKPOINTS: usize = 2;

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
}

impl ReusePolicy for Session {
    fn prepare(
        &mut self,
        engine: &mut dyn Engine,
        request: &GenerateRequest,
    ) -> Result<u32, EngineError> {
        let prompt = &request.prompt;
        let common = self
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len() - 1) as u32;
        self.retain_checkpoints(engine, common);
        let keep = self.cut(engine, common);
        self.tokens.truncate(keep as usize);
        Ok(keep)
    }

    fn plan(&self, engine: &dyn Engine, request: &GenerateRequest, keep: u32) -> ChunkPlan {
        let end = request.prompt.len() as u32 - 1;
        ChunkPlan {
            boundaries: Vec::new(),
            checkpoints: request
                .generation_start
                .filter(|&at| engine.info().is_hybrid() && at > keep && at <= end)
                .into_iter()
                .collect(),
        }
    }

    fn evict(&mut self, engine: &mut dyn Engine, _bytes: u64) -> bool {
        if self.checkpoints.is_empty() {
            return false;
        }
        engine.drop_checkpoint(self.checkpoints.remove(0).id);
        true
    }

    fn commit(
        &mut self,
        engine: &mut dyn Engine,
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
