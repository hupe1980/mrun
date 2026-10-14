//! mrun is a planned local LLM runtime for Apple Silicon.
//!
//! This placeholder release provides no inference functionality or stable public API.
//! Development plans are available in the project repository.

#![deny(unsafe_code)]

mod api {}
#[expect(
    dead_code,
    reason = "Private report and native types precede owner integration"
)]
mod backend;
#[expect(
    dead_code,
    reason = "Private foundation policies precede owner integration"
)]
mod budget;
mod cache {}
pub mod config;
mod runtime {}
mod telemetry {}
