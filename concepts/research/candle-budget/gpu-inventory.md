# Candle Metal GPU/Backing Allocation Inventory (T114)

- Candle revision: `c68b24997319b8d76e5a4e775dab197c7f26322a` (workspace 0.11.0)
- Date: 2026-10-14
- Scope: `candle-core` Metal backend, quantized Metal storage, sort, safetensors, `candle-nn`
  Metal custom ops, `candle-metal-kernels` (host-side Rust only; `.metal` shader sources are out of scope)
- Status: source review only. Nothing was built, no test was run and no Metal hardware was used.
  All lifetime/retain statements about Apple frameworks that are not visible in Candle source are
  marked "unverified".

## Method

1. Grepped the checkout (excluding `tests.rs`, `tests/`, `examples/`, `benches/`) for
   `new_buffer`, `new_buffer_builder`, `allocate_buffer`, `allocate_zeros`, `new_private_buffer`,
   `new_buffer_with_data`, `newBuffer*`, `newHeap`, `newTexture`, `NoCopy`, `MetalStorage::new`,
   `register_buffers`, `residency`, `newFence`, `commandBuffer`, `in_flight`, pipeline caches.
2. Read every constructor body and each distinct caller class; traced `BufferBuilder::build`
   (`candle-core/src/metal_backend/device.rs:416-427`) to the four `MetalDevice` constructors and from
   there to the two raw `candle_metal_kernels::metal::Device` constructors.
3. Read sweep/release code (`drop_unused_buffers`, `wait_until_completed`, `flush_and_wait_current`)
   and command submission (`candle-metal-kernels/src/metal/commands.rs`).
4. `candle-transformers` hits for `with_size`/`with_zeros`/`with_data` were checked and are unrelated
   names (e.g. `pad_with_zeros`); no model crate constructs Metal buffers directly. No `newHeap`,
   `newTexture` or `newBufferWithBytesNoCopy` call exists in the tree (only a doc comment at
   `device.rs:175`).

Abbreviations: `dev.rs` = `candle-core/src/metal_backend/device.rs`; `mod.rs` =
`candle-core/src/metal_backend/mod.rs`; `q.rs` = `candle-core/src/quantized/metal.rs`; `mk/` =
`candle-metal-kernels/src/`. "Raw ctor" = `mk/metal/device.rs:73` (`newBufferWithLength_options`) or
`mk/metal/device.rs:86` (`newBufferWithBytes_length_options`). "Sweep" = `drop_unused_buffers`
(`dev.rs:106-130`), run only after a successful `wait_until_completed` (`dev.rs:145-152`) or
`flush_and_wait_current` (`dev.rs:155-162`).

## Inventory

| ID | file:line | What allocates | Size rule | Pre-alloc bound known? | Possible denial point | Unique lease owner | Safe release event | Status |
|---|---|---|---|---|---|---|---|---|
| G-01 | `dev.rs:285-302` (`allocate_buffer`, via `BufferInit::Size` `dev.rs:421`) | Shared-pool buffer, `StorageModeShared`; callers: readback staging `mod.rs:2007`, `q.rs:45`, `q.rs:444` | Reuse any free buffer with bucket >= request (G-04), else `next_power_of_two(size)` (`dev.rs:327-329`); bucket key = rounded size | Yes for the new-allocation branch: rounded size is computed at `dev.rs:291` before raw ctor `dev.rs:296`; reuse branch allocates nothing | Between `dev.rs:291` and `dev.rs:294` (under pool write lock) or inside raw ctor `mk/metal/device.rs:73` | Pool `Vec<Arc<Buffer>>` (`dev.rs:300`) co-owns with every tensor `Arc`; no single owner | Pool capacity freed only by sweep when `Arc::strong_count == 1` after a successful wait; MTLBuffer may outlive sweep while a committed CB retains it (unverified Apple semantics) | covered-by-single-hook (raw ctor) / needs-patch |
| G-02 | `dev.rs:197-219` (`MetalDevice::new_buffer`, via `with_size_for` `dev.rs:419`) | Private-pool buffer, `StorageModePrivate` (Shared on iOS, `dev.rs:76-81`); ~all op outputs: `mod.rs:130,194,250,363,422,552,674,860,927,1014,1063,1129,1215,1280,1327,1376,1427,1464,1615,1738,1961,1980,2068`; `q.rs:255,317,384`; `candle-core/src/sort.rs:228`; `candle-nn/src/ops.rs:150,415,637,886,1053,1140,1146,1151`; `candle-nn/src/rotary_emb.rs:223,528,802` | `elem_count * dtype.size_in_bytes()` (`dev.rs:203`, unchecked multiply) then reuse >= size in private pool, else `next_power_of_two` (`dev.rs:208`) | Yes after `dev.rs:208`; but input multiply can overflow (debug panic / release wrap) and F6/F4/F8E8M0 report 0 bytes (`candle-core/src/dtype.rs:108ff`) | Between `dev.rs:208` and `dev.rs:211`, or raw ctor | Private pool `Vec` + tensor `Arc`s | Sweep (as G-01) | covered-by-single-hook (raw ctor) / needs-patch |
| G-03 | `dev.rs:260-282` (`allocate_zeros`, via `with_zeros` `dev.rs:422`); callers `mod.rs:2083` (`zeros_impl`), `q.rs:17` (`QMetalStorage::zeros`) | G-01 buffer plus a blit fill of the whole `buffer.length()` (rounded/reused size) | As G-01 | As G-01; the blit also creates encoder/fence metadata (G-15) | As G-01 | As G-01 | As G-01 after fill CB completes | covered-by-single-hook / needs-patch |
| G-04 | `dev.rs:463-477` (`find_available_buffer`) | No new backing: hands out an existing pooled buffer | Smallest bucket >= request with a strong_count==1 entry; slack unbounded (a 1 GiB free buffer can serve 4 bytes) | n/a (no allocation) | None needed; but charge must stay with pool capacity, not logical tensor bytes | Lease silently moves from previous tensor to new tensor via `Arc` clone (`dev.rs:289`, `dev.rs:206`) | Same as G-01/G-02 | needs-patch (lease identity/charge transfer not observable) |
| G-05 | `dev.rs:244-258` (`MetalDevice::new_buffer_with_data`, via `with_data` `dev.rs:389-393`) | Shared buffer copied from host slice; callers `mod.rs:2100-2163` (`storage_from_slice`), `mod.rs:2183-2246` (`storage_from_cpu_storage[_owned]` `mod.rs:2266`), `candle-core/src/safetensors.rs:264,368` (F4/F6/F8E8M0 only), `q.rs:122,143,165,191,212,463` | Exact `size_of_val(data)` (`dev.rs:245`), NO rounding; always allocates; raw ctor runs at `dev.rs:246-249` BEFORE pool lock `dev.rs:250`; then inserted in shared pool under exact (non power-of-two) key `dev.rs:252` and becomes reusable by G-01 | Yes (`dev.rs:245`) | Before `dev.rs:246` or in raw ctor `mk/metal/device.rs:86` | Shared pool + tensor `Arc`s | Sweep | covered-by-single-hook (raw ctor) / needs-patch |
| G-06 | `dev.rs:224-238` (`new_private_buffer`) | Unpooled private buffer; public; no caller found in tree | `elem_count * dtype.size_in_bytes()` exact, no rounding | Yes (`dev.rs:230`) | Before `dev.rs:231` or raw ctor | Caller's `Arc` only | Never removed from residency set (`dev.rs:236`); if the set retains allocations, backing is never freed (unverified) | unsupported (forbid in experimental route) |
| G-07 | `mod.rs:2031-2041` (RNG seed in `MetalDevice::new`) | 8-byte shared buffer via raw `Device::new_buffer_with_data`, bypassing `MetalDevice`/builder and pools | Exact 8 bytes | Yes (constant) | Raw ctor only (builder hook does not see it) | `MetalDevice.seed: Arc<Mutex<Buffer>>` (`dev.rs:64`) | Device drop; residency entry (`mod.rs:2040`) never removed | needs-patch (not via builder); charge as fixed device overhead |
| G-08 | `mod.rs:2270-2348` (`rand_uniform`/`rand_normal`) | Output via G-02 (`mod.rs:2284`, `mod.rs:2324`); reads seed G-07 under `lock().unwrap()` (`mod.rs:2297`, `mod.rs:2337`) | As G-02 | As G-02 | As G-02 | As G-02 | As G-02 | covered-by-single-hook (via G-02) |
| G-09 | `q.rs:458-470` (`load_quantized`), dispatched from `candle-core/src/quantized/mod.rs:102-117` | GGUF/GGML weight upload via G-05 | Exact byte length of block slice | Yes | As G-05 | `QMetalStorage.buffer` (`q.rs:10`) + shared pool | Sweep after QTensor drop | covered-by-single-hook / needs-patch |
| G-10 | `q.rs:40-131` (`QMetalStorage::dequantize`); `candle-core/src/quantized/mod.rs:699-712` (`dequantize_f16` adds `to_dtype`, G-02) | (a) staging G-01 of `self.buffer.length()` (rounded/reused, not logical size) `q.rs:45-48`; (b) blit + `flush_and_wait_current`; (c) host `Vec<f32>` of `elem_count` + block Vec; (d) upload G-05 of `elem_count*4` `q.rs:120-125` | Peak GPU = staging(rounded) + `4*elem_count` exact; host = `4*elem_count` + block bytes | Yes from `buffer.length()` and `elem_count` before the first build | Before `q.rs:43` (both GPU sizes known up front) or raw ctor per allocation | Staging: local `Arc` + pool; output: returned `MetalStorage` | Staging returns to pool on function exit; freed at next sweep | covered-by-single-hook / needs-patch (no GPU dequant kernel on this path) |
| G-11 | `q.rs:134-218` (`quantize`, `quantize_imatrix`, `*_onto`) | `to_cpu` readback (G-12) + host quantize + G-05 upload; old `self.buffer` replaced (`q.rs:147` etc.) | Exact quantized byte length | Yes once host data exists (after host quantize) | Before each `with_data` build or raw ctor | `QMetalStorage.buffer` | Old buffer returns to pool, freed at sweep | covered-by-single-hook / needs-patch |
| G-12 | `mod.rs:2003-2019` (`MetalStorage::to_cpu`), `q.rs:441-456` (`QMetalStorage::data`) | Readback staging via G-01 + blit + `flush_and_wait_current` | `count * dtype.size_in_bytes()` (storage count, not layout) then G-01 rounding/reuse | Yes | As G-01 | Local `Arc` + pool | Returned to pool on function exit; the same call's sweep runs before the local `Arc` is dropped, so it is freed at a LATER sweep | covered-by-single-hook / needs-patch |
| G-13 | `mk/kernels/sort.rs:58-83` (`multi_block_sort`), entry `mk/kernels/sort.rs:290-315` (`call_mlx_arg_sort`, public) | 5 raw shared buffers via `Device::new_buffer` directly: 2x `el_count*dtype`, 2x `el_count*4`, `nrows*(nblocks+1)*4`; not pooled, not in residency set | Exact, unchecked multiplies | Yes (computable from nrows/ncols/dtype before call) | Raw ctor only; caller-side check before `call_mlx_arg_sort` | Function-local `Buffer` values | Dropped at function return while the CB is uncommitted; backing lifetime then depends on CB retained references (unverified) | unsupported in experimental route; not reachable from candle-core (`candle-core/src/sort.rs:236` uses single-block `call_arg_sort`), reachable only from external callers |
| G-14 | `candle-core/src/sort.rs:173-253`, `mk/kernels/sort.rs:8-44` (`call_arg_sort`) | Output via G-02 (`sort.rs:228`); threadgroup memory `max(ncols_pad*4,16)` (`mk/kernels/sort.rs:41`), on-chip, not backing | `ncols_pad` = next power of two >= ncols, unbounded in code | Yes for G-02 part | As G-02; ncols limit must be a precondition (threadgroup limit failure mode unaudited) | As G-02 | As G-02 | covered-by-single-hook (G-02); threadgroup limit unaudited |
| G-15 | `mk/metal/commands.rs:20-24,101,160-229,294-315`; `mk/metal/fence.rs:18-25`; `mk/metal/encoder.rs:17,32,143` | Command buffers (`commandBuffer()`, retained-reference mode per Apple API, unverified), one `MTLFence` per encoder (`commands.rs:170,206`), `prev_ce_outputs` map, `in_flight` Vec | Up to `compute_per_buffer` encoders per CB (default 50, env `CANDLE_METAL_COMPUTE_PER_BUFFER`, `commands.rs:136-141`); number of in-flight CBs unbounded until a wait | No byte bound (driver objects, size not exposed) | None without patch; `Fence::new` panics on failure (`fence.rs:23`), encoder creation `unwrap()`s (`command_buffer.rs:26,37,48`) | `Commands.state.in_flight` | `in_flight` drained only in `flush_and_wait` (`commands.rs:241`), filtered to non-Completed in `flush_and_wait_current` (`commands.rs:279-281`); `flush` (`commands.rs:286`) and `Drop` (`commands.rs:391-395`) commit without waiting | unaudited (metadata); retained buffers: needs-patch |
| G-16 | `mk/metal/residency_set.rs:17-25,36-65`; inserts `dev.rs:216,236,255,299`, `mod.rs:2040`; removes only `dev.rs:111,122` and `dev.rs:184-186` | Residency set object + per-allocation membership | n/a | n/a | n/a | `MetalDevice.residency_set` (`dev.rs:68`) | Removal at sweep; G-06/G-07 never removed. Whether membership retains the MTLBuffer is unverified | unaudited |
| G-17 | `mk/kernel.rs:61-67,109-175` (`Kernels` library/pipeline caches) | `MTLLibrary` per `Source`, `MTLComputePipelineState` per `(Source, name, Option<ConstantValues>)` | Key space includes constant values and formatted kernel names (e.g. `mk/kernels/sort.rs:90`); no eviction | No | None without patch (`load_pipeline_with_constants` `kernel.rs:151` is the single insertion point) | `MetalDevice.kernels` (`dev.rs:62`) | Device drop only | unaudited |
| G-18 | `mk/metal/encoder.rs:151-160` (`setBytes`) | Inline argument bytes copied by the driver; no `MTLBuffer` | Per-call struct size | Not checked in Candle | n/a | Driver | CB completion (unverified) | unaudited |
| G-19 | `mod.rs:2024-2055` (`MetalDevice::new`), `mk/metal/commands.rs:132-158` | Command queue, residency set, `Kernels`, first CB, G-07 seed | Fixed object count | No byte bound | Before `Device::new_metal` (`candle-core/src/device.rs:258`); `Device::all().swap_remove(ordinal)` panics on bad ordinal (`mod.rs:2025`) | `MetalDevice` (all clones share `Arc`s) | Last `MetalDevice` clone dropped; `Commands::drop` flushes without waiting | unaudited (charge as fixed device overhead) |
| G-20 | `mod.rs:1907-1914` (`MetalStorage::new`, public); `dev.rs:102` (`metal_device`), `mk/metal/device.rs:39-43` (`AsRef<MTLDevice>`), `mk/metal/device.rs:73,86` (public raw ctors), `dev.rs:177-186` (`register_buffers`) | Foreign/no-copy import: any caller can wrap an externally created `Arc<Buffer>` (incl. `newBufferWithBytesNoCopy` via objc2) into a tensor | Arbitrary | No | Only by forbidding these APIs in the experimental route; a raw-ctor hook does not see objc2 calls made directly on the `MTLDevice` | Caller-defined | Caller-defined | unsupported |
| G-21 | `candle-core/src/custom_op.rs:26,73,129,269,327` (`metal_fwd` traits) | Third-party custom ops may allocate via any route above | Op-defined | Op-defined | Raw ctor hook covers only routes using `mk::metal::Device` | Op-defined | Op-defined | candle-nn ops covered via G-02; third-party unaudited |
| G-22 | `mod.rs:92-94` (`try_clone`), `candle-core/src/storage.rs:29-31` | No allocation: storage clone shares `Arc<Buffer>` (see research C-02) | 0 | n/a | n/a | Shared; no uniqueness | Last `Arc` + sweep | n/a (aliasing hazard for lease identity) |

## Single choke point analysis

- **Is there one function?** No. In `candle-core` there are four `MetalDevice` constructors
  (`new_buffer` `dev.rs:197`, `new_private_buffer` `dev.rs:224`, `new_buffer_with_data` `dev.rs:244`,
  `allocate_buffer` `dev.rs:285`), and the builder (`dev.rs:416`) does not reach G-06 or the seed
  (G-07, `mod.rs:2031`). All Candle-internal backing creations do bottom out in exactly two raw
  functions, `mk/metal/device.rs:73` (`new_buffer`) and `mk/metal/device.rs:86`
  (`new_buffer_with_data`), including sort scratch (G-13) and the seed. A fallible reservation
  check placed at the top of those two functions would deny before `newBufferWith*` is sent, would
  see the post-rounding size for pooled paths, and requires no change to `.metal` kernels. It still
  requires a source patch to `candle-metal-kernels` (no hook/callback API exists at this revision),
  and it does NOT cover: foreign/no-copy imports and direct objc2 calls on the public `MTLDevice`
  (G-20), command/fence/pipeline/residency metadata (G-15..G-17, G-19), or `setBytes` (G-18).
- **Lease identity at the raw hook:** the raw functions return a `Buffer` with no owner context;
  pooled paths wrap it in an `Arc` shared by the pool and tensors, and G-04 reuse re-leases the
  same backing without any constructor call. A per-tensor lease therefore cannot be attached at
  the raw hook; charging must follow pool capacity (pool is the only persistent owner).
- **Pool retention:** dropping a tensor only decrements the `Arc`; the pool `Vec` keeps the buffer
  (`dev.rs:217,256,300`). Capacity stays allocated and reusable until the sweep removes entries
  with `strong_count == 1` (`dev.rs:106-130`). The sweep runs only after a *successful*
  `wait_until_completed` / `flush_and_wait_current`; on a command-buffer error the `?` at
  `dev.rs:148/158` returns before sweeping, so capacity is retained (conservative). There is no
  periodic or pressure-driven trimming, no size cap and no `Drop` for the pools other than dropping
  the last `MetalDevice` clone. `in_flight` CBs are not part of `strong_count`, so "pool released" is
  not proof that the GPU or driver released the backing (CB retained-reference behaviour unverified).
- **Reuse during queued work:** a buffer whose tensor was dropped while queued GPU work still
  references it is reusable immediately (`strong_count==1`, `dev.rs:469`); ordering relies on the
  serial queue plus fences (`commands.rs:169-187`, untracked hazard mode `dev.rs:72-81`). For
  accounting this is harmless (capacity stays charged) but release must not be credited before
  completion.
- **Fallible vs panic:** raw allocation failure is a `Result` (`MetalKernelError::FailedToCreateResource`,
  `mk/metal/device.rs:81,98`) and propagates through `MetalError` to `candle::Error`. Panics/UB-like
  paths: unchecked `elem_count * size` (`dev.rs:203,230`, `q.rs:15`, `mk/kernels/sort.rs:73-83`);
  `next_power_of_two` overflow (`dev.rs:328`: debug panic, release returns 0); lock `unwrap()`s
  (`commands.rs:161,193,351`, `mod.rs:2297,2337,2351`); `Fence::new` `expect` (`fence.rs:23`);
  encoder creation `unwrap()` (`command_buffer.rs:26,37,48`); `unreachable!()` on unknown CB status
  (`commands.rs:328`); `swap_remove` on bad ordinal (`mod.rs:2025`). A reservation layer must
  pre-check arithmetic itself and treat these as outside the fallible contract.

## Gaps and decisive counterexamples

1. **Builder is not the choke point.** The RNG seed (`mod.rs:2031`) and `call_mlx_arg_sort` scratch
   (`mk/kernels/sort.rs:75-83`) call the raw `Device` directly; `new_private_buffer` (`dev.rs:224`)
   is outside the builder and the pools. A hook in `BufferBuilder::build` alone misses all three.
2. **Data uploads allocate before the pool lock and skip rounding** (`dev.rs:245-252`): a hook in
   `allocate_buffer` / `MetalDevice::new_buffer` misses every weight/host upload (G-05, G-09).
3. **Unbounded pooled slack.** `find_available_buffer` (`dev.rs:463-477`) may serve a tiny request
   from an arbitrarily large free buffer and exact-size upload buckets join the shared pool; charges
   keyed to logical tensor bytes undercount. Worst case pooled capacity is the sum of all buffers
   ever simultaneously live (rounded up to 2x for G-01/G-02) and persists until a successful sweep.
4. **No release without a successful wait.** Code that never calls `synchronize`/readback never
   sweeps; `Commands::flush`/`Drop` (`commands.rs:286,391`) do not wait, and `in_flight` grows.
5. **Readback/dequant staging lingers.** G-10/G-12 staging is returned to the pool after the call's
   own sweep has already run, so it is held until the next successful sweep. Dequantize staging is
   sized by `buffer.length()` (rounded), not by logical quantized bytes.
6. **Foreign imports are unbudgetable here** (G-20): `MetalStorage::new`, public `metal_device()` +
   `AsRef<MTLDevice>`, public raw ctors and `register_buffers` allow allocations no hook sees.
7. **Metadata has no byte bound** (G-15..G-19): CB/fence/pipeline/library/residency objects and
   `setBytes` are driver-managed with sizes not exposed by Candle; pipeline cache never evicts.
8. **Residency retention unverified.** Whether `MTLResidencySet` membership keeps a buffer alive is
   not established from source; G-06/G-07 are never removed (`dev.rs:236`, `mod.rs:2040`).
9. **Not audited:** `.metal` shader sources; objc2/objc2-metal crate internals; Apple driver
   behaviour (retained references, `newBufferWithLength` with length 0, residency, `setBytes`
   limits); third-party crates' custom `metal_fwd`; `candle-metal-kernels` examples/tests; CUDA/CPU
   backends. No claim in this file was exercised on hardware.

Conclusion for T117: a two-function raw-constructor patch can deny all Candle-internal `MTLBuffer`
creation before allocation with a pre-computed (post-rounding) size, but it cannot attach unique
per-tensor leases (pool aliasing/reuse), cannot bound command/pipeline metadata, and cannot see
foreign imports. G-06, G-13 and G-20 must be forbidden in the experimental route; G-15..G-19 stay
unaudited and keep the capability gate blocked until a defensible bound is recorded.
